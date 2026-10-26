"""
Numerical inverse kinematics for the Niryo Ned2 (6-DOF).

Author:  Sina Keshavarz
Contact: SinaKeshavarz.eng@gmail.com

Forward kinematics come from a DH model of the arm, the Jacobian is the
geometric (z-axis / origin) one, and the solver is damped least squares with
a backtracking line search, adaptive damping and staged orientation:
position is solved first, orientation is switched on once the end effector
is within a few millimetres of the target.

Units: metres and radians throughout.
"""

import math
import numpy as np

# ----------------------------------------------------------------------------
# Solver settings
# ----------------------------------------------------------------------------
tol_pos = 1e-4                  # 0.1 mm
tol_ori = 1e-4                  # rad

# Damping (adaptive)
lam_init = 3e-3
lam_min = 1e-5
lam_max = 1e-1
lam_up = 2.0                    # multiply lambda when a step fails
lam_down = 1.5                  # divide lambda when a step succeeds

# Orientation staging
pos_for_orientation_mm = 3.0    # turn orientation on once |ep| < 3 mm
w_rot_far = 0.0                 # position-only stage
w_rot_near = 2.0                # position + orientation stage

# Step clamps (a crude trust region)
dq_comp_clamp = 0.3             # max per-joint step [rad]
dq_norm_clamp = 0.5             # max overall step norm [rad]

# Joint limits from the Ned2 datasheet [rad]
qmin = np.array([-2.949, -1.83, -1.34, -2.089, -1.919, -2.53])
qmax = np.array([2.949, 0.61, 1.57, 2.089, 1.922, 2.53])

# DH table: (a, alpha, d, joint offset). theta_i = q_i + offset_i
DH = [
    (0.0,     np.pi / 2,  0.183, 0.0),
    (0.221,   0.0,        0.0,   np.pi / 2),
    (0.0282,  np.pi / 2,  0.0,   0.0),
    (0.0,    -np.pi / 2,  0.235, np.pi),
    (0.0,     np.pi / 2,  0.0,   0.0),
    (0.035,   0.0,        0.06,  np.pi / 2),
]

_EPS = np.finfo(float).eps * 4.0


# ----------------------------------------------------------------------------
# Rotation helpers
# ----------------------------------------------------------------------------
def euler_to_rot_matrix(roll, pitch, yaw):
    """R = Rz(yaw) @ Ry(pitch) @ Rx(roll)"""
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def rot_matrix_to_euler(R):
    """Inverse of euler_to_rot_matrix (ZYX). Returns [roll, pitch, yaw]."""
    M = np.asarray(R, dtype=np.float64)[:3, :3]
    cy = math.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2)
    if cy > _EPS:
        ax = math.atan2(M[2, 1], M[2, 2])
        ay = math.atan2(-M[2, 0], cy)
        az = math.atan2(M[1, 0], M[0, 0])
    else:
        ax = math.atan2(-M[1, 2], M[1, 1])
        ay = math.atan2(-M[2, 0], cy)
        az = 0.0
    return [ax, ay, az]


def so3_log(R):
    """Rotation vector (axis * angle) of a rotation matrix."""
    tr = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(tr)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if theta < 1e-6:
        return 0.5 * w
    if np.pi - theta < 1e-3:
        # Near 180 deg sin(theta) -> 0 and the usual formula blows up.
        # Recover the axis from the symmetric part instead: R + I = 2 n n^T at theta = pi.
        B = 0.5 * (R + np.eye(3))
        k = int(np.argmax(np.diag(B)))
        n = B[:, k] / math.sqrt(max(B[k, k], 1e-12))
        if np.dot(n, w) < 0:        # keep the sign consistent with the skew part
            n = -n
        return theta * n / np.linalg.norm(n)
    return (theta / (2.0 * np.sin(theta))) * w


def orientation_error(R_current, R_desired):
    """e_r = Log(Rd @ Rc^T), expressed in the base frame (rad)."""
    return so3_log(R_desired @ R_current.T)


# ----------------------------------------------------------------------------
# Kinematics
# ----------------------------------------------------------------------------
def dh_transform(a, alpha, d, theta):
    """Standard DH homogeneous transform."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,      sa,       ca,      d],
        [0.0,     0.0,      0.0,    1.0],
    ])


def _frames(q):
    """Cumulative transforms [T00, T01, ..., T06]."""
    Ts = [np.eye(4)]
    for (a, alpha, d, offset), qi in zip(DH, q):
        Ts.append(Ts[-1] @ dh_transform(a, alpha, d, qi + offset))
    return Ts


def forward_kinematics(q):
    """4x4 base-to-flange transform T06."""
    return _frames(q)[-1]


def geometric_jacobian(q):
    """Return (J, T06), J being the 6x6 geometric Jacobian [Jv; Jw] in the base frame."""
    Ts = _frames(q)
    o_n = Ts[-1][:3, 3]
    J = np.zeros((6, 6))
    for i in range(6):
        z = Ts[i][:3, 2]
        o = Ts[i][:3, 3]
        J[:3, i] = np.cross(z, o_n - o)
        J[3:, i] = z
    return J, Ts[-1]


# ----------------------------------------------------------------------------
# IK solver
# ----------------------------------------------------------------------------
def inverse_kinematics(q0, pose, max_iterations=10000, verbose=True, seed=None):
    """
    Solve for joint angles that reach pose = [x, y, z, roll, pitch, yaw].

    Returns (q, info) where info has 'converged', 'iterations' and a
    'history' of (|ep|, |er|) per iteration, handy for plotting.
    """
    rng = np.random.default_rng(seed)
    q = np.clip(np.array(q0, dtype=float), qmin, qmax)
    p_des = np.array(pose[:3], dtype=float)
    R_des = euler_to_rot_matrix(*pose[3:])

    lam = lam_init
    step_size = 1.0
    stuck = 0
    ori_enabled = False
    history = []

    for it in range(1, max_iterations + 1):
        J, T = geometric_jacobian(q)
        e_p = p_des - T[:3, 3]
        e_r = orientation_error(T[:3, :3], R_des)
        n_ep, n_er = np.linalg.norm(e_p), np.linalg.norm(e_r)
        history.append((n_ep, n_er))

        if n_ep < tol_pos and n_er < tol_ori:
            if verbose:
                print(f"Converged in {it} iterations: |ep| = {n_ep:.2e} m, |er| = {n_er:.2e} rad")
            return q, {"converged": True, "iterations": it, "history": np.array(history)}

        # Orientation is switched on once position is close, and then stays on.
        # Letting it switch back off makes the solver bounce around the threshold.
        if not ori_enabled and n_ep < pos_for_orientation_mm * 1e-3:
            ori_enabled = True
            step_size = 1.0
        w_rot = w_rot_near if ori_enabled else w_rot_far

        # Weighted DLS: (W J) dq = W e. The weight has to go on the Jacobian rows
        # too, otherwise the solver is asked to rotate w_rot times the real error.
        W = np.diag([1.0, 1.0, 1.0, w_rot, w_rot, w_rot])
        e = W @ np.hstack([e_p, e_r])
        Jw = W @ J
        dq = Jw.T @ np.linalg.solve(Jw @ Jw.T + lam ** 2 * np.eye(6), e)

        dq = np.clip(dq, -dq_comp_clamp, dq_comp_clamp)
        nrm = np.linalg.norm(dq)
        if nrm > dq_norm_clamp:
            dq *= dq_norm_clamp / nrm

        # Backtracking line search on the same weighted error
        base = np.linalg.norm(e)
        step = step_size
        improved = False
        for _ in range(8):
            q_try = np.clip(q + step * dq, qmin, qmax)
            T_try = forward_kinematics(q_try)
            e_try = W @ np.hstack([p_des - T_try[:3, 3], orientation_error(T_try[:3, :3], R_des)])
            if np.linalg.norm(e_try) < base * 0.999:
                q = q_try
                improved = True
                break
            step *= 0.5

        # Adaptive step: grow 10% after a good step (capped at 1), halve after a bad one
        if improved:
            step_size = min(step * 1.1, 1.0)
            lam = max(lam / lam_down, lam_min)
            stuck = 0
        else:
            step_size = max(step_size * 0.5, 1e-2)
            lam = min(lam * lam_up, lam_max)
            stuck += 1
            if stuck % 25 == 0:     # small kick when stuck
                q = np.clip(q + 1e-3 * rng.uniform(-1, 1, 6), qmin, qmax)

        if verbose and it % 50 == 0:
            s = np.linalg.svd(J, compute_uv=False)
            print(f"it {it:5d} | |ep| {n_ep:.3e} | |er| {n_er:.3e} | "
                  f"cond(J) {s[0] / max(s[-1], 1e-9):7.1f} | step {step_size:.3f}")

    if verbose:
        print("Hit the iteration limit without converging.")
    return q, {"converged": False, "iterations": max_iterations, "history": np.array(history)}


if __name__ == "__main__":
    q0 = [0, 0, 0, 0, 0, 0]
    target = [0.2, -0.05, 0.5, 0.2, 0.1, -0.1]     # [x, y, z, roll, pitch, yaw]

    q, info = inverse_kinematics(q0, target)
    T = forward_kinematics(q)
    np.set_printoptions(precision=6, suppress=True)
    print("q*            :", q)
    print("target pos    :", np.array(target[:3]))
    print("reached pos   :", T[:3, 3])
    print("target rpy    :", np.array(target[3:]))
    print("reached rpy   :", np.array(rot_matrix_to_euler(T[:3, :3])))
