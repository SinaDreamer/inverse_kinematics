"""
Convergence benchmark: sample random joint configurations inside the limits,
take their FK pose as the target (so every target is reachable), and see how
often the solver gets there from the zero configuration.

    python benchmark.py                 # 40 poses, 2000 iterations each
    python benchmark.py -n 100 --iters 10000
"""

import argparse

import numpy as np

import ned2_ik as ik


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=40, help="number of random targets")
    parser.add_argument("--iters", type=int, default=2000, help="iteration limit per solve")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    converged, iters, failures = 0, [], []

    for k in range(args.n):
        q_target = rng.uniform(ik.qmin * 0.9, ik.qmax * 0.9)
        T = ik.forward_kinematics(q_target)
        pose = list(T[:3, 3]) + ik.rot_matrix_to_euler(T[:3, :3])

        q, info = ik.inverse_kinematics(np.zeros(6), pose, args.iters, verbose=False, seed=k)
        if info["converged"]:
            converged += 1
            iters.append(info["iterations"])
        else:
            ep, er = info["history"][-1]
            failures.append((k, ep, er))

    print(f"converged {converged}/{args.n}")
    if iters:
        print(f"iterations (converged runs): median {int(np.median(iters))}, max {max(iters)}")
    for k, ep, er in failures:
        print(f"  pose {k:3d}: |ep| = {ep * 1000:7.2f} mm, |er| = {er:.3f} rad")


if __name__ == "__main__":
    main()
