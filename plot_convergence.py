"""Plot |ep| and |er| over the iterations for one IK solve and save it as a PNG."""

import matplotlib.pyplot as plt
import numpy as np

import ned2_ik as ik

q0 = [0, 0, 0, 0, 0, 0]
target = [0.2, -0.05, 0.5, 0.2, 0.1, -0.1]

q, info = ik.inverse_kinematics(q0, target, verbose=False)
h = info["history"]
it = np.arange(1, len(h) + 1)

fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogy(it, h[:, 0], label="position error |ep| [m]")
ax.semilogy(it, h[:, 1], label="orientation error |er| [rad]")
ax.axhline(ik.tol_pos, color="gray", ls=":", lw=1)
ax.axhline(ik.tol_ori, color="gray", ls=":", lw=1)

on = np.argmax(h[:, 0] < ik.pos_for_orientation_mm * 1e-3)
ax.axvline(on + 1, color="gray", ls="--", lw=1)
ax.text(on + 1, ax.get_ylim()[1], "  orientation on", va="top", fontsize=8, color="gray")

ax.set_xlabel("iteration")
ax.set_ylabel("error norm")
ax.set_title(f"IK convergence, target {target}", fontsize=9)
ax.legend()
fig.tight_layout()
fig.savefig("convergence.png", dpi=150)
print(f"converged={info['converged']} in {info['iterations']} iterations -> convergence.png")
