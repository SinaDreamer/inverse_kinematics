# Ned2 DH Parameters

These are the standard (distal) Denavit–Hartenberg parameters I use to model the Niryo Ned2 in `ned2_ik.py`. They're the same values as the `DH` table at the top of that file.

## Convention

Each link transform is

    A_i = Rot_z(θ_i) · Trans_z(d_i) · Trans_x(a_i) · Rot_x(α_i)

with `θ_i = q_i + offset_i`, where `q_i` is the joint angle the robot reports. The offsets line up the DH zero pose with the robot's own zero pose. The full chain is `T06 = A1 · A2 · A3 · A4 · A5 · A6`, from the base frame to the tool flange.

## Table

| Joint *i* | a<sub>i</sub> [m] | α<sub>i</sub> [rad] | d<sub>i</sub> [m] | θ<sub>i</sub> [rad] |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 0      | +π/2 | 0.183 | q₁ |
| 2 | 0.221  | 0    | 0     | q₂ + π/2 |
| 3 | 0.0282 | +π/2 | 0     | q₃ |
| 4 | 0      | −π/2 | 0.235 | q₄ + π |
| 5 | 0      | +π/2 | 0     | q₅ |
| 6 | 0.035  | 0    | 0.060 | q₆ + π/2 |

What each row represents:

- **1:** base rotation; `d₁` is the height from the base frame to the shoulder axis.
- **2:** shoulder; `a₂` is the upper-arm length.
- **3:** elbow; `a₃` is the small elbow offset.
- **4:** forearm roll; `d₄` is the forearm length up to the wrist.
- **5:** wrist pitch.
- **6:** wrist roll; `d₆` and `a₆` are the offsets out to the tool flange.

## Joint limits

| Joint | q<sub>min</sub> [rad] | q<sub>max</sub> [rad] |
|:---:|:---:|:---:|
| 1 | −2.949 | 2.949 |
| 2 | −1.830 | 0.610 |
| 3 | −1.340 | 1.570 |
| 4 | −2.089 | 2.089 |
| 5 | −1.919 | 1.922 |
| 6 | −2.530 | 2.530 |

## Sanity check: zero configuration

At `q = [0, 0, 0, 0, 0, 0]` the model puts the flange at

    p = [0.295, 0.035, 0.4322] m

    R = [[0, 0, 1],
         [1, 0, 0],
         [0, 1, 0]]        (roll, pitch, yaw = π/2, 0, π/2)

That's the arm pointing straight up at the shoulder, with the forearm horizontal along +x. The z value is d₁ + a₂ + a₃ = 0.183 + 0.221 + 0.0282, and the x value is d₄ + d₆ = 0.235 + 0.060. You can reproduce it with:

```python
import numpy as np, ned2_ik as ik
print(ik.forward_kinematics(np.zeros(6)))
```

## Notes

- The parameters haven't been calibrated against a physical robot, so expect small (millimetre-level) differences on real hardware.
- The Jacobian built from this table matches finite differences of the forward kinematics to about 2e-7, so the FK and the Jacobian are consistent with each other.
- The axes of joints 4, 5 and 6 intersect (a₄ = a₅ = d₅ = 0); `a₆` and `d₆` only offset the flange from that wrist point.
