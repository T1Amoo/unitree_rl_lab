"""A1 table-tennis serve sampler for MuJoCo sim2sim viewing."""

from __future__ import annotations

import numpy as np


SERVE_LAUNCH = np.array([1.35, 0.0, 1.03], dtype=np.float64)
G = 9.81
Z_BOUNCE = 0.78
# Final backhand-v1 hard-stage distribution (the last 10k consolidation stage).
# The next training version will additionally reject samples whose ball center
# does not clear the net; keep this sampler faithful to the already-trained v1.
BOUNCE_X = (-1.10, -0.35)
BOUNCE_VZ = (0.50, 2.00)
SERVE_Y_CENTER = 0.055
SERVE_Y_HALF = 0.105


class Serve:
    def __init__(self, rng: np.random.Generator | None = None, interval_steps: int = 250):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.interval_steps = int(interval_steps)

    def due(self, control_step: int) -> bool:
        return control_step > 0 and control_step % self.interval_steps == 0

    def sample(self) -> tuple[np.ndarray, np.ndarray]:
        x_b = self.rng.uniform(*BOUNCE_X)
        y_b = SERVE_Y_CENTER + self.rng.uniform(-SERVE_Y_HALF, SERVE_Y_HALF)
        vz = self.rng.uniform(*BOUNCE_VZ)
        t_b = (vz + np.sqrt(vz * vz + 2.0 * G * (SERVE_LAUNCH[2] - Z_BOUNCE))) / G
        vx = (x_b - SERVE_LAUNCH[0]) / t_b
        vy = y_b / t_b
        return SERVE_LAUNCH.copy(), np.array([vx, vy, vz], dtype=np.float64)
