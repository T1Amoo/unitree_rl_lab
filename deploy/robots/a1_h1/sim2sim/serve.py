"""A1 table-tennis serve sampler for MuJoCo sim2sim viewing."""

from __future__ import annotations

import numpy as np


SERVE_LAUNCH = np.array([1.35, 0.0, 1.03], dtype=np.float64)
G = 9.81
Z_BOUNCE = 0.78
NET_CENTER_CLEARANCE_Z = 0.95
GENERALIZED_BOUNCE_X = (-1.33, -0.03)
GENERALIZED_SERVE_Y_CENTER = 0.055
GENERALIZED_SERVE_Y_HALF = 0.200
# Match the explicit aerodynamic force used by run_a1_tt_sim2sim.py.
BALL_DRAG_ACCEL_K = 0.5 * 1.225 * (np.pi * 0.02**2) * 0.4378 / 0.0034
FLIGHT_DT = 0.002
# Final backhand-v1 hard-stage distribution (the last 10k consolidation stage).
# The next training version will additionally reject samples whose ball center
# does not clear the net; keep this sampler faithful to the already-trained v1.
BOUNCE_X = (-1.10, -0.35)
BOUNCE_VZ = (0.50, 2.00)
SERVE_Y_CENTER = 0.055
SERVE_Y_HALF = 0.105

# Direct launch-velocity ranges used only by the generalized MuJoCo probe.
# Rejection sampling below guarantees that all accepted balls clear the net and
# first bounce on the robot side.  The three equally likely bands deliberately
# span fast/low, medium and slow/high arrivals.
GENERALIZED_VELOCITY_BANDS = (
    ((-6.00, -4.00), (0.40, 2.40)),
    ((-4.50, -2.70), (1.00, 3.30)),
    ((-3.00, -1.60), (2.10, 4.30)),
)


def first_flight_metrics(velocity: np.ndarray) -> tuple[float, float, float] | None:
    """Integrate launch-to-first-bounce flight with the sim2sim drag model.

    Returns ``(net_center_z, first_bounce_x, first_bounce_time_s)``.  The table
    bounce itself is not integrated; this is the pre-bounce validity check.
    """

    pos = SERVE_LAUNCH.copy()
    vel = np.asarray(velocity, dtype=np.float64).reshape(3).copy()
    net_z: float | None = None
    for step in range(2500):
        prev = pos.copy()
        speed = float(np.linalg.norm(vel))
        vel += (np.array([0.0, 0.0, -G]) - BALL_DRAG_ACCEL_K * speed * vel) * FLIGHT_DT
        pos += vel * FLIGHT_DT
        if net_z is None and prev[0] > 0.0 >= pos[0]:
            alpha = prev[0] / (prev[0] - pos[0])
            net_z = float(prev[2] + alpha * (pos[2] - prev[2]))
        if prev[2] > Z_BOUNCE >= pos[2]:
            alpha = (prev[2] - Z_BOUNCE) / (prev[2] - pos[2])
            bounce_x = float(prev[0] + alpha * (pos[0] - prev[0]))
            bounce_t = float((step + alpha) * FLIGHT_DT)
            if net_z is None:
                return None
            return net_z, bounce_x, bounce_t
    return None


class Serve:
    def __init__(
        self,
        rng: np.random.Generator | None = None,
        interval_steps: int = 250,
        profile: str = "trained_v1",
    ):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.interval_steps = int(interval_steps)
        if profile not in {"trained_v1", "generalized"}:
            raise ValueError(f"unsupported serve profile: {profile}")
        self.profile = profile

    def due(self, control_step: int) -> bool:
        return control_step > 0 and control_step % self.interval_steps == 0

    def sample(self) -> tuple[np.ndarray, np.ndarray]:
        if self.profile == "generalized":
            return self._sample_generalized()

        x_b = self.rng.uniform(*BOUNCE_X)
        y_b = SERVE_Y_CENTER + self.rng.uniform(-SERVE_Y_HALF, SERVE_Y_HALF)
        vz = self.rng.uniform(*BOUNCE_VZ)
        t_b = (vz + np.sqrt(vz * vz + 2.0 * G * (SERVE_LAUNCH[2] - Z_BOUNCE))) / G
        vx = (x_b - SERVE_LAUNCH[0]) / t_b
        vy = y_b / t_b
        return SERVE_LAUNCH.copy(), np.array([vx, vy, vz], dtype=np.float64)

    def _sample_generalized(self) -> tuple[np.ndarray, np.ndarray]:
        for _ in range(256):
            band = GENERALIZED_VELOCITY_BANDS[int(self.rng.integers(len(GENERALIZED_VELOCITY_BANDS)))]
            vx = self.rng.uniform(*band[0])
            vz = self.rng.uniform(*band[1])
            metrics = first_flight_metrics(np.array([vx, 0.0, vz], dtype=np.float64))
            if metrics is None:
                continue
            net_z, bounce_x, bounce_t = metrics
            if net_z < NET_CENTER_CLEARANCE_Z or not (
                GENERALIZED_BOUNCE_X[0] <= bounce_x <= GENERALIZED_BOUNCE_X[1]
            ):
                continue
            y_b = GENERALIZED_SERVE_Y_CENTER + self.rng.uniform(
                -GENERALIZED_SERVE_Y_HALF, GENERALIZED_SERVE_Y_HALF
            )
            vy = y_b / bounce_t
            velocity = np.array([vx, vy, vz], dtype=np.float64)
            checked = first_flight_metrics(velocity)
            if checked is not None and checked[0] >= NET_CENTER_CLEARANCE_Z and (
                GENERALIZED_BOUNCE_X[0] <= checked[1] <= GENERALIZED_BOUNCE_X[1]
            ):
                return SERVE_LAUNCH.copy(), velocity
        raise RuntimeError("failed to sample a valid generalized serve after 256 attempts")
