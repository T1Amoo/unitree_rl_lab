"""A1 table-tennis serve sampler for MuJoCo sim2sim viewing.

``trained_v6`` mirrors the translational serve proposal and drag-aware
rejection contract used by the backhand-v6 Isaac training task.  The legacy
profiles remain available for explicit historical/edge-case playback.
"""

from __future__ import annotations

import numpy as np


SERVE_LAUNCH = np.array([1.35, 0.0, 1.03], dtype=np.float64)
G = 9.81
Z_BOUNCE = 0.78
NET_CENTER_CLEARANCE_Z = 0.95
GENERALIZED_BOUNCE_X = (-1.33, -0.03)
GENERALIZED_SERVE_Y_CENTER = 0.055
GENERALIZED_SERVE_Y_HALF = 0.200
HIT_PLANE_X = -1.243
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

# Backhand-v6 inherits the v4 translational proposal.  Sampling a curriculum
# value per serve exposes the viewer to the complete easy-to-hard support
# instead of freezing it at one curriculum boundary.  The hard slow/high tail
# is deliberately a minority, matching training after rejection rather than
# making high balls dominate the visualization.
V6_EASY_BOUNCE_X = (-0.80, -0.45)
V6_EASY_BOUNCE_VZ = (1.60, 2.60)
V6_EASY_Y_CENTER = 0.050
V6_EASY_Y_HALF = 0.050
V6_HARD_BOUNCE_X = (-0.95, -0.75)
V6_HARD_BOUNCE_VZ = (1.20, 2.30)
V6_HARD_Y_CENTER = 0.020
V6_HARD_Y_HALF = 0.180
V6_TAIL_CANDIDATE_WEIGHT_HARD = 0.45
V6_TAIL_BOUNCE_X_HARD = (-0.75, -0.08)
V6_TAIL_BOUNCE_VZ_HARD = (3.00, 3.80)

# This is the same reset-time acceptance window as v6 training.  The net
# threshold already contains ball radius and prediction margin.
V6_PHYSICAL_BOUNCE_X = (-1.18, -0.04)
V6_HIT_Y = (-0.22, 0.26)
V6_HIT_Z = (0.88, 1.32)
V6_HIT_ABS_VX = (1.0, 5.8)
V6_TABLE_RESTITUTION = 0.95
V6_TABLE_DYNAMIC_FRICTION = 0.10
V6_MAX_ATTEMPTS = 256

# Coverage-oriented viewer profile.  Each stratum is still a subset of the v6
# training contract, but selecting the stratum before rejection prevents rare
# low/slow/high arrivals from disappearing behind the dominant medium cluster.
V6_GENERALIZED_Y_CENTER = 0.020
V6_GENERALIZED_Y_HALF = 0.180
V6_GENERALIZED_MAX_ATTEMPTS = 64
V6_GENERALIZED_STRATA = (
    {
        "name": "low_fast",
        "weight": 0.08,
        "bounce_x": (-0.95, -0.85),
        "bounce_vz": (1.20, 1.40),
        "hit_z": (0.88, 1.00),
        "hit_abs_vx": (3.2, 5.8),
    },
    {
        "name": "mid_fast",
        "weight": 0.12,
        "bounce_x": (-0.95, -0.75),
        "bounce_vz": (1.25, 1.80),
        "hit_z": (1.00, 1.16),
        "hit_abs_vx": (3.2, 5.8),
    },
    {
        "name": "mid_medium",
        "weight": 0.55,
        "bounce_x": (-0.95, -0.75),
        "bounce_vz": (1.65, 2.28),
        "hit_z": (1.00, 1.16),
        "hit_abs_vx": (1.8, 3.2),
    },
    {
        "name": "high_medium",
        "weight": 0.10,
        "bounce_x": (-0.90, -0.75),
        "bounce_vz": (2.20, 2.35),
        "hit_z": (1.16, 1.32),
        "hit_abs_vx": (1.8, 3.2),
    },
    {
        "name": "low_slow",
        "weight": 0.02,
        "bounce_x": (-0.75, -0.56),
        "bounce_vz": (3.00, 3.75),
        "hit_z": (0.88, 1.00),
        "hit_abs_vx": (1.0, 1.8),
    },
    {
        "name": "mid_slow",
        "weight": 0.05,
        "bounce_x": (-0.75, -0.61),
        "bounce_vz": (3.00, 3.62),
        "hit_z": (1.00, 1.16),
        "hit_abs_vx": (1.0, 1.8),
    },
    {
        "name": "high_slow",
        "weight": 0.08,
        "bounce_x": (-0.75, -0.69),
        "bounce_vz": (3.00, 3.38),
        "hit_z": (1.16, 1.32),
        "hit_abs_vx": (1.0, 1.8),
    },
)

# Direct launch-velocity ranges used only by the generalized MuJoCo probe.
# Rejection sampling below guarantees that all accepted balls clear the net and
# first bounce on the robot side.  The three equally likely bands deliberately
# span fast/low, medium and slow/high arrivals.
GENERALIZED_VELOCITY_BANDS = (
    ((-6.00, -4.00), (0.40, 2.40)),
    ((-4.50, -2.70), (1.00, 3.30)),
    ((-3.00, -1.60), (2.10, 4.30)),
)


def _flight_rhs_x(state: np.ndarray) -> np.ndarray:
    """Return d[y,z,vx,vy,vz,t]/dx for the training drag model."""

    _y, _z, vx, vy, vz, _t = state
    safe_vx = min(float(vx), -1.0e-4)
    speed = float(np.sqrt(max(vx * vx + vy * vy + vz * vz, 1.0e-12)))
    return np.array(
        [
            vy / safe_vx,
            vz / safe_vx,
            -BALL_DRAG_ACCEL_K * speed,
            (-BALL_DRAG_ACCEL_K * speed * vy) / safe_vx,
            (-G - BALL_DRAG_ACCEL_K * speed * vz) / safe_vx,
            1.0 / safe_vx,
        ],
        dtype=np.float64,
    )


def _rk4_x(state: np.ndarray, dx: float) -> np.ndarray:
    k1 = _flight_rhs_x(state)
    k2 = _flight_rhs_x(state + 0.5 * dx * k1)
    k3 = _flight_rhs_x(state + 0.5 * dx * k2)
    k4 = _flight_rhs_x(state + dx * k3)
    return state + (dx / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def v6_contract_metrics(velocity: np.ndarray) -> dict[str, float] | None:
    """Propagate one launch through the v6 net/bounce/hit-plane contract."""

    velocity = np.asarray(velocity, dtype=np.float64).reshape(3)
    state = np.array(
        [
            SERVE_LAUNCH[1],
            SERVE_LAUNCH[2],
            velocity[0],
            velocity[1],
            velocity[2],
            0.0,
        ],
        dtype=np.float64,
    )
    x = float(SERVE_LAUNCH[0])
    dx_net = (0.0 - x) / 8.0
    for _ in range(8):
        state = _rk4_x(state, dx_net)
        x += dx_net
    net_z = float(state[1])

    bounced = False
    bounce_x = float("nan")
    dx_hit = (HIT_PLANE_X - 0.0) / 16.0
    for _ in range(16):
        previous = state.copy()
        following = _rk4_x(previous, dx_hit)
        crossing = (
            not bounced
            and previous[1] > Z_BOUNCE >= following[1]
            and following[4] < 0.0
        )
        if crossing:
            denom = max(float(previous[1] - following[1]), 1.0e-8)
            alpha = float(np.clip((previous[1] - Z_BOUNCE) / denom, 0.0, 1.0))
            pre_bounce = previous + alpha * (following - previous)
            post_bounce = pre_bounce.copy()
            post_bounce[1] = Z_BOUNCE
            tangent = pre_bounce[[2, 3]]
            tangent_speed = float(np.linalg.norm(tangent))
            coulomb_delta = (
                V6_TABLE_DYNAMIC_FRICTION
                * (1.0 + V6_TABLE_RESTITUTION)
                * abs(float(pre_bounce[4]))
            )
            delta_speed = min(coulomb_delta, tangent_speed)
            tangent_scale = max(
                0.0,
                1.0 - delta_speed / max(tangent_speed, 1.0e-8),
            )
            post_bounce[2] = pre_bounce[2] * tangent_scale
            post_bounce[3] = pre_bounce[3] * tangent_scale
            post_bounce[4] = -V6_TABLE_RESTITUTION * pre_bounce[4]
            state = _rk4_x(post_bounce, dx_hit * (1.0 - alpha))
            bounce_x = x + alpha * dx_hit
            bounced = True
        else:
            state = following
        x += dx_hit

    if not bounced or not np.isfinite(state).all():
        return None
    return {
        "net_z": net_z,
        "bounce_x": float(bounce_x),
        "hit_y": float(state[0]),
        "hit_z": float(state[1]),
        "hit_vx": float(state[2]),
        "hit_vz": float(state[4]),
        "hit_time": float(state[5]),
    }


def _inside(value: float, bounds: tuple[float, float]) -> bool:
    return bounds[0] <= value <= bounds[1]


def _v6_contract_accepts(metrics: dict[str, float] | None) -> bool:
    return bool(
        metrics is not None
        and metrics["net_z"] >= NET_CENTER_CLEARANCE_Z
        and _inside(metrics["bounce_x"], V6_PHYSICAL_BOUNCE_X)
        and _inside(metrics["hit_y"], V6_HIT_Y)
        and _inside(metrics["hit_z"], V6_HIT_Z)
        and _inside(abs(metrics["hit_vx"]), V6_HIT_ABS_VX)
    )


def _lerp_pair(
    easy: tuple[float, float],
    hard: tuple[float, float],
    curriculum: float,
) -> tuple[float, float]:
    return (
        easy[0] + curriculum * (hard[0] - easy[0]),
        easy[1] + curriculum * (hard[1] - easy[1]),
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
        profile: str = "v6_generalized",
    ):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.interval_steps = int(interval_steps)
        if profile not in {
            "trained_v1",
            "trained_v6",
            "v6_generalized",
            "generalized",
        }:
            raise ValueError(f"unsupported serve profile: {profile}")
        self.profile = profile
        self.last_stratum = ""
        self.last_attempts = 0

    def due(self, control_step: int) -> bool:
        return control_step > 0 and control_step % self.interval_steps == 0

    def sample(self) -> tuple[np.ndarray, np.ndarray]:
        if self.profile == "v6_generalized":
            return self._sample_v6_generalized()
        if self.profile == "trained_v6":
            return self._sample_trained_v6()
        if self.profile == "generalized":
            return self._sample_generalized()

        x_b = self.rng.uniform(*BOUNCE_X)
        y_b = SERVE_Y_CENTER + self.rng.uniform(-SERVE_Y_HALF, SERVE_Y_HALF)
        vz = self.rng.uniform(*BOUNCE_VZ)
        t_b = (vz + np.sqrt(vz * vz + 2.0 * G * (SERVE_LAUNCH[2] - Z_BOUNCE))) / G
        vx = (x_b - SERVE_LAUNCH[0]) / t_b
        vy = y_b / t_b
        self.last_stratum = "trained_v1"
        self.last_attempts = 1
        return SERVE_LAUNCH.copy(), np.array([vx, vy, vz], dtype=np.float64)

    def _sample_bounce_box(
        self,
        x_range: tuple[float, float],
        vz_range: tuple[float, float],
        y_center: float,
        y_half: float,
    ) -> np.ndarray:
        x_b = float(self.rng.uniform(*x_range))
        y_b = y_center + float(self.rng.uniform(-y_half, y_half))
        vz = float(self.rng.uniform(*vz_range))
        t_b = (vz + np.sqrt(vz * vz + 2.0 * G * (SERVE_LAUNCH[2] - Z_BOUNCE))) / G
        return np.array(
            [
                (x_b - SERVE_LAUNCH[0]) / t_b,
                (y_b - SERVE_LAUNCH[1]) / t_b,
                vz,
            ],
            dtype=np.float64,
        )

    def _sample_trained_v6(self) -> tuple[np.ndarray, np.ndarray]:
        for _ in range(V6_MAX_ATTEMPTS):
            curriculum = float(self.rng.uniform(0.0, 1.0))
            x_range = _lerp_pair(
                V6_EASY_BOUNCE_X,
                V6_HARD_BOUNCE_X,
                curriculum,
            )
            vz_range = _lerp_pair(
                V6_EASY_BOUNCE_VZ,
                V6_HARD_BOUNCE_VZ,
                curriculum,
            )
            if self.rng.random() < curriculum * V6_TAIL_CANDIDATE_WEIGHT_HARD:
                x_range = _lerp_pair(
                    V6_EASY_BOUNCE_X,
                    V6_TAIL_BOUNCE_X_HARD,
                    curriculum,
                )
                vz_range = _lerp_pair(
                    V6_EASY_BOUNCE_VZ,
                    V6_TAIL_BOUNCE_VZ_HARD,
                    curriculum,
                )

            y_center = V6_EASY_Y_CENTER + curriculum * (
                V6_HARD_Y_CENTER - V6_EASY_Y_CENTER
            )
            y_half = V6_EASY_Y_HALF + curriculum * (
                V6_HARD_Y_HALF - V6_EASY_Y_HALF
            )
            velocity = self._sample_bounce_box(
                x_range,
                vz_range,
                y_center,
                y_half,
            )
            if _v6_contract_accepts(v6_contract_metrics(velocity)):
                self.last_stratum = "natural_v6"
                self.last_attempts = _ + 1
                return SERVE_LAUNCH.copy(), velocity
        raise RuntimeError(
            f"failed to sample a valid trained_v6 serve after {V6_MAX_ATTEMPTS} attempts"
        )

    def _sample_v6_generalized(self) -> tuple[np.ndarray, np.ndarray]:
        weights = np.asarray(
            [float(stratum["weight"]) for stratum in V6_GENERALIZED_STRATA],
            dtype=np.float64,
        )
        stratum = V6_GENERALIZED_STRATA[int(self.rng.choice(len(weights), p=weights))]
        for attempt in range(1, V6_GENERALIZED_MAX_ATTEMPTS + 1):
            velocity = self._sample_bounce_box(
                stratum["bounce_x"],
                stratum["bounce_vz"],
                V6_GENERALIZED_Y_CENTER,
                V6_GENERALIZED_Y_HALF,
            )
            metrics = v6_contract_metrics(velocity)
            if not _v6_contract_accepts(metrics):
                continue
            assert metrics is not None
            if not _inside(metrics["hit_z"], stratum["hit_z"]):
                continue
            if not _inside(abs(metrics["hit_vx"]), stratum["hit_abs_vx"]):
                continue
            self.last_stratum = str(stratum["name"])
            self.last_attempts = attempt
            return SERVE_LAUNCH.copy(), velocity
        raise RuntimeError(
            "failed to sample v6_generalized stratum "
            f"{stratum['name']} after {V6_GENERALIZED_MAX_ATTEMPTS} attempts"
        )

    def _sample_generalized(self) -> tuple[np.ndarray, np.ndarray]:
        for attempt in range(1, 257):
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
                self.last_stratum = "legacy_generalized"
                self.last_attempts = attempt
                return SERVE_LAUNCH.copy(), velocity
        raise RuntimeError("failed to sample a valid generalized serve after 256 attempts")
