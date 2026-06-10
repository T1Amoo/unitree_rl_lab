"""Ball serve controller for the G1 table-tennis sim2sim scene.

RALLY serve (mirrors training tt_env.reset_ball bounce mode): sample a TARGET
BOUNCE POINT in the robot's own half and back-compute the launch velocity so the
ball ALWAYS bounces in-court (proper table tennis; no volleys), then the policy
returns it. Matches g1_tt_rally training (serve_bounce_enable=True).
"""
import numpy as np

# Scene geometry: table top z=0.76, x in [-1.37, 1.37], net at x=0 (top z=0.915);
# robot pelvis at x=-1.6 facing +x -> robot half is x in [-1.37, 0].
# Launch from ball default (1.35, 0, 1.03); ball center bounces at z=0.78 (top+radius).
SERVE_LAUNCH = (1.35, 0.0, 1.03)
G = 9.81
Z_BOUNCE = 0.78
# Target-bounce-point ranges. EASY (like yesterday): centered, mid-court, narrow.
# (Policy v3 was trained on the full deep+wide distribution, so these easy serves
#  are a subset it handles comfortably. Widen BOUNCE_X / BOUNCE_Y for harder play.)
BOUNCE_X = (-0.65, -0.45)   # mid court, narrow (bounce ~x=-0.55)
BOUNCE_VZ = (1.6, 1.8)     # gentle arc
BOUNCE_Y = (-0.10, 0.10)   # centered (almost straight)


class Serve:
    def __init__(self, rng=None, interval_steps=150):
        # interval_steps * control_dt(0.02) = serve period; 150 -> 3 s
        self.rng = rng if rng is not None else np.random.default_rng()
        self.interval_steps = interval_steps

    def due(self, step):
        return step > 0 and step % self.interval_steps == 0

    def sample(self):
        x0, _, z0 = SERVE_LAUNCH
        x_b = self.rng.uniform(*BOUNCE_X)
        y_b = self.rng.uniform(*BOUNCE_Y)
        vz = self.rng.uniform(*BOUNCE_VZ)
        # time to bounce (descending root of z0 + vz t - 0.5 g t^2 = Z_BOUNCE)
        t_b = (vz + np.sqrt(vz * vz + 2.0 * G * (z0 - Z_BOUNCE))) / G
        vx = (x_b - x0) / t_b
        vy = y_b / t_b   # launch y = 0
        pos = np.array(SERVE_LAUNCH, dtype=np.float64)
        vel = np.array([vx, vy, vz], dtype=np.float64)
        return pos, vel
