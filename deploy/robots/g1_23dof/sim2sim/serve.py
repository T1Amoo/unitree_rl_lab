"""Ball serve controller for the G1 table-tennis sim2sim scene.

RALLY serve (mirrors training tt_env.reset_ball bounce mode): sample a TARGET
BOUNCE POINT in the robot's own half and back-compute the launch velocity so the
ball ALWAYS bounces in-court (proper table tennis; no volleys), then the policy
returns it. Matches g1_tt_rally training (serve_bounce_enable=True).
"""
import numpy as np
import os

# Scene geometry: table top z=0.76, x in [-1.37, 1.37], net at x=0 (top z=0.915);
# robot pelvis at x=-2.0 facing +x (v7 station) -> robot half is x in [-1.37, 0].
# Launch from ball default (1.35, 0, 1.03); ball center bounces at z=0.78 (top+radius).
SERVE_LAUNCH = (1.35, 0.0, 1.03)
G = 9.81
Z_BOUNCE = 0.78
# v11-MATCHED serve (2026-06-26): robot moved -2.0 -> -1.8, serve re-tuned (vz dropped ~0.2) so the
# ball bounces DEEP near the table edge (-1.37) and reaches the robot at -1.8 at racket-ready height
# (~z1.0). Mirrors g1_tt_v11 training easy serve: bounce_x(-1.37,-1.31) vz(1.5,1.8). mujoco now has
# air drag (tt_sim_mujoco) matching training, so the same vz reaches -1.8 as in sim.
# TEMP: HIGH-ball test (2026-06-29) — serve consistently to arrival z~1.15-1.17 at -1.8 (the high
# end of the v13 curriculum) to eyeball whether 1.17 is too high for the robot to hit cleanly.
BOUNCE_X = (-1.10, -1.05)   # deep bounce -> high arrival
BOUNCE_VZ = (2.2, 2.5)     # -> z ~1.14-1.18 at -1.8
BOUNCE_Y = (-0.30, 0.30)   # near-center so height is the only variable
BOUNCE_Y_MAG = (0.55, 0.72)  # alternating wide corners (>= training hard 0.72)
# TT_SERVE_MID=1: easy MID-table serve for VIEWING — bounces mid-court so the post-bounce arc to
# the robot at -2.0 is longer and easier to predict (the deep v7 serve bounces right in front of
# the robot, little time to react). mujoco has no air drag so the higher arc still carries to -2.0.
if os.environ.get("TT_SERVE_MID", "0") == "1":
    BOUNCE_X = (-0.95, -0.75)



class Serve:
    def __init__(self, rng=None, interval_steps=150):
        # interval_steps * control_dt(0.02) = serve period; 150 -> 3 s
        self.rng = rng if rng is not None else np.random.default_rng()
        self.interval_steps = interval_steps
        self._serve_idx = 0   # for TT_SERVE_ALTERNATE left/right toggling

    def due(self, step):
        return step > 0 and step % self.interval_steps == 0

    def sample(self):
        # Mix in INVALID/junk serves (TT_INVALID_FRAC, default 0.4) to test robustness: the policy
        # should recognize them as un-hittable (tracker disengages -> HOME-anchor idle) and stay
        # STABLE at the ready stance, NOT chase/flail/fall. Set TT_INVALID_FRAC=0 for all-valid.
        p_inv = float(os.environ.get("TT_INVALID_FRAC", "0.4"))
        if self.rng.random() < p_inv:
            return self._invalid_serve()
        return self._normal_serve()

    def _normal_serve(self):
        x0, _, z0 = SERVE_LAUNCH
        x_b = self.rng.uniform(*BOUNCE_X)
        if os.environ.get("TT_SERVE_ALTERNATE", "0") == "1":
            # alternate LEFT (+y) / RIGHT (-y) wide corners each serve
            side = 1.0 if (self._serve_idx % 2 == 0) else -1.0
            self._serve_idx += 1
            y_b = side * self.rng.uniform(*BOUNCE_Y_MAG)
        else:
            y_b = self.rng.uniform(*BOUNCE_Y)
        vz = self.rng.uniform(*BOUNCE_VZ)
        # time to bounce (descending root of z0 + vz t - 0.5 g t^2 = Z_BOUNCE)
        t_b = (vz + np.sqrt(vz * vz + 2.0 * G * (z0 - Z_BOUNCE))) / G
        vx = (x_b - x0) / t_b
        vy = y_b / t_b   # launch y = 0
        return np.array(SERVE_LAUNCH, dtype=np.float64), np.array([vx, vy, vz], dtype=np.float64)

    def _invalid_serve(self):
        """Degenerate / un-returnable serves the robot should IGNORE (stay stable, don't chase)."""
        kind = int(self.rng.integers(0, 4))
        if kind == 0:    # ROLL: launch just above the table, near-flat -> bounces low and rolls in
            pos = np.array([1.35, self.rng.uniform(-0.3, 0.3), 0.80], dtype=np.float64)
            vel = np.array([self.rng.uniform(-2.5, -1.5), self.rng.uniform(-0.3, 0.3), self.rng.uniform(-0.1, 0.2)], dtype=np.float64)
        elif kind == 1:  # LONG OUT: fast & flat -> flies past the robot, lands behind it (out of court)
            pos = np.array(SERVE_LAUNCH, dtype=np.float64)
            vel = np.array([self.rng.uniform(-6.0, -5.0), self.rng.uniform(-0.4, 0.4), self.rng.uniform(1.2, 1.8)], dtype=np.float64)
        elif kind == 2:  # WIDE OUT: large lateral -> ball goes way off the side, out of reach
            pos = np.array(SERVE_LAUNCH, dtype=np.float64)
            vel = np.array([self.rng.uniform(-3.0, -2.0), self.rng.uniform(2.5, 3.5) * float(self.rng.choice([-1.0, 1.0])), self.rng.uniform(1.5, 2.2)], dtype=np.float64)
        else:            # SHORT / NET: too weak -> drops short / into the net, never reaches robot
            pos = np.array(SERVE_LAUNCH, dtype=np.float64)
            vel = np.array([self.rng.uniform(-1.2, -0.6), self.rng.uniform(-0.2, 0.2), self.rng.uniform(0.6, 1.0)], dtype=np.float64)
        return pos, vel
