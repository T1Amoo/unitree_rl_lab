"""Ball serve controller for the G1 table-tennis sim2sim scene.

Mirrors training `reset_ball` (Pingpong_TTRL tt_env.reset_ball) with the eval
serving ranges from g1_tt_config.G1TT_EvalEnvCfg.
"""
import numpy as np

# env-local ball default (table frame); matches BALL_CFG init_state.pos
SERVE_DEFAULT_POS = (1.35, 0.0, 1.03)
SERVE_POS_Y_RANGE = (-0.1, 0.1)      # ball.ball_pos_y_range
# TRAINING serve params (base tt_env_config) — restored for in-distribution comparison.
SERVE_VEL_X = (-6.5, -5.0)           # ball.ball_speed_x_range
SERVE_VEL_Y = (-0.8, 0.4)            # ball.ball_speed_y_range
SERVE_VEL_Z = (1.5, 2.0)             # ball.ball_speed_z_range


class Serve:
    def __init__(self, rng=None, interval_steps=150):
        # interval_steps * control_dt(0.02) = serve period; 150 -> 3 s
        self.rng = rng if rng is not None else np.random.default_rng()
        self.interval_steps = interval_steps

    def due(self, step):
        return step > 0 and step % self.interval_steps == 0

    def sample(self):
        pos = np.array(SERVE_DEFAULT_POS, dtype=np.float64)
        pos[1] += self.rng.uniform(*SERVE_POS_Y_RANGE)
        vel = np.array([
            self.rng.uniform(*SERVE_VEL_X),
            self.rng.uniform(*SERVE_VEL_Y),
            self.rng.uniform(*SERVE_VEL_Z),
        ], dtype=np.float64)
        return pos, vel
