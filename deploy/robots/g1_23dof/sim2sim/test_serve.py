import numpy as np
from serve import Serve, SERVE_DEFAULT_POS, SERVE_POS_Y_RANGE, SERVE_VEL_X, SERVE_VEL_Y, SERVE_VEL_Z


def test_sample_serve_within_ranges():
    rng = np.random.default_rng(0)
    s = Serve(rng=rng)
    for _ in range(500):
        pos, vel = s.sample()
        assert abs(pos[0] - SERVE_DEFAULT_POS[0]) < 1e-9
        assert SERVE_DEFAULT_POS[1] + SERVE_POS_Y_RANGE[0] - 1e-9 <= pos[1] <= SERVE_DEFAULT_POS[1] + SERVE_POS_Y_RANGE[1] + 1e-9
        assert abs(pos[2] - SERVE_DEFAULT_POS[2]) < 1e-9
        assert SERVE_VEL_X[0] <= vel[0] <= SERVE_VEL_X[1]
        assert SERVE_VEL_Y[0] <= vel[1] <= SERVE_VEL_Y[1]
        assert SERVE_VEL_Z[0] <= vel[2] <= SERVE_VEL_Z[1]


def test_due_fires_on_interval():
    s = Serve(rng=np.random.default_rng(0), interval_steps=100)
    fired = [t for t in range(1, 351) if s.due(t)]
    assert fired == [100, 200, 300]
