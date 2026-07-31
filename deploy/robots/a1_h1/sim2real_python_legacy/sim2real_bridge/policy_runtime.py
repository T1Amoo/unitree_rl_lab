"""Policy-side runtime shared by ROS sim2real and offline smoke tests.

This module intentionally has no ROS or MuJoCo dependency. The hardware SDK
boundary remains in armcontrol/inference_arm_control_node:

    Float64MultiArray(q_des[7]) -> DAMIAO control_mit(...)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


ACTION_SCALE = 0.25
CLIP_ACTIONS = 10.0
CLIP_OBS = 100.0
SOFT_JOINT_LIMIT_FACTOR = 0.95
FRAME_SIZE = 39
HISTORY = 5
OBS_SIZE = FRAME_SIZE * HISTORY

RIGHT_JOINT_NAMES = [
    "joint1-a1_r",
    "joint2-a1_r",
    "joint3-a1_r",
    "joint4-a1_r",
    "joint5-a1_r",
    "joint6-a1_r",
    "joint7-a1_r",
]
RIGHT_JOINT_ALIASES = [
    ("joint1-a1_r", "joint1-r", "r1"),
    ("joint2-a1_r", "joint2-r", "r2"),
    ("joint3-a1_r", "joint3-r", "r3"),
    ("joint4-a1_r", "joint4-r", "r4"),
    ("joint5-a1_r", "joint5-r", "r5"),
    ("joint6-a1_r", "joint6-r", "r6"),
    ("joint7-a1_r", "joint7-r", "r7"),
]

DEFAULT_RIGHT_Q = np.array([0.569, -0.692, 0.717, 1.13, -1.24, 0.0314, 0.772], dtype=np.float64)
RIGHT_Q_MIN = np.array([-1.05, -3.14, -2.76, -1.92, -2.76, -1.57, -2.76], dtype=np.float64)
RIGHT_Q_MAX = np.array([3.14, 0.262, 2.76, 1.92, 2.76, 1.57, 2.76], dtype=np.float64)

ROBOT_TABLE_POS = np.array([-1.8, 0.76, 0.0282], dtype=np.float32)
HIT_PLANE_X = -1.60
HOME_Y = 0.76
PADDLE_Y_OFFSET = -0.66
HIT_BODY_HEIGHT = 0.028
PRED_SENTINEL = np.array([HIT_PLANE_X, HOME_Y + PADDLE_Y_OFFSET, HIT_BODY_HEIGHT + 0.2], dtype=np.float32)

G = 9.81
Z_BOUNCE = 0.78


def guess_lgy_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "Pingpong_TTRL").exists() and (parent / "unitree_rl_lab").exists():
            return parent
    return Path.cwd()


LGY_ROOT = guess_lgy_root()
DEFAULT_POLICY = LGY_ROOT / "Pingpong_TTRL/logs/a1_tt_real_v7/2026-07-24_14-18-11_resume10000_range10k_hold20k/exported/policy.onnx"


def soft_joint_ranges() -> tuple[np.ndarray, np.ndarray]:
    mid = 0.5 * (RIGHT_Q_MIN + RIGHT_Q_MAX)
    half = 0.5 * (RIGHT_Q_MAX - RIGHT_Q_MIN) * SOFT_JOINT_LIMIT_FACTOR
    return mid - half, mid + half


def ordered_joint_vector(
    names: Iterable[str],
    values: Iterable[float],
    fallback: np.ndarray | None = None,
) -> np.ndarray | None:
    names = list(names)
    values = list(values)
    if not values:
        return None
    if not names and len(values) >= 7:
        return np.asarray(values[:7], dtype=np.float64)
    index = {name: i for i, name in enumerate(names)}
    out = np.asarray(fallback, dtype=np.float64).copy() if fallback is not None else np.zeros(7, dtype=np.float64)
    found = 0
    for j, aliases in enumerate(RIGHT_JOINT_ALIASES):
        for alias in aliases:
            if alias in index and index[alias] < len(values):
                out[j] = float(values[index[alias]])
                found += 1
                break
    if found == 7:
        return out
    if fallback is None and len(values) >= 7:
        return np.asarray(values[:7], dtype=np.float64)
    return out if found > 0 and fallback is not None else None


class OnnxPolicy:
    def __init__(self, policy_path: Path | str):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required for policy inference") from exc
        self.path = Path(policy_path)
        if not self.path.exists():
            raise FileNotFoundError(f"policy not found: {self.path}")
        self.session = ort.InferenceSession(str(self.path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        shape = self.session.get_inputs()[0].shape
        if len(shape) < 2 or (shape[1] not in (None, "None") and int(shape[1]) != OBS_SIZE):
            raise ValueError(f"policy input must be {OBS_SIZE}, got {shape}")

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(1, OBS_SIZE)
        out = self.session.run([self.output_name], {self.input_name: obs})[0]
        return np.asarray(out[0], dtype=np.float32)


class OnnxBallPredictor:
    def __init__(self, predictor_path: Path | str, history_len: int = 5):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required for predictor inference") from exc
        self.path = Path(predictor_path)
        if not self.path.exists():
            raise FileNotFoundError(f"predictor not found: {self.path}")
        self.history_len = int(history_len)
        self.session = ort.InferenceSession(str(self.path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.history: deque[np.ndarray] = deque(maxlen=self.history_len)

    def clear(self) -> None:
        self.history.clear()

    def __call__(self, ball_pos: np.ndarray, valid_ball: bool) -> np.ndarray | None:
        if not valid_ball or not np.isfinite(ball_pos).all():
            self.clear()
            return None
        b = np.asarray(ball_pos, dtype=np.float32).reshape(3)
        self.history.append(b)
        rows = list(self.history)
        if len(rows) < self.history_len:
            rows = [rows[0]] * (self.history_len - len(rows)) + rows
        x = np.concatenate(rows).reshape(1, 3 * self.history_len).astype(np.float32)
        out = self.session.run([self.output_name], {self.input_name: x})[0]
        return np.asarray(out[0], dtype=np.float32)


@dataclass
class BallGateConfig:
    own_x_lo: float = -1.37
    own_x_hi: float = 0.0
    y_abs: float = 1.2
    z_min: float = 0.75
    z_max: float = 2.6
    x_max: float = 1.65
    speed_max: float = 15.0
    vx_away: float = 0.0
    behind_margin: float = 0.05
    bounce_vz_down: float = 0.30
    bounce_vz_up: float = 0.05
    bounce_near_table: float = 0.15
    confirm_frames: int = 1
    coast_frames: int = 1


@dataclass
class BallGateOutput:
    live: bool
    engaged: bool
    reason: str
    first_bounce_x: float
    own_bounces: int
    hit_seen: bool


class BallValidityGate:
    def __init__(self, cfg: BallGateConfig | None = None):
        self.cfg = cfg if cfg is not None else BallGateConfig()
        self.reset()

    def reset(self) -> None:
        self._prev_vz: float | None = None
        self._own_bounces = 0
        self._hit_seen = False
        self._live_run = 0
        self._dead_run = 0
        self._engaged = False
        self.last = BallGateOutput(False, False, "reset", float("nan"), 0, False)

    def register_paddle_hit(self) -> None:
        self._hit_seen = True

    def update(self, ball_pos: np.ndarray, ball_vel: np.ndarray) -> BallGateOutput:
        pos = np.asarray(ball_pos, dtype=np.float64).reshape(3)
        vel = np.asarray(ball_vel, dtype=np.float64).reshape(3)
        self._update_bounces(pos, vel)
        first_bounce_x = self._predicted_first_bounce_x(pos, vel)
        live, reason = self._classify(pos, vel, first_bounce_x)

        if live:
            self._live_run += 1
            self._dead_run = 0
        else:
            self._dead_run += 1
            self._live_run = 0

        if not self._engaged and self._live_run >= max(1, self.cfg.confirm_frames):
            self._engaged = True
        if self._engaged and self._dead_run >= max(1, self.cfg.coast_frames):
            self._engaged = False

        self._prev_vz = float(vel[2]) if np.isfinite(vel[2]) else None
        self.last = BallGateOutput(live, self._engaged, reason, first_bounce_x, self._own_bounces, self._hit_seen)
        return self.last

    def _classify(self, pos: np.ndarray, vel: np.ndarray, first_bounce_x: float) -> tuple[bool, str]:
        cfg = self.cfg
        if not np.isfinite(pos).all() or not np.isfinite(vel).all():
            return False, "nonfinite"
        if self._hit_seen:
            return False, "paddle_hit"
        speed = float(np.linalg.norm(vel))
        if speed > cfg.speed_max:
            return False, "speed"
        if abs(float(pos[1])) > cfg.y_abs or pos[2] > cfg.z_max or pos[0] > cfg.x_max:
            return False, "out_volume"
        if pos[0] < HIT_PLANE_X - cfg.behind_margin:
            return False, "behind_hit_plane"
        if vel[0] > cfg.vx_away:
            return False, "moving_away"
        if pos[2] < cfg.z_min:
            return False, "low_or_dead"
        if self._own_bounces >= 2:
            return False, "double_bounce"
        if self._own_bounces == 0:
            if not np.isfinite(first_bounce_x):
                return False, "no_table_bounce"
            if not (cfg.own_x_lo <= first_bounce_x <= cfg.own_x_hi):
                return False, "first_bounce_out"
        return True, "valid"

    def _update_bounces(self, pos: np.ndarray, vel: np.ndarray) -> None:
        if self._prev_vz is None or not np.isfinite(pos).all() or not np.isfinite(vel).all():
            return
        cfg = self.cfg
        was_descending = self._prev_vz < -cfg.bounce_vz_down
        ascending_now = vel[2] > cfg.bounce_vz_up
        near_table = pos[2] < Z_BOUNCE + cfg.bounce_near_table
        own_half = cfg.own_x_lo <= pos[0] <= cfg.own_x_hi
        if was_descending and ascending_now and near_table and own_half:
            self._own_bounces += 1

    @staticmethod
    def _predicted_first_bounce_x(pos: np.ndarray, vel: np.ndarray) -> float:
        if not np.isfinite(pos).all() or not np.isfinite(vel).all():
            return float("nan")
        a = -0.5 * G
        b = float(vel[2])
        c = float(pos[2] - Z_BOUNCE)
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return float("nan")
        root = float(np.sqrt(disc))
        denom = 2.0 * a
        roots = [(-b - root) / denom, (-b + root) / denom]
        positive = [t for t in roots if t > 1e-4]
        if not positive:
            return float("nan")
        t = max(positive)
        return float(pos[0] + vel[0] * t)


@dataclass
class PolicyStep:
    raw_action: np.ndarray
    q_des: np.ndarray
    dq_des: np.ndarray
    ball_pred: np.ndarray
    obs: np.ndarray


class A1DeployPolicy:
    def __init__(
        self,
        policy_path: Path | str = DEFAULT_POLICY,
        predictor_path: Path | str | None = None,
        use_predictor: bool = True,
    ):
        self.policy = OnnxPolicy(policy_path)
        pp = Path(predictor_path) if predictor_path else Path(policy_path).with_name("predictor.onnx")
        self.predictor = OnnxBallPredictor(pp) if use_predictor and pp.exists() else None
        self.q_min, self.q_max = soft_joint_ranges()
        self.last_action = np.zeros(7, dtype=np.float32)
        self.last_q_des = DEFAULT_RIGHT_Q.copy()
        self.last_ball_pred = PRED_SENTINEL.copy()
        self.history: deque[np.ndarray] = deque(maxlen=HISTORY)
        frame = self.compute_frame(DEFAULT_RIGHT_Q, np.zeros(7), np.zeros(3), np.zeros(3), False)
        for _ in range(HISTORY):
            self.history.append(frame)

    def reset(self, q: np.ndarray | None = None) -> None:
        if self.predictor is not None:
            self.predictor.clear()
        self.last_action.fill(0.0)
        if q is not None:
            self.last_q_des = np.asarray(q, dtype=np.float64).reshape(7).copy()
        self.history.clear()
        frame = self.compute_frame(
            DEFAULT_RIGHT_Q if q is None else np.asarray(q, dtype=np.float64).reshape(7),
            np.zeros(7),
            np.zeros(3),
            np.zeros(3),
            False,
        )
        for _ in range(HISTORY):
            self.history.append(frame)

    def analytic_prediction(self, ball_pos: np.ndarray, ball_vel: np.ndarray, valid_ball: bool) -> np.ndarray:
        if not valid_ball or not np.isfinite(ball_pos).all() or ball_vel[0] >= -0.05:
            return PRED_SENTINEL.copy()
        t = (HIT_PLANE_X - ball_pos[0]) / ball_vel[0]
        if t <= 0.0 or t > 2.0:
            return PRED_SENTINEL.copy()
        pred = ball_pos + ball_vel * t
        pred[2] += -0.5 * G * t * t
        if pred[2] < 0.2 or pred[2] > 2.0:
            return PRED_SENTINEL.copy()
        return pred.astype(np.float32)

    @staticmethod
    def plausible_prediction(pred: np.ndarray) -> bool:
        y_center = HOME_Y + PADDLE_Y_OFFSET
        return bool(
            np.isfinite(pred).all()
            and HIT_PLANE_X - 0.50 < pred[0] < HIT_PLANE_X + 0.30
            and abs(float(pred[1] - y_center)) < 0.45
            and 0.85 < pred[2] < 1.55
        )

    def predict_ball(self, ball_pos: np.ndarray, ball_vel: np.ndarray, valid_ball: bool) -> np.ndarray:
        pred = self.predictor(ball_pos, valid_ball) if self.predictor is not None else None
        if pred is not None and self.plausible_prediction(pred):
            pred = pred.copy()
            pred[0] = HIT_PLANE_X
            return pred.astype(np.float32)
        return self.analytic_prediction(ball_pos, ball_vel, valid_ball)

    def compute_frame(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        ball_pos: np.ndarray,
        ball_vel: np.ndarray,
        valid_ball: bool,
    ) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64).reshape(7)
        dq = np.asarray(dq, dtype=np.float64).reshape(7)
        ball_pos = np.asarray(ball_pos, dtype=np.float32).reshape(3)
        ball_vel = np.asarray(ball_vel, dtype=np.float32).reshape(3)
        joint_pos_rel = q - DEFAULT_RIGHT_Q
        ball_pred = self.predict_ball(ball_pos, ball_vel, valid_ball)
        self.last_ball_pred = ball_pred.copy()
        ball_obs = ball_pos if valid_ball else PRED_SENTINEL.copy()
        perception = np.concatenate([ball_obs, ROBOT_TABLE_POS]).astype(np.float32)
        rel_target_xy = np.array(
            [
                (ball_pred[0] - 0.1) - ROBOT_TABLE_POS[0],
                (ball_pred[1] - PADDLE_Y_OFFSET) - ROBOT_TABLE_POS[1],
            ],
            dtype=np.float32,
        )
        frame = np.concatenate(
            [
                np.zeros(3, dtype=np.float32),
                np.array([0.0, 0.0, -1.0], dtype=np.float32),
                joint_pos_rel.astype(np.float32),
                dq.astype(np.float32),
                self.last_action.astype(np.float32),
                perception,
                ball_pred.astype(np.float32),
                rel_target_xy,
                np.array([0.0], dtype=np.float32),
            ]
        ).astype(np.float32)
        if frame.shape != (FRAME_SIZE,):
            raise RuntimeError(f"bad frame shape {frame.shape}")
        return frame

    def observe(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        ball_pos: np.ndarray,
        ball_vel: np.ndarray,
        valid_ball: bool,
    ) -> np.ndarray:
        frame = self.compute_frame(q, dq, ball_pos, ball_vel, valid_ball)
        self.history.append(frame)
        obs = np.concatenate(list(self.history), dtype=np.float32)
        return np.clip(obs, -CLIP_OBS, CLIP_OBS)

    def action_to_q_des(self, raw_action: np.ndarray) -> np.ndarray:
        raw_action = np.asarray(raw_action, dtype=np.float32).reshape(7)
        self.last_action = raw_action
        clipped = np.clip(raw_action, -CLIP_ACTIONS, CLIP_ACTIONS).astype(np.float64)
        q_des = clipped * ACTION_SCALE + DEFAULT_RIGHT_Q
        self.last_q_des = np.clip(q_des, self.q_min, self.q_max)
        return self.last_q_des.copy()

    def step(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        ball_pos: np.ndarray,
        ball_vel: np.ndarray,
        valid_ball: bool,
        dt: float,
    ) -> PolicyStep:
        obs = self.observe(q, dq, ball_pos, ball_vel, valid_ball)
        raw_action = self.policy(obs)
        prev_q_des = self.last_q_des.copy()
        q_des = self.action_to_q_des(raw_action)
        dq_des = (q_des - prev_q_des) / max(float(dt), 1e-6)
        return PolicyStep(raw_action, q_des, dq_des, self.last_ball_pred.copy(), obs)
