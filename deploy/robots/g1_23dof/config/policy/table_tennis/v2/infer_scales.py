#!/usr/bin/env python3
"""
Infer per-term obs scales from recorded ref.npz.

Usage:
    python infer_scales.py <path/to/ref.npz>

For each obs term, computes:
    scale = actor_obs_slice / raw_value   (element-wise, per step)
and takes the median over steps. Reports mean+std across the term's elements.

The actor_obs is frame-major: [frame0(87), frame1(87), ..., frame4(87)]
NEWEST frame = actor_obs[i, 348:435] (offset = 4*87 = 348)
OLDEST frame = actor_obs[i, 0:87]
"""

import sys
import numpy as np
import ast

NEWEST_OFFSET = 4 * 87  # = 348


def infer_term_scale(actor_obs, raw, lo, hi, term_name, frame_offset=NEWEST_OFFSET):
    """
    Compute elementwise scale for a term spanning [lo:hi] in single frame.
    Returns (scale_vec, uniform, frame_used, notes).
    """
    obs_slice = actor_obs[:, frame_offset + lo: frame_offset + hi]  # (N, dim)
    raw_arr = raw  # (N, dim)
    assert obs_slice.shape == raw_arr.shape, (
        f"{term_name}: shape mismatch obs {obs_slice.shape} vs raw {raw_arr.shape}"
    )

    N, D = obs_slice.shape
    scale_per_step = np.full((N, D), np.nan)
    for d in range(D):
        r = raw_arr[:, d]
        o = obs_slice[:, d]
        valid = np.abs(r) > 1e-6
        if valid.sum() < 5:
            # Nearly-zero element everywhere — convention: 1.0
            scale_per_step[:, d] = 1.0
        else:
            scale_per_step[valid, d] = o[valid] / r[valid]
            # For near-zero raw, fill with median of valid steps
            med = np.nanmedian(scale_per_step[valid, d])
            scale_per_step[~valid, d] = med

    # Median per element across steps
    scale_vec = np.nanmedian(scale_per_step, axis=0)  # (D,)

    # Check std across steps for each element (robustness)
    scale_std = np.nanstd(scale_per_step, axis=0)

    # Check uniformity across elements
    elem_mean = scale_vec.mean()
    elem_std = scale_vec.std()
    uniform = bool(elem_std < 1e-4 * abs(elem_mean) + 1e-6)

    notes = []
    # Check for near-zero raw elements
    all_zero_mask = (np.abs(raw_arr).max(axis=0) < 1e-6)
    if all_zero_mask.any():
        notes.append(f"elements {np.where(all_zero_mask)[0].tolist()} all-zero raw -> scale set 1.0")

    # Report per-step instability
    unstable = np.where(scale_std > 0.05 * (np.abs(scale_vec) + 1e-8))[0]
    if len(unstable) > 0:
        notes.append(f"unstable elements (std>5% of scale): {unstable.tolist()}")

    return scale_vec, uniform, "newest", notes


def try_both_frames(actor_obs, raw, lo, hi, term_name):
    """Try newest frame first; fall back to oldest if scales look unstable."""
    sv_new, uni_new, _, notes_new = infer_term_scale(actor_obs, raw, lo, hi, term_name, NEWEST_OFFSET)
    # Check quality: stable if std is small relative to scale magnitude
    N, D = raw.shape
    obs_new = actor_obs[:, NEWEST_OFFSET + lo: NEWEST_OFFSET + hi]
    obs_old = actor_obs[:, lo:hi]

    # Compute ratio std for newest
    valid_new = np.abs(raw) > 1e-6
    ratio_new = np.where(valid_new, obs_new / np.where(valid_new, raw, 1.0), np.nan)
    std_new = np.nanstd(ratio_new)

    # Compute ratio std for oldest
    ratio_old = np.where(valid_new, obs_old / np.where(valid_new, raw, 1.0), np.nan)
    std_old = np.nanstd(ratio_old)

    if std_old < std_new * 0.5:
        # Oldest frame aligns better
        sv_old, uni_old, _, notes_old = infer_term_scale(actor_obs, raw, lo, hi, term_name, 0)
        return sv_old, uni_old, "oldest", notes_old + [f"[FRAME MISMATCH: oldest better, std_new={std_new:.4f} std_old={std_old:.4f}]"]
    else:
        return sv_new, uni_new, "newest", notes_new


def main():
    if len(sys.argv) < 2:
        print("Usage: infer_scales.py <ref.npz>")
        sys.exit(1)

    npz_path = sys.argv[1]
    d = np.load(npz_path, allow_pickle=True)

    actor_obs = d['actor_obs']  # (N, 435)
    N = actor_obs.shape[0]
    print(f"Loaded {npz_path}: N={N} steps, actor_obs shape={actor_obs.shape}")
    print()

    # Parse meta
    meta = ast.literal_eval(str(d['meta'][0]))
    print(f"Meta: {meta}")
    print()

    # ---- Term definitions ----
    # (name, lo, hi, raw_key or None if derived)
    terms = [
        ("base_ang_vel",       0,   3,  "ang_vel"),
        ("projected_gravity",  3,   6,  "proj_grav"),
        ("joint_pos_rel",      6,  29,  "joint_pos_rel"),
        ("joint_vel_rel",     29,  52,  "joint_vel"),
        ("last_action",       52,  75,  "last_action"),
        ("tt_ball_perception",75,  81,  "delayed_perception"),
        ("tt_ball_prediction",81,  84,  "ball_pred"),
        ("tt_rel_target_xy",  84,  86,  None),   # derived
        ("tt_heading",        86,  87,  "heading"),
    ]

    results = {}

    for term_name, lo, hi, raw_key in terms:
        dim = hi - lo
        print(f"=== {term_name} (slice {lo}:{hi}, dim={dim}) ===")

        if raw_key is not None:
            raw = d[raw_key]  # (N, dim)
        else:
            # tt_rel_target_xy: [ball_pred_x - 0.1 - robot_pos_x, ball_pred_y + 0.6 - robot_pos_y]
            # raw is in world/env units; infer scale
            ball_pred = d['ball_pred']   # (N, 3)
            robot_pos = d['robot_pos']   # (N, 3)
            raw = np.stack([
                ball_pred[:, 0] - 0.1 - robot_pos[:, 0],
                ball_pred[:, 1] + 0.6 - robot_pos[:, 1],
            ], axis=1)  # (N, 2)
            raw_key = "(derived: ball_pred_xy - offset - robot_pos_xy)"

        raw = raw.reshape(N, -1)
        assert raw.shape[1] == dim, f"{term_name}: raw dim {raw.shape[1]} != expected {dim}"

        sv, uniform, frame_used, notes = try_both_frames(actor_obs, raw, lo, hi, term_name)

        print(f"  Frame used: {frame_used}")
        print(f"  Scale vector: {np.round(sv, 6).tolist()}")
        print(f"  Uniform: {uniform}  |  mean={sv.mean():.6f}  std={sv.std():.6f}")
        if notes:
            for n in notes:
                print(f"  NOTE: {n}")
        print()

        results[term_name] = {
            "scale": sv,
            "uniform": uniform,
            "frame": frame_used,
            "lo": lo, "hi": hi,
        }

    # ---- Summary ----
    print("=" * 60)
    print("SUMMARY (for deploy.yaml)")
    print("=" * 60)
    for term_name, info in results.items():
        sv = info["scale"]
        if info["uniform"]:
            s = round(float(sv[0]), 6)
            print(f"  {term_name}: {s}  (uniform, frame={info['frame']})")
        else:
            s_list = [round(float(x), 6) for x in sv]
            print(f"  {term_name}: {s_list}  (NON-UNIFORM, frame={info['frame']})")

    return results


if __name__ == "__main__":
    main()
