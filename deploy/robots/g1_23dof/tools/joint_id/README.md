# G1 23-DoF Joint ID

Single-joint chirp collection for the five right-arm joints of the Unitree G1
23-DoF table-tennis setup. The script does not run a policy or FSM. It publishes
full-body `rt/lowcmd` only when `--enable-control` is passed:

- legs, waist, and left arm hold the table-tennis default pose
- one right-arm joint runs a linear chirp around that default pose
- feedback comes from `rt/lowstate`
- output is a CSV plus a manifest JSON

Output layout:

```text
系统辨识/unitree/YYYYMMDD/jointN/
  real/      # real G1 test CSV + manifest
  sim/       # replay/simulation CSV
  plots/     # comparison figures
  merged/    # aligned real/sim tables
  summary/   # fitted metrics or notes
```

Right-arm joint order:

| `--joint` | policy index | SDK motor id | joint name |
|---:|---:|---:|---|
| 1 | 18 | 22 | `right_shoulder_pitch_joint` |
| 2 | 19 | 23 | `right_shoulder_roll_joint` |
| 3 | 20 | 24 | `right_shoulder_yaw_joint` |
| 4 | 21 | 25 | `right_elbow_joint` |
| 5 | 22 | 26 | `right_wrist_roll_joint` |

Dry-run:

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy
conda run --no-capture-output -n g1tt_sim2sim python \
  unitree_rl_lab/deploy/robots/g1_23dof/tools/joint_id/record_g1_chirp_sdk2.py \
  --joint 1 \
  --duration-s 1 \
  --move-to-default-s 0.1 \
  --warmup-s 0.1 \
  --post-hold-s 0.1 \
  --rate-hz 100 \
  --output /tmp/g1_joint_id_dryrun.csv
```

Real robot, one joint:

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy
ip link show dev enx6c1ff76cb7d7
ip route get 192.168.123.161

conda run --no-capture-output -n g1tt_sim2sim python \
  unitree_rl_lab/deploy/robots/g1_23dof/tools/joint_id/record_g1_chirp_sdk2.py \
  --enable-control \
  --network-interface enx6c1ff76cb7d7 \
  --joint 1 \
  --amplitude 0.08 \
  --chirp-start-hz 0.1 \
  --chirp-end-hz 2.0 \
  --duration-s 30
```

By default this writes to
`系统辨识/unitree/YYYYMMDD/joint1/real/j1_chirp_<run_id>.csv` and creates sibling
`sim/`, `plots/`, `merged/`, and `summary/` directories. Use `--date-tag
YYYYMMDD` to force a specific date folder.

Run only one controlling process at a time. Stop `g1_ctrl` or other low-level
publishers before enabling this script.
