#!/usr/bin/env bash
set -euo pipefail

# The graph node disables AEC/AGC after opening the ZED-X.  Without an
# explicit follow-up write, the sensor keeps whichever exposure/gain happened
# to be present at startup.  Apply and verify the versioned moving-ball
# profile on both members of the stereo pair.

exposure_us="${1:-${A1_CAMERA_EXPOSURE_US:-6000}}"
gain_raw="${2:-${A1_CAMERA_GAIN_RAW:-1601}}"
post_open_delay_s="${3:-${A1_CAMERA_POST_OPEN_DELAY_S:-7}}"
stable_checks="${A1_CAMERA_STABLE_CHECKS:-3}"
timeout_s="${A1_CAMERA_CONFIG_TIMEOUT_S:-20}"

is_uint() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

if ! is_uint "$exposure_us" || (( exposure_us < 28 || exposure_us > 16600 )); then
    echo "CAMERA_SETTINGS_FAIL invalid exposure_us=$exposure_us allowed=[28,16600]" >&2
    exit 2
fi
if ! is_uint "$gain_raw" || (( gain_raw < 100 || gain_raw > 1601 )); then
    echo "CAMERA_SETTINGS_FAIL invalid gain_raw=$gain_raw allowed=[100,1601]" >&2
    exit 2
fi
if ! is_uint "$stable_checks" || (( stable_checks < 1 || stable_checks > 10 )); then
    echo "CAMERA_SETTINGS_FAIL invalid stable_checks=$stable_checks allowed=[1,10]" >&2
    exit 2
fi
if ! is_uint "$timeout_s" || (( timeout_s < 1 || timeout_s > 120 )); then
    echo "CAMERA_SETTINGS_FAIL invalid timeout_s=$timeout_s allowed=[1,120]" >&2
    exit 2
fi
if ! is_uint "$post_open_delay_s" || (( post_open_delay_s > 30 )); then
    echo "CAMERA_SETTINGS_FAIL invalid post_open_delay_s=$post_open_delay_s allowed=[0,30]" >&2
    exit 2
fi

# ros2 run first starts a Python wrapper; the actual graph process appears a
# few seconds later and zed_.open() then rewrites sensor controls.  Wait for
# that child and apply the profile after the open path has settled.  Live
# exposure sweeps pass a third argument of 0 because the node is already open.
if (( post_open_delay_s > 0 )); then
    while ! pgrep -f '/pingpong_detect_graph_node --ros-args' >/dev/null; do
        if (( SECONDS > timeout_s )); then
            echo "CAMERA_SETTINGS_FAIL graph node did not start within ${timeout_s}s" >&2
            exit 4
        fi
        sleep 0.2
    done
    sleep "$post_open_delay_s"
fi

devices=(/dev/video0 /dev/video1)
deadline=$((SECONDS + timeout_s))
consecutive=0

while (( SECONDS <= deadline )); do
    ready=1
    for device in "${devices[@]}"; do
        if [[ ! -e "$device" ]]; then
            ready=0
            break
        fi
        card="$(v4l2-ctl -d "$device" --info 2>/dev/null | sed -n 's/^[[:space:]]*Card type[[:space:]]*:[[:space:]]*//p' | head -n 1)"
        if [[ "$card" != *zedx* ]]; then
            echo "CAMERA_SETTINGS_FAIL unexpected device=$device card=$card" >&2
            exit 3
        fi
        if ! v4l2-ctl -d "$device" \
            --set-ctrl="exposure=${exposure_us},gain=${gain_raw}" >/dev/null 2>&1; then
            ready=0
            break
        fi
        values="$(v4l2-ctl -d "$device" --get-ctrl=exposure,gain 2>/dev/null || true)"
        actual_exposure="$(sed -n 's/^exposure:[[:space:]]*//p' <<<"$values")"
        actual_gain="$(sed -n 's/^gain:[[:space:]]*//p' <<<"$values")"
        if [[ "$actual_exposure" != "$exposure_us" || "$actual_gain" != "$gain_raw" ]]; then
            ready=0
            break
        fi
    done

    if (( ready )); then
        consecutive=$((consecutive + 1))
        if (( consecutive >= stable_checks )); then
            echo "CAMERA_SETTINGS_OK exposure_us=$exposure_us gain_raw=$gain_raw devices=${devices[*]} stable_checks=$consecutive"
            exit 0
        fi
    else
        consecutive=0
    fi
    sleep 0.2
done

echo "CAMERA_SETTINGS_FAIL timeout=${timeout_s}s exposure_us=$exposure_us gain_raw=$gain_raw" >&2
for device in "${devices[@]}"; do
    v4l2-ctl -d "$device" --get-ctrl=exposure,gain 2>/dev/null || true
done
exit 4
