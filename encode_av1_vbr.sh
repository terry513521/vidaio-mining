#!/usr/bin/env bash
# Encode one video to AV1 VBR with aomenc.
#
# Usage:
#   ./encode_av1_vbr.sh -i INPUT.mp4 -b BITRATE_KBPS -o OUTPUT.mp4
#   ./encode_av1_vbr.sh -i INPUT.mp4 -r 0.04 -o OUTPUT.mp4
#
# Examples:
#   ./encode_av1_vbr.sh -i /root/workspace/video/1.mp4 -b 1440 -o /tmp/out.mp4
#   ./encode_av1_vbr.sh -i /root/workspace/video/1.mp4 -r 0.04 -o /tmp/out.mp4 --vmaf-neg

set -euo pipefail

AOMENC="${AOMENC:-/root/workspace/aom/build/aomenc}"
VMAF_MODEL_NEG="${VMAF_MODEL_NEG:-/usr/local/share/model/vmaf_v0.6.1neg.json}"

INPUT=""
OUTPUT=""
BITRATE_KBPS=""
TARGET_RATE=""
CPU_USED=6
THREADS=16
PASSES=1
TUNE_VMAF_NEG=0
USAGE_PROFILE="good"   # good: cpu-used 0..9 | rt: cpu-used 0..12

usage() {
  cat <<'EOF'
Usage: encode_av1_vbr.sh -i INPUT -o OUTPUT (-b BITRATE_KBPS | -r COMPRESSION_RATE) [options]

Required:
  -i, --input PATH          Source video (mp4, etc.)
  -o, --output PATH         Output AV1 mp4
  -b, --bitrate KBPS        Target bitrate in kbps (e.g. 1440)
  -r, --rate RATE           Target compression rate (0,1), e.g. 0.04
                            (used only if -b is not set)

Options:
  --cpu-used N              Speed/quality. Default: 6
                            --good (default): 0..9 (0=best, 9=fastest)
                            --rt: 0..12
  --good                    Good-quality profile (default; cpu-used max 9)
  --rt                      Realtime profile (allows cpu-used up to 12)
  --threads N               Encoder threads. Default: 16
  --passes N                1 or 2. Default: 1
  --vmaf-neg                Enable --tune=vmaf_neg
  --aomenc PATH             Path to aomenc binary
  -h, --help                Show this help

Env:
  AOMENC                    Override aomenc path
  VMAF_MODEL_NEG            VMAF NEG model path (with --vmaf-neg)
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--input) INPUT="${2:-}"; shift 2 ;;
    -o|--output) OUTPUT="${2:-}"; shift 2 ;;
    -b|--bitrate) BITRATE_KBPS="${2:-}"; shift 2 ;;
    -r|--rate) TARGET_RATE="${2:-}"; shift 2 ;;
    --cpu-used) CPU_USED="${2:-}"; shift 2 ;;
    --good) USAGE_PROFILE="good"; shift ;;
    --rt) USAGE_PROFILE="rt"; shift ;;
    --threads) THREADS="${2:-}"; shift 2 ;;
    --passes) PASSES="${2:-}"; shift 2 ;;
    --vmaf-neg) TUNE_VMAF_NEG=1; shift ;;
    --aomenc) AOMENC="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1 (try --help)" ;;
  esac
done

[[ -n "$INPUT" ]] || die "missing -i/--input"
[[ -n "$OUTPUT" ]] || die "missing -o/--output"
[[ -f "$INPUT" ]] || die "input not found: $INPUT"
[[ -x "$AOMENC" || -f "$AOMENC" ]] || die "aomenc not found: $AOMENC"
command -v ffmpeg >/dev/null || die "ffmpeg not found"
command -v ffprobe >/dev/null || die "ffprobe not found"

[[ "$CPU_USED" =~ ^[0-9]+$ ]] || die "cpu-used must be an integer, got: $CPU_USED"
if [[ "$USAGE_PROFILE" == "rt" ]]; then
  CPU_MAX=12
  USAGE_FLAG="--rt"
else
  CPU_MAX=9
  USAGE_FLAG="--good"
fi
if (( CPU_USED > CPU_MAX )); then
  echo "WARN: --cpu-used=$CPU_USED is out of range for $USAGE_FLAG (max $CPU_MAX); clamping to $CPU_MAX" >&2
  CPU_USED=$CPU_MAX
fi
if (( CPU_USED < 0 )); then
  die "cpu-used must be >= 0"
fi

if [[ -z "$BITRATE_KBPS" ]]; then
  [[ -n "$TARGET_RATE" ]] || die "need -b/--bitrate or -r/--rate"
  BYTES=$(stat -c%s "$INPUT")
  DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$INPUT")
  BITRATE_KBPS=$(python3 -c "
rate=float('${TARGET_RATE}')
if not (0.0 < rate < 1.0):
    raise SystemExit('rate must be in (0,1)')
bytes=float('${BYTES}')
dur=float('${DUR}')
print(max(1, int((bytes * rate * 8.0 / 1000.0) / dur)))
")
fi

[[ "$BITRATE_KBPS" =~ ^[0-9]+$ ]] || die "bitrate must be an integer kbps, got: $BITRATE_KBPS"

OUT_DIR=$(dirname "$OUTPUT")
mkdir -p "$OUT_DIR"
IVF=$(mktemp /tmp/aomenc_XXXXXX.ivf)
trap 'rm -f "$IVF"' EXIT

AOM_ARGS=(
  "$USAGE_FLAG"
  --end-usage=vbr
  --target-bitrate="$BITRATE_KBPS"
  --cpu-used="$CPU_USED"
  --threads="$THREADS"
  --passes="$PASSES"
  -o "$IVF"
  -
)

if [[ "$TUNE_VMAF_NEG" -eq 1 ]]; then
  [[ -f "$VMAF_MODEL_NEG" ]] || die "VMAF model not found: $VMAF_MODEL_NEG"
  AOM_ARGS=(
    "$USAGE_FLAG"
    --end-usage=vbr
    --target-bitrate="$BITRATE_KBPS"
    --tune=vmaf_neg
    --vmaf-model-path="$VMAF_MODEL_NEG"
    --cpu-used="$CPU_USED"
    --threads="$THREADS"
    --passes="$PASSES"
    -o "$IVF"
    -
  )
fi

echo "input : $INPUT"
echo "output: $OUTPUT"
echo "bitrate: ${BITRATE_KBPS} kbps"
[[ -n "$TARGET_RATE" ]] && echo "rate  : $TARGET_RATE"
echo "aomenc: $AOMENC ($USAGE_FLAG cpu-used=$CPU_USED threads=$THREADS passes=$PASSES vmaf_neg=$TUNE_VMAF_NEG)"
echo

ffmpeg -y -hide_banner -loglevel error -i "$INPUT" -an -f yuv4mpegpipe - | \
  "$AOMENC" "${AOM_ARGS[@]}"

ffmpeg -y -hide_banner -loglevel error -i "$IVF" -c copy "$OUTPUT"

echo
echo "Wrote: $OUTPUT"
ls -lh "$OUTPUT"
