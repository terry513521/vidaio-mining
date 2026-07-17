#!/usr/bin/env bash
# Repair NVIDIA device nodes so nvidia-smi / NVENC / Docker --gpus work.
# Run on the host (not inside a restricted sandbox):
#   sudo bash fix_gpu.sh

set -euo pipefail

echo "== before =="
ls -la /dev/nvidia* 2>&1 || true
nvidia-smi 2>&1 | head -5 || true

echo "== driver/library check =="
if nvidia-smi 2>&1 | grep -q "Driver/library version mismatch"; then
  echo "Detected driver/library version mismatch."
  echo "Loading DKMS module for current kernel (do NOT rmmod nvidia on a mismatch)."
  if command -v dkms >/dev/null 2>&1; then
    dkms install nvidia/580.159.03 -k "$(uname -r)" 2>/dev/null || true
  fi
fi

echo "== ensure modules =="
modprobe nvidia || true
modprobe nvidia_uvm || true
modprobe nvidia_modeset || true
modprobe nvidia_drm || true

if command -v nvidia-modprobe >/dev/null 2>&1; then
  echo "== nvidia-modprobe =="
  nvidia-modprobe -u -c 0 || true
  nvidia-modprobe -c 0 || true
else
  echo "== nvidia-modprobe missing; installing =="
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nvidia-modprobe
  nvidia-modprobe -u -c 0 || true
  nvidia-modprobe -c 0 || true
fi

echo "== create device nodes if still missing =="
# majors from /proc/devices on this host (nvidia 195, nvidia-uvm 236)
if [ ! -e /dev/nvidiactl ]; then
  mknod -m 666 /dev/nvidiactl c 195 255
fi
if [ ! -e /dev/nvidia0 ]; then
  mknod -m 666 /dev/nvidia0 c 195 0
fi
if [ ! -e /dev/nvidia-modeset ]; then
  mknod -m 666 /dev/nvidia-modeset c 195 254
fi
if [ ! -e /dev/nvidia-uvm ]; then
  mknod -m 666 /dev/nvidia-uvm c 236 0
fi
if [ ! -e /dev/nvidia-uvm-tools ]; then
  mknod -m 666 /dev/nvidia-uvm-tools c 236 1
fi

echo "== restart persistenced =="
systemctl restart nvidia-persistenced 2>/dev/null || true

echo "== after =="
ls -la /dev/nvidia*
nvidia-smi

echo "== docker GPU smoke test =="
docker run --rm --gpus all --entrypoint nvidia-smi nvidia/cuda:12.4.0-base-ubuntu22.04 2>&1 | head -30 \
  || docker run --rm --gpus all vmaf_ffmpeg -hide_banner -f lavfi -i color=c=black:s=64x64:d=0.1 -f null - 2>&1 | tail -20

echo "OK: GPU userspace looks usable."
