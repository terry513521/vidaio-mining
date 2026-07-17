#!/usr/bin/env bash
echo "devices:"; ls -la /dev/nvidia* 2>&1
echo "nvidia-smi:"; nvidia-smi 2>&1 | head -15
echo "docker gpus:"; docker info 2>/dev/null | egrep -i 'Runtimes|nvidia' || true
