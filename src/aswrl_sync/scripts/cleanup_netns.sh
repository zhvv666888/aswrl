#!/usr/bin/env bash
set -e

for NS in ns_dog1 ns_dog2 ns_uav1; do
  sudo ip netns del ${NS} 2>/dev/null || true
done

sudo ip link del br_aswrl 2>/dev/null || true
echo "[OK] cleaned"
