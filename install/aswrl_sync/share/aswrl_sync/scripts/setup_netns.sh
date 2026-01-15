#!/usr/bin/env bash
set -e

BR=br_aswrl

sudo ip link add ${BR} type bridge 2>/dev/null || true
sudo ip link set ${BR} up
sudo ip link set ${BR} multicast on

create_ns () {
  local NS=$1
  local IPADDR=$2
  local VETH_HOST=veth_${NS}_h
  local VETH_NS=veth_${NS}_n

  sudo ip netns add ${NS} 2>/dev/null || true

  sudo ip link add ${VETH_HOST} type veth peer name ${VETH_NS} 2>/dev/null || true
  sudo ip link set ${VETH_NS} netns ${NS}

  sudo ip link set ${VETH_HOST} master ${BR}
  sudo ip link set ${VETH_HOST} up

  sudo ip -n ${NS} link set lo up
  sudo ip -n ${NS} link set ${VETH_NS} up
  sudo ip -n ${NS} addr add ${IPADDR}/24 dev ${VETH_NS} 2>/dev/null || true
  sudo ip -n ${NS} link set ${VETH_NS} multicast on

  # Helpful debug
  echo "[NS ${NS}] iface=${VETH_NS}, ip=${IPADDR}"
}

create_ns ns_dog1 10.10.0.11
create_ns ns_dog2 10.10.0.12
create_ns ns_uav1 10.10.0.13

echo "[OK] namespaces + bridge ready"
