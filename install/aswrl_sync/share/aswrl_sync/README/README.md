# aswrl_sync (ROS2 Humble + Gazebo Classic 11) — ASW-RL time synchronization reproduction scaffold

This project provides:
- Gazebo Classic 11 world with 3 simple robots: dog1, dog2, uav1
- Linux network namespace scripts (netns + bridge + veth) for isolated networking
- `tc netem` controller to emulate Stable / High-Jitter / Multi-peak / Route-switch conditions
- Time sync stack:
  - PTP-like 4-timestamp exchange
  - Kalman filter state: [offset, delay, drift]
  - Window smoothing + 4 strategies: NoComp / Baseline-FW / Baseline-Rule-ASW / ASW-RL (DQN policy)
- Logging to JSONL for analysis + optional rosbag2 capture

> Notes
- Gazebo Classic is deprecated upstream, but this repo targets Classic 11 as requested.
- Run ROS nodes inside network namespaces using `sudo ip netns exec ...`.
- `netem_controller` requires sudo to change qdisc.

## Quick start (minimal)

### 1) Build
```bash
cd ~/aswrl_ws
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```

### 2) Create network namespaces
```bash
ros2 pkg prefix aswrl_sync
# then:
bash $(ros2 pkg prefix aswrl_sync)/share/aswrl_sync/scripts/setup_netns.sh
```

### 3) Start Gazebo (host namespace)
```bash
ros2 launch aswrl_sync sim_multi_robot.launch.py
```

### 4) Start time sync nodes (in namespaces)

Master (dog1):
```bash
sudo ip netns exec ns_dog1 bash -lc '
  source /opt/ros/humble/setup.bash
  source ~/aswrl_ws/install/setup.bash
  export ROS_DOMAIN_ID=42
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  export CYCLONEDDS_URI=file://$(ros2 pkg prefix aswrl_sync)/share/aswrl_sync/config/cyclonedds_dog1.xml
  export ROS_NAMESPACE=/dog1
  ros2 run aswrl_sync sync_master --ros-args -p use_sim_time:=true -p sync_period:=0.1
'
```

Client (dog2):
```bash
sudo ip netns exec ns_dog2 bash -lc '
  source /opt/ros/humble/setup.bash
  source ~/aswrl_ws/install/setup.bash
  export ROS_DOMAIN_ID=42
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  export CYCLONEDDS_URI=file://$(ros2 pkg prefix aswrl_sync)/share/aswrl_sync/config/cyclonedds_dog2.xml
  export ROS_NAMESPACE=/dog2
  ros2 run aswrl_sync sync_client --ros-args -p use_sim_time:=true -p strategy:=baseline_fw -p fixed_window:=16
'
```

Client (uav1):
```bash
sudo ip netns exec ns_uav1 bash -lc '
  source /opt/ros/humble/setup.bash
  source ~/aswrl_ws/install/setup.bash
  export ROS_DOMAIN_ID=42
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  export CYCLONEDDS_URI=file://$(ros2 pkg prefix aswrl_sync)/share/aswrl_sync/config/cyclonedds_uav1.xml
  export ROS_NAMESPACE=/uav1
  ros2 run aswrl_sync sync_client --ros-args -p use_sim_time:=true -p strategy:=nocomp
'
```

### 5) Apply a network scenario (example: jitter on dog2)
```bash
sudo ip netns exec ns_dog2 tc qdisc replace dev veth_ns_dog2_n root netem delay 30ms 20ms distribution normal loss 2%
```

### 6) Logs
By default, nodes write JSONL into `~/.aswrl_logs/<node>/<date>/...jsonl`.

## RL training (conda environment recommended)
See `aswrl_sync/rl/train_dqn.py` and `aswrl_sync/rl/aswrl_env.py`.

Example:
```bash
conda activate aswrl
python -m aswrl_sync.rl.train_dqn --scenario multimodal --steps 300000 --device cuda
python -m aswrl_sync.rl.export_policy --ckpt outputs/dqn_latest.pt --out policy.pt
```

Then run clients with:
```bash
-p strategy:=asw_rl -p policy_path:=/path/to/policy.pt
```
