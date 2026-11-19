#!/usr/bin/env bash
# Prepare the robotiq-ros submodule for this workspace.
#
# Only robotiq_description (2F-85 xacro + meshes) is used. The other packages in the
# repo are for real hardware and must not be built here:
#   robotiq_driver / robotiq_controllers / robotiq_grippers_sdk_vendor
#       -> the SDK vendor git-clones Robotiq's SDK from GitHub during the build
#   robotiq_tsf / robotiq_tactile_sdk_vendor
#       -> same, and needs libserialport-dev
#   robotiq_msgs / robotiq_test_support / robotiq_hardware_tests -> not needed
# An empty COLCON_IGNORE file makes colcon skip a package. The markers live inside the
# submodule (untracked there; .gitmodules sets ignore = untracked), so run this script
# again after a fresh clone.
#
# Usage:  tools/setup_robotiq.sh        (from anywhere in the repo)
set -euo pipefail
ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
SUB="$ROOT/ros2_ws/src/robotiq-ros"

git -C "$ROOT" submodule update --init ros2_ws/src/robotiq-ros

for pkg in grippers/robotiq_driver grippers/robotiq_controllers grippers/robotiq_grippers_sdk_vendor \
           grippers/robotiq_hardware_tests grippers/robotiq_test_support grippers/robotiq_msgs \
           robotiq_tsf robotiq_tactile_sdk_vendor; do
  touch "$SUB/$pkg/COLCON_IGNORE"
  echo "ignored: $pkg"
done
echo "robotiq-ros ready: colcon will build only robotiq_description from it."
