"""Make a logging-only copy of the production follower in /tmp and build it."""
from __future__ import annotations
import difflib,hashlib,json,shutil,subprocess
from pathlib import Path
from eat_nav_clock_diagnostic_stage0 import OUT,FOLLOWER,atomic_json,sha

COPY=OUT/'diagnostic-follower-source'
BUILD=OUT/'diagnostic-follower-build'
CPP_REL=Path('src/simple_trajectory_follower.cpp')
HPP_REL=Path('include/simple_trajectory_follower/simple_trajectory_follower.hpp')

def insert_once(s,old,new):
    if s.count(old)!=1:raise ValueError(f'expected one insertion anchor, got {s.count(old)}: {old[:80]}')
    return s.replace(old,new,1)

def diagnostic_sources():
    original_cpp=(FOLLOWER/CPP_REL).read_text()
    original_hpp=(FOLLOWER/HPP_REL).read_text()
    cpp=original_cpp
    cpp=insert_once(cpp,'#include <algorithm>\n','#include <algorithm>\n#include <chrono>\n#include <unistd.h>\n')
    cpp=insert_once(cpp,'void SimpleTrajectoryFollower::pathCallback(const nav_msgs::msg::Path::SharedPtr path)\n{',
'''void SimpleTrajectoryFollower::pathCallback(const nav_msgs::msg::Path::SharedPtr path)
{
  if (std::string(this->get_name()) == "eat_nav_follower_0") {
    const auto wall_ns = std::chrono::system_clock::now().time_since_epoch();
    const auto steady_ns = std::chrono::steady_clock::now().time_since_epoch();
    RCLCPP_INFO(this->get_logger(),
      "CLOCK_DIAG path seq=%lu pid=%d wall_ns=%ld steady_ns=%ld stamp_sec=%d stamp_nsec=%u poses=%zu recovery=%d",
      ++diagnostic_path_seq_, getpid(),
      std::chrono::duration_cast<std::chrono::nanoseconds>(wall_ns).count(),
      std::chrono::duration_cast<std::chrono::nanoseconds>(steady_ns).count(),
      path->header.stamp.sec, path->header.stamp.nanosec, path->poses.size(),
      linear_stuck_recovery_active_ ? 1 : 0);
  }''')
    cpp=insert_once(cpp,'void SimpleTrajectoryFollower::controlLoop()\n{',
'''void SimpleTrajectoryFollower::controlLoop()
{
  const auto diagnostic_seq = ++diagnostic_callback_seq_;''')
    cpp=insert_once(cpp,'    current_pose_.orientation = transform.transform.rotation;\n    has_robot_pose_ = true;',
'''    current_pose_.orientation = transform.transform.rotation;
    if (std::string(this->get_name()) == "eat_nav_follower_0") {
      const auto steady_ns = std::chrono::steady_clock::now().time_since_epoch();
      RCLCPP_INFO(this->get_logger(),
        "CLOCK_DIAG tf seq=%lu steady_ns=%ld stamp_sec=%d stamp_nsec=%u x=%.9f y=%.9f qz=%.9f qw=%.9f",
        diagnostic_callback_seq_, std::chrono::duration_cast<std::chrono::nanoseconds>(steady_ns).count(),
        transform.header.stamp.sec, transform.header.stamp.nanosec,
        current_pose_.position.x, current_pose_.position.y,
        current_pose_.orientation.z, current_pose_.orientation.w);
    }
    has_robot_pose_ = true;''')
    cpp=insert_once(cpp,'  if (!linear_stuck_recovery_active_ && shouldTriggerProgressStallRecovery(now)) {',
'''  if (std::string(this->get_name()) == "eat_nav_follower_0") {
    const auto wall_ns = std::chrono::system_clock::now().time_since_epoch();
    const auto steady_ns = std::chrono::steady_clock::now().time_since_epoch();
    const double window_elapsed = motion_history_.empty() ? 0.0 :
      (motion_history_.back().stamp - motion_history_.front().stamp).seconds();
    const double ratio = summed_commanded_linear_distance_ > 1.0e-6 ?
      summed_actual_linear_distance_ / summed_commanded_linear_distance_ : 0.0;
    RCLCPP_INFO(this->get_logger(),
      "CLOCK_DIAG pre seq=%lu pid=%d wall_ns=%ld steady_ns=%ld node_ns=%ld clock_type=%d count=%zu window_elapsed=%.9f ready=%d cmd_dist=%.9f actual_dist=%.9f ratio=%.9f threshold=%.9f motion_ref=%d last_sample_ns=%ld recovery=%d progress_elapsed=%.9f x=%.9f y=%.9f",
      diagnostic_seq, getpid(),
      std::chrono::duration_cast<std::chrono::nanoseconds>(wall_ns).count(),
      std::chrono::duration_cast<std::chrono::nanoseconds>(steady_ns).count(),
      now.nanoseconds(), static_cast<int>(now.get_clock_type()), motion_history_.size(),
      window_elapsed, isMotionWindowReady() ? 1 : 0,
      summed_commanded_linear_distance_, summed_actual_linear_distance_, ratio,
      linear_stuck_ratio_threshold_, has_motion_reference_ ? 1 : 0,
      has_motion_reference_ ? last_motion_sample_time_.nanoseconds() : 0,
      linear_stuck_recovery_active_ ? 1 : 0,
      has_progress_reference_ ? (now - last_progress_time_).seconds() : 0.0,
      current_pose_.position.x, current_pose_.position.y);
  }
  if (!linear_stuck_recovery_active_ && shouldTriggerProgressStallRecovery(now)) {''')
    cpp=insert_once(cpp,'  updateMotionTracking(now, robot_yaw, cmd.linear.x, cmd.angular.z);\n\n  if (linear_stuck_recovery_active_) {',
'''  updateMotionTracking(now, robot_yaw, cmd.linear.x, cmd.angular.z);
  if (std::string(this->get_name()) == "eat_nav_follower_0") {
    const auto steady_ns = std::chrono::steady_clock::now().time_since_epoch();
    const double window_elapsed = motion_history_.empty() ? 0.0 :
      (motion_history_.back().stamp - motion_history_.front().stamp).seconds();
    RCLCPP_INFO(this->get_logger(),
      "CLOCK_DIAG post seq=%lu steady_ns=%ld node_ns=%ld count=%zu window_elapsed=%.9f cmd_dist=%.9f actual_dist=%.9f recovery=%d vx=%.9f wz=%.9f",
      diagnostic_seq, std::chrono::duration_cast<std::chrono::nanoseconds>(steady_ns).count(),
      now.nanoseconds(), motion_history_.size(), window_elapsed,
      summed_commanded_linear_distance_, summed_actual_linear_distance_,
      linear_stuck_recovery_active_ ? 1 : 0, cmd.linear.x, cmd.angular.z);
  }

  if (linear_stuck_recovery_active_) {''')
    hpp=insert_once(original_hpp,'  FollowerStatus status_{FollowerStatus::Idle};',
'''  // Diagnostic-only counters; never read by a control decision.
  uint64_t diagnostic_callback_seq_{0};
  uint64_t diagnostic_path_seq_{0};
  FollowerStatus status_{FollowerStatus::Idle};''')
    return original_cpp,cpp,original_hpp,hpp

def main():
    audit=json.loads((OUT/'clock-source-audit.json').read_text())
    assert sha(FOLLOWER/CPP_REL)==audit['source_sha256']['follower_source']['sha256']
    assert sha(FOLLOWER/HPP_REL)==audit['source_sha256']['follower_header']['sha256']
    original_cpp,cpp,original_hpp,hpp=diagnostic_sources()
    if COPY.exists():
        assert (COPY/CPP_REL).read_text()==cpp and (COPY/HPP_REL).read_text()==hpp
    else:
        shutil.copytree(FOLLOWER,COPY)
        (COPY/CPP_REL).write_text(cpp)
        (COPY/HPP_REL).write_text(hpp)
    diff=''.join(difflib.unified_diff(original_cpp.splitlines(True),cpp.splitlines(True),fromfile='production/simple_trajectory_follower.cpp',tofile='diagnostic/simple_trajectory_follower.cpp'))
    diff+=''.join(difflib.unified_diff(original_hpp.splitlines(True),hpp.splitlines(True),fromfile='production/simple_trajectory_follower.hpp',tofile='diagnostic/simple_trajectory_follower.hpp'))
    patch=OUT/'diagnostic-follower.patch'
    if patch.exists():assert patch.read_text()==diff
    else:patch.write_text(diff)
    atomic_json(OUT/'diagnostic-instrumentation-diff.json',{
        'production_cpp_sha256':sha(FOLLOWER/CPP_REL),'diagnostic_cpp_sha256':sha(COPY/CPP_REL),
        'production_header_sha256':sha(FOLLOWER/HPP_REL),'diagnostic_header_sha256':sha(COPY/HPP_REL),
        'patch_sha256':sha(patch),'patch_path':str(patch),
        'changes':['logging-only path callback','logging-only TF lookup','logging-only pre/post control snapshots','diagnostic callback/path counters'],
        'control_branch_changed':False,'parameter_changed':False,'timer_changed':False,'publisher_or_subscriber_changed':False,
        'production_build_modified':False,
    })
    subprocess.run(['bash','-lc',f'source /opt/ros/jazzy/setup.bash && cmake -S {COPY} -B {BUILD} -DBUILD_TESTING=OFF && cmake --build {BUILD} -j4'],check=True)
    binary=BUILD/'simple_trajectory_follower_node'
    if not binary.exists():raise FileNotFoundError(binary)
    atomic_json(OUT/'diagnostic-build.json',{'binary':str(binary),'binary_sha256':sha(binary),'production_binary_sha256':audit['source_sha256']['follower_binary']['sha256']})
    print('DIAGNOSTIC_BUILD',binary)

if __name__=='__main__':main()
