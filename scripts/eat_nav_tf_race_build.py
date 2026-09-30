"""Build a logging-only follower copy; production source and binary stay immutable."""
from __future__ import annotations
import difflib,hashlib,json,shutil,subprocess
from pathlib import Path
ROOT=Path('/tmp/eatnav-candidate-coverage/tf-race-diagnostic')
PROD=Path('/home/kojima/onix/dep/onix_automation/teaching_playback/simple_trajectory_follower')
COPY=ROOT/'diagnostic-follower-source-r1';BUILD=ROOT/'diagnostic-follower-build-r1'
CPP=Path('src/simple_trajectory_follower.cpp');HPP=Path('include/simple_trajectory_follower/simple_trajectory_follower.hpp')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def repl(s,a,b):
 if s.count(a)!=1:raise ValueError(f'expected one anchor, got {s.count(a)}: {a[:100]}')
 return s.replace(a,b,1)
def main():
 audit=json.loads((ROOT/'tf-race-source-audit.json').read_text())['sources']
 assert sha(PROD/CPP)==audit['production_follower_cpp']['sha256']
 assert sha(PROD/HPP)==audit['production_follower_header']['sha256']
 oc=(PROD/CPP).read_text();oh=(PROD/HPP).read_text();c=oc;h=oh
 h=repl(h,'  FollowerStatus status_{FollowerStatus::Idle};', '''  // Logging-only callback state; no controller branch reads these fields.
  uint64_t diag_callback_seq_{0};
  uint64_t diag_path_seq_{0};
  uint64_t diag_publish_seq_{0};
  int64_t diag_callback_steady_ns_{0};
  int64_t diag_path_received_steady_ns_{0};
  int64_t diag_path_stamp_ns_{0};
  int64_t diag_tf_start_steady_ns_{0};
  int64_t diag_tf_end_steady_ns_{0};
  int64_t diag_tf_stamp_ns_{0};
  int64_t diag_command_stamp_ns_{0};
  bool diag_tf_success_{false};
  double diag_tf_x_{0.0};
  double diag_tf_y_{0.0};
  double diag_tf_z_{0.0};
  double diag_tf_yaw_{0.0};
  double diag_command_vx_{0.0};
  double diag_command_wz_{0.0};
  double diag_carrot_x_{0.0};
  double diag_carrot_y_{0.0};
  double diag_goal_distance_{0.0};
  double diag_heading_error_{0.0};
  bool diag_turn_in_place_{false};
  void diagEmit(const char * phase) const;
  FollowerStatus status_{FollowerStatus::Idle};''')
 c=repl(c,'#include <algorithm>\n','#include <algorithm>\n#include <chrono>\n#include <cstdio>\n#include <unistd.h>\n')
 c=repl(c,'namespace simple_trajectory_follower\n{','''namespace simple_trajectory_follower
{
namespace {
int64_t diagSteadyNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
}
int64_t diagSystemNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::system_clock::now().time_since_epoch()).count();
}
}  // namespace

void SimpleTrajectoryFollower::diagEmit(const char * phase) const
{
  const int64_t ros_ns = this->get_clock()->now().nanoseconds();
  const auto & p = current_path_;
  const auto & goal = p && !p->poses.empty() ? p->poses.back().pose.position : current_pose_.position;
  const int64_t oldest = motion_history_.empty() ? 0 : motion_history_.front().stamp.nanoseconds();
  const int64_t newest = motion_history_.empty() ? 0 : motion_history_.back().stamp.nanoseconds();
  const double ratio = summed_commanded_linear_distance_ > 1e-9 ?
    summed_actual_linear_distance_ / summed_commanded_linear_distance_ : 0.0;
  std::fprintf(stderr,
    "TF_RACE_JSON {\\"phase\\":\\"%s\\",\\"pid\\":%d,\\"callback_seq\\":%lu,\\"wall_steady_ns\\":%ld,\\"wall_system_ns\\":%ld,\\"ros_clock_ns\\":%ld,\\"use_sim_time\\":%s,"
    "\\"path_seq\\":%lu,\\"path_stamp_ns\\":%ld,\\"path_received_steady_ns\\":%ld,\\"path_points\\":%zu,\\"goal_x\\":%.17g,\\"goal_y\\":%.17g,\\"goal_reached\\":%s,\\"mode\\":\\"%s\\","
    "\\"tf_request\\":\\"TimePointZero\\",\\"tf_start_steady_ns\\":%ld,\\"tf_end_steady_ns\\":%ld,\\"tf_success\\":%s,\\"tf_stamp_ns\\":%ld,\\"tf_parent\\":\\"map\\",\\"tf_child\\":\\"%s\\",\\"tf_x\\":%.17g,\\"tf_y\\":%.17g,\\"tf_z\\":%.17g,\\"tf_yaw\\":%.17g,\\"tf_age_ns\\":%ld,"
    "\\"carrot_x\\":%.17g,\\"carrot_y\\":%.17g,\\"goal_distance\\":%.17g,\\"heading_error\\":%.17g,\\"turn_in_place\\":%s,\\"vx\\":%.17g,\\"wz\\":%.17g,\\"command_stamp_ns\\":%ld,\\"publish_seq\\":%lu,\\"linear_recovery\\":%s,\\"angular_recovery\\":%s,"
    "\\"motion_oldest_ns\\":%ld,\\"motion_newest_ns\\":%ld,\\"motion_elapsed_sec\\":%.17g,\\"motion_samples\\":%zu,\\"commanded_distance\\":%.17g,\\"actual_distance\\":%.17g,\\"motion_ratio\\":%.17g,\\"motion_ready\\":%s}\\n",
    phase, getpid(), diag_callback_seq_, diag_callback_steady_ns_, diagSystemNs(), ros_ns,
    this->get_parameter("use_sim_time").as_bool() ? "true" : "false",
    diag_path_seq_, diag_path_stamp_ns_, diag_path_received_steady_ns_, p ? p->poses.size() : 0,
    goal.x, goal.y, goal_reached_ ? "true" : "false", pathSourceName(active_path_source_),
    diag_tf_start_steady_ns_, diag_tf_end_steady_ns_, diag_tf_success_ ? "true" : "false",
    diag_tf_stamp_ns_, robot_frame_.c_str(), diag_tf_x_, diag_tf_y_, diag_tf_z_, diag_tf_yaw_,
    diag_tf_success_ ? ros_ns-diag_tf_stamp_ns_ : 0,
    diag_carrot_x_, diag_carrot_y_, diag_goal_distance_, diag_heading_error_,
    diag_turn_in_place_ ? "true" : "false", diag_command_vx_, diag_command_wz_,
    diag_command_stamp_ns_, diag_publish_seq_, linear_stuck_recovery_active_ ? "true" : "false",
    ros_ns < angular_stuck_boost_until_.nanoseconds() ? "true" : "false",
    oldest, newest, motion_history_.empty() ? 0.0 : (newest-oldest)*1e-9, motion_history_.size(),
    summed_commanded_linear_distance_, summed_actual_linear_distance_, ratio,
    isMotionWindowReady() ? "true" : "false");
  std::fflush(stderr);
}''')
 c=repl(c,'void SimpleTrajectoryFollower::pathCallback(const nav_msgs::msg::Path::SharedPtr path)\n{', '''void SimpleTrajectoryFollower::pathCallback(const nav_msgs::msg::Path::SharedPtr path)
{
  ++diag_path_seq_;
  diag_path_stamp_ns_ = rclcpp::Time(path->header.stamp).nanoseconds();
  diag_path_received_steady_ns_ = diagSteadyNs();
  std::fprintf(stderr, "TF_RACE_PATH {\\"pid\\":%d,\\"path_seq\\":%lu,\\"stamp_ns\\":%ld,\\"received_steady_ns\\":%ld,\\"points\\":%zu}\\n",
    getpid(), diag_path_seq_, diag_path_stamp_ns_, diag_path_received_steady_ns_, path->poses.size());
  std::fflush(stderr);''')
 c=repl(c,'void SimpleTrajectoryFollower::controlLoop()\n{', '''void SimpleTrajectoryFollower::controlLoop()
{
  ++diag_callback_seq_;
  diag_callback_steady_ns_ = diagSteadyNs();
  diag_tf_start_steady_ns_ = 0;
  diag_tf_end_steady_ns_ = 0;
  diag_tf_stamp_ns_ = 0;
  diag_tf_success_ = false;
  diag_command_stamp_ns_ = 0;''')
 c=repl(c,'    clearCarrotMarker();\n    return;\n  }\n\n  if (!updateRobotPoseInPathFrame())',
  '    clearCarrotMarker();\n    diagEmit("no_path");\n    return;\n  }\n\n  if (!updateRobotPoseInPathFrame())')
 c=repl(c,'  if (!updateRobotPoseInPathFrame()) {\n    return;\n  }',
  '  if (!updateRobotPoseInPathFrame()) {\n    diagEmit("tf_failure");\n    return;\n  }')
 # All publish-return branches log immediately before returning.
 c=repl(c,'      clearCarrotMarker();\n      return;\n    }\n    setStatus(FollowerStatus::Goal);',
  '      clearCarrotMarker();\n      diagEmit("recovery_path_done");\n      return;\n    }\n    setStatus(FollowerStatus::Goal);')
 c=repl(c,'    clearCarrotMarker();\n    return;\n  }\n\n  const double dx_goal',
  '    clearCarrotMarker();\n    diagEmit("goal_hold");\n    return;\n  }\n\n  const double dx_goal')
 c=repl(c,'      clearCarrotMarker();\n      return;\n    }\n    goal_reached_ = true;',
  '      clearCarrotMarker();\n      diagEmit("recovery_goal");\n      return;\n    }\n    goal_reached_ = true;')
 c=repl(c,'    clearCarrotMarker();\n    return;\n  }\n\n  if (!linear_stuck_recovery_active_)',
  '    clearCarrotMarker();\n    diagEmit("goal_reached");\n    return;\n  }\n\n  if (!linear_stuck_recovery_active_)')
 c=repl(c,'  const double heading_error = normalizeAngle(desired_motion_yaw - control_yaw);',
  '''  const double heading_error = normalizeAngle(desired_motion_yaw - control_yaw);
  diag_carrot_x_ = carrot.x; diag_carrot_y_ = carrot.y;
  diag_goal_distance_ = distance_to_goal; diag_heading_error_ = heading_error;''')
 c=repl(c,'  if (abs_heading_error < turn_in_place_heading_threshold_) {',
  '  diag_turn_in_place_ = !(abs_heading_error < turn_in_place_heading_threshold_);\n  if (abs_heading_error < turn_in_place_heading_threshold_) {')
 c=repl(c,'  if (publish_subcrawler_commands_) {\n    publishSubcrawlerCommand();\n  }\n}\n\nvoid SimpleTrajectoryFollower::publishStatus()',
  '  if (publish_subcrawler_commands_) {\n    publishSubcrawlerCommand();\n  }\n  diagEmit("tracking");\n}\n\nvoid SimpleTrajectoryFollower::publishStatus()')
 c=repl(c,'    const auto transform =\n      tf_buffer_.lookupTransform(target_frame, robot_frame_, tf2::TimePointZero,',
  '    diag_tf_start_steady_ns_ = diagSteadyNs();\n    const auto transform =\n      tf_buffer_.lookupTransform(target_frame, robot_frame_, tf2::TimePointZero,')
 c=repl(c,'    current_pose_.position.x = transform.transform.translation.x;',
  '''    diag_tf_end_steady_ns_ = diagSteadyNs();
    diag_tf_success_ = true;
    diag_tf_stamp_ns_ = rclcpp::Time(transform.header.stamp).nanoseconds();
    diag_tf_x_ = transform.transform.translation.x;
    diag_tf_y_ = transform.transform.translation.y;
    diag_tf_z_ = transform.transform.translation.z;
    diag_tf_yaw_ = quaternionToYaw(transform.transform.rotation);
    current_pose_.position.x = transform.transform.translation.x;''')
 c=repl(c,'  } catch (const tf2::TransformException & ex) {\n    RCLCPP_WARN_THROTTLE(',
  '  } catch (const tf2::TransformException & ex) {\n    diag_tf_end_steady_ns_ = diagSteadyNs();\n    RCLCPP_WARN_THROTTLE(')
 c=repl(c,'    stamped.header.stamp = this->get_clock()->now();',
  '''    stamped.header.stamp = this->get_clock()->now();
    diag_command_stamp_ns_ = rclcpp::Time(stamped.header.stamp).nanoseconds();
    diag_command_vx_ = cmd.linear.x; diag_command_wz_ = cmd.angular.z;
    ++diag_publish_seq_;
    diagEmit("command_pre_publish");''')
 # publishVelocityCommand is const; all changed diagnostic members must be mutable.
 h=h.replace('  uint64_t diag_', '  mutable uint64_t diag_').replace('  int64_t diag_', '  mutable int64_t diag_').replace('  bool diag_', '  mutable bool diag_').replace('  double diag_', '  mutable double diag_')
 if COPY.exists():
  assert (COPY/CPP).read_text()==c and (COPY/HPP).read_text()==h
 else:
  shutil.copytree(PROD,COPY);(COPY/CPP).write_text(c);(COPY/HPP).write_text(h)
 patch=''.join(difflib.unified_diff(oc.splitlines(True),c.splitlines(True),fromfile='production.cpp',tofile='diagnostic.cpp'))
 patch+=''.join(difflib.unified_diff(oh.splitlines(True),h.splitlines(True),fromfile='production.hpp',tofile='diagnostic.hpp'))
 (ROOT/'tf-race-instrumentation.patch').write_text(patch)
 (ROOT/'tf-race-instrumentation-diff.json').write_text(json.dumps({
  'production_cpp_sha256':sha(PROD/CPP),'production_header_sha256':sha(PROD/HPP),
  'diagnostic_cpp_sha256':sha(COPY/CPP),'diagnostic_header_sha256':sha(COPY/HPP),
  'patch_sha256':sha(ROOT/'tf-race-instrumentation.patch'),
  'control_branches_changed':False,'parameters_changed':False,'timer_changed':False,
  'publisher_subscriber_changed':False,'tf_lookup_timepoint_changed':False,
  'logging_added':['path callback','TF lookup result','callback output and motion state']},indent=2,sort_keys=True)+'\n')
 subprocess.run(['bash','-lc',f'source /opt/ros/jazzy/setup.bash && cmake -S {COPY} -B {BUILD} -DBUILD_TESTING=OFF && cmake --build {BUILD} -j4'],check=True)
 (ROOT/'tf-race-build.json').write_text(json.dumps({'binary':str(BUILD/'simple_trajectory_follower_node'),
  'binary_sha256':sha(BUILD/'simple_trajectory_follower_node')},indent=2)+'\n')
 print('TF_RACE_BUILD',BUILD/'simple_trajectory_follower_node')
if __name__=='__main__':main()
