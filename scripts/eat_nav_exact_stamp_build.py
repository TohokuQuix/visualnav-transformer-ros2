"""Build the simulation-only exact-TF follower from the immutable diagnostic copy."""
from __future__ import annotations
import difflib, hashlib, json, shutil, subprocess
from pathlib import Path

ROOT = Path('/tmp/eatnav-candidate-coverage/exact-stamp-long-horizon')
BASE = Path('/tmp/eatnav-candidate-coverage/tf-race-diagnostic/diagnostic-follower-source-r1')
COPY = ROOT/'exact-follower-source'
BUILD = ROOT/'exact-follower-build'
CPP = Path('src/simple_trajectory_follower.cpp')
HPP = Path('include/simple_trajectory_follower/simple_trajectory_follower.hpp')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f'expected one anchor, found {source.count(old)}: {old[:80]}')
    return source.replace(old, new, 1)

def render_exact(cpp, hpp):
    hpp = replace_once(hpp, '  FollowerStatus status_{FollowerStatus::Idle};',
        '  mutable int64_t last_emitted_stamp_ns_{-1};\n  FollowerStatus status_{FollowerStatus::Idle};')
    cpp = replace_once(cpp, 'tf_request\\\":\\\"TimePointZero', 'tf_request\\\":\\\"exact_stamp')
    cpp = replace_once(cpp, '  PathSource source)\n{\n  current_path_ = path;\n  active_path_source_ = source;',
        '  PathSource source)\n{\n  last_emitted_stamp_ns_ = -1;\n  current_path_ = path;\n  active_path_source_ = source;')
    # A new default Path is the episode-local reset event in the diagnostic.
    cpp = replace_once(cpp, '  const bool reset_progress = shouldResetProgressForPathUpdate(*path);',
        '  last_emitted_stamp_ns_ = -1;\n  const bool reset_progress = shouldResetProgressForPathUpdate(*path);')
    cpp = replace_once(cpp, '  diag_command_stamp_ns_ = 0;\n',
        '''  diag_command_stamp_ns_ = 0;
  const int64_t logical_stamp_ns = this->get_clock()->now().nanoseconds();
  if (logical_stamp_ns < last_emitted_stamp_ns_) last_emitted_stamp_ns_ = -1;
  if (last_emitted_stamp_ns_ == logical_stamp_ns) {
    diagEmit("stamp_latched");
    return;
  }
''')
    cpp = replace_once(cpp, '''    diag_tf_start_steady_ns_ = diagSteadyNs();
    const auto transform =
      tf_buffer_.lookupTransform(target_frame, robot_frame_, tf2::TimePointZero,
        tf2::durationFromSec(0.05));
    diag_tf_end_steady_ns_ = diagSteadyNs();
    diag_tf_success_ = true;
    diag_tf_stamp_ns_ = rclcpp::Time(transform.header.stamp).nanoseconds();''',
        '''    diag_tf_start_steady_ns_ = diagSteadyNs();
    const int64_t request_ns = this->get_clock()->now().nanoseconds();
    const auto request_time = tf2::TimePoint(std::chrono::nanoseconds(request_ns));
    if (!tf_buffer_.canTransform(target_frame, robot_frame_, request_time, tf2::durationFromSec(0.0))) {
      diag_tf_end_steady_ns_ = diagSteadyNs();
      return false;
    }
    const auto transform = tf_buffer_.lookupTransform(
      target_frame, robot_frame_, request_time, tf2::durationFromSec(0.0));
    diag_tf_end_steady_ns_ = diagSteadyNs();
    diag_tf_stamp_ns_ = rclcpp::Time(transform.header.stamp).nanoseconds();
    if (diag_tf_stamp_ns_ != request_ns) {
      RCLCPP_ERROR(this->get_logger(), "Exact TF mismatch request=%ld returned=%ld",
        request_ns, diag_tf_stamp_ns_);
      return false;
    }
    diag_tf_success_ = true;''')
    cpp = replace_once(cpp, '    cmd_vel_stamped_pub_->publish(stamped);',
        '''    const int64_t stamp_ns = rclcpp::Time(stamped.header.stamp).nanoseconds();
    if (last_emitted_stamp_ns_ == stamp_ns) return;
    cmd_vel_stamped_pub_->publish(stamped);
    last_emitted_stamp_ns_ = stamp_ns;''')
    return cpp, hpp

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    source_cpp, source_hpp = (BASE/CPP).read_text(), (BASE/HPP).read_text()
    cpp, hpp = render_exact(source_cpp, source_hpp)
    if not COPY.exists():
        shutil.copytree(BASE, COPY)
    for relative, content in ((CPP, cpp), (HPP, hpp)):
        dst = COPY/relative
        if dst.read_text() not in (content, (BASE/relative).read_text()):
            raise ValueError(f'existing exact follower source differs: {dst}')
        dst.write_text(content)
    patch = ''.join(difflib.unified_diff(source_cpp.splitlines(True), cpp.splitlines(True), fromfile='diagnostic.cpp', tofile='exact.cpp'))
    patch += ''.join(difflib.unified_diff(source_hpp.splitlines(True), hpp.splitlines(True), fromfile='diagnostic.hpp', tofile='exact.hpp'))
    (ROOT/'exact-follower.patch').write_text(patch)
    with (ROOT/'build.log').open('w') as log:
        subprocess.run(['bash', '-lc', f'source /opt/ros/jazzy/setup.bash && cmake -S {COPY} -B {BUILD} -DBUILD_TESTING=OFF && cmake --build {BUILD} -j4'], stdout=log, stderr=subprocess.STDOUT, check=True)
    data = {'source_cpp_sha256': sha(COPY/CPP), 'source_header_sha256': sha(COPY/HPP),
            'binary_sha256': sha(BUILD/'simple_trajectory_follower_node'), 'patch_sha256': sha(ROOT/'exact-follower.patch')}
    (ROOT/'build.json').write_text(json.dumps(data, indent=2, sort_keys=True)+'\n')
    print(json.dumps(data, sort_keys=True))

if __name__ == '__main__': main()
