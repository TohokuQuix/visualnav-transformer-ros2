// ROS-independent P1 control core. Semantic reference: simple_trajectory_follower.cpp
// SHA 75249a834a90feb25fc2967e2e253eeaeffa498491c68692434bd08099a8d2be.
// The production node and binary are not modified. Single-Path episode-local core.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
constexpr double pi = 3.14159265358979323846;
struct Point { double x, y; };
double distance(Point a, Point b) { return std::hypot(a.x-b.x, a.y-b.y); }
Point interpolate(Point a, Point b, double r) { return {a.x+(b.x-a.x)*r,a.y+(b.y-a.y)*r}; }
double angle(double a) { while (a>pi) a-=2*pi; while(a<-pi) a+=2*pi; return a; }
struct Projection { bool found=false; size_t segment=0; double ratio=0, remaining=0; Point point{}; };
struct MotionSample { long long stamp; double commanded_linear, actual_linear, commanded_angular, actual_angular; };
// Stable C diagnostic ABI. All flags use 0/1; cause is 0 none, 1 progress, 2 linear.
enum Diag { CARROT_X, CARROT_Y, PROJECTION_X, PROJECTION_Y, REMAINING,
  HEADING, DIRECTION, GOAL, LINEAR_RECOVERY, ANGULAR_RECOVERY,
  LINEAR_STARTED, LINEAR_FINISHED, ANGULAR_STARTED, CAUSE,
  COMMANDED_LINEAR, ACTUAL_LINEAR, COMMANDED_ANGULAR, ACTUAL_ANGULAR,
  MOTION_READY, BEST_REMAINING, PROGRESS_AGE, BACKTRACK_POINTS,
  BACKTRACK_LENGTH, USE_EXTENSION, STAMP_AGE, PRE_MOTION_SAMPLES,
  PRE_MOTION_OLDEST_NS, PRE_MOTION_NEWEST_NS, PRE_MOTION_READY,
  PRE_COMMANDED_LINEAR, PRE_ACTUAL_LINEAR, PRE_MOTION_RATIO,
  GOAL_DISTANCE, TURN_IN_PLACE, PROJECTION_SEGMENT, DIAG_COUNT };

struct Follower {
  std::vector<Point> path;
  size_t closest=0;
  bool goal=false, direction_known=false;
  double direction=1;
  long long first_stamp=-1, last_stamp=-1, progress_stamp=-1, boost_until=-1;
  bool recovery=false, motion_reference=false;
  double best_remaining=0, recovery_distance=.3;
  Point previous_position{};
  double previous_yaw=0;
  long long previous_motion_stamp=-1;
  std::deque<MotionSample> motion;
  double commanded_linear=0, actual_linear=0, commanded_angular=0, actual_angular=0;
  std::deque<Point> carrot_history;
  std::vector<Point> backtrack;
  double diag[DIAG_COUNT]{};
  explicit Follower(const double *xy, size_t n) {
    if (!xy || n<2) throw std::invalid_argument("Path requires two or more points");
    for (size_t i=0;i<n;i++) {
      if (!std::isfinite(xy[2*i]) || !std::isfinite(xy[2*i+1]))
        throw std::invalid_argument("nonfinite Path");
      path.push_back({xy[2*i],xy[2*i+1]});
    }
  }
  bool tangent(size_t reference, double &yaw) const {
    reference=std::min(reference,path.size()-1);
    auto accumulate=[&](bool forward) {
      double length=0,dx=0,dy=0;
      if (forward) {
        for(size_t i=reference;i+1<path.size() && length<.5;i++) {
          auto a=path[i],b=path[i+1]; double seg=distance(a,b);
          if(seg<1e-6) continue;
          double take=std::min(seg,.5-length);
          dx+=(b.x-a.x)/seg*take; dy+=(b.y-a.y)/seg*take; length+=take;
        }
      } else {
        for(size_t i=reference;i>0 && length<.5;i--) {
          auto a=path[i-1],b=path[i]; double seg=distance(a,b);
          if(seg<1e-6) continue;
          double take=std::min(seg,.5-length);
          dx+=(b.x-a.x)/seg*take; dy+=(b.y-a.y)/seg*take; length+=take;
        }
      }
      if(std::hypot(dx,dy)<1e-6) return false;
      yaw=std::atan2(dy,dx); return true;
    };
    return accumulate(true) || accumulate(false);
  }
  Projection project(Point robot,const std::vector<Point> &points,size_t start_index) const {
    Projection best; double best_dist=std::numeric_limits<double>::max();
    double best_remaining=std::numeric_limits<double>::max();
    size_t start=std::min(start_index,points.size()-1);
    for(size_t i=start;i+1<points.size();i++) {
      auto a=points[i],b=points[i+1]; double dx=b.x-a.x,dy=b.y-a.y,seg=std::hypot(dx,dy);
      if(seg<1e-6) continue;
      double ratio=std::clamp(((robot.x-a.x)*dx+(robot.y-a.y)*dy)/(seg*seg),0.,1.);
      Point point=interpolate(a,b,ratio);
      double dist=distance(robot,point),remaining=(1-ratio)*seg;
      for(size_t j=i+1;j+1<points.size();j++) remaining+=distance(points[j],points[j+1]);
      if(!best.found || dist<best_dist || (std::abs(dist-best_dist)<1e-9 && remaining<best_remaining)) {
        best={true,i,ratio,remaining,point}; best_dist=dist; best_remaining=remaining;
      }
    }
    return best;
  }
  Point carrot(Projection p,const std::vector<Point> &points,bool backwards=false) const {
    Point current=p.point; double remaining=.5;
    if(backwards) {
      for(size_t i=p.segment+1;i>0;i--) {
        Point target=points[i-1]; double seg=distance(current,target);
        if(seg<1e-6) {current=target;continue;}
        if(remaining<=seg) return interpolate(current,target,remaining/seg);
        remaining-=seg;current=target;
      }
      return points.front();
    }
    for(size_t i=p.segment+1;i<points.size();i++) {
      Point target=points[i]; double seg=distance(current,target);
      if(seg<1e-6) { current=target; continue; }
      if(remaining<=seg) return interpolate(current,target,remaining/seg);
      remaining-=seg; current=target;
    }
    return path.back();
  }
  Point extension(Point robot, double line_yaw) const {
    auto goal=path.back(); double lx=std::cos(line_yaw),ly=std::sin(line_yaw);
    double projection=(robot.x-goal.x)*lx+(robot.y-goal.y)*ly;
    return {goal.x+(projection+.5)*lx,goal.y+(projection+.5)*ly};
  }
  void reset_motion() {
    motion.clear(); commanded_linear=actual_linear=commanded_angular=actual_angular=0;
    motion_reference=false; previous_motion_stamp=-1;
  }
  void reset_progress() {progress_stamp=-1;best_remaining=0;}
  void update_progress(long long stamp,double remaining) {
    if(progress_stamp<0) {progress_stamp=stamp;best_remaining=remaining;}
    else if(remaining+.1<best_remaining) {progress_stamp=stamp;best_remaining=remaining;}
  }
  void update_motion(long long stamp,Point robot,double yaw,double vx,double wz) {
    if(!motion_reference) {motion_reference=true;previous_motion_stamp=stamp;previous_position=robot;previous_yaw=yaw;return;}
    double dt=double(stamp-previous_motion_stamp)*1e-9;
    if(dt<=0) {previous_motion_stamp=stamp;previous_position=robot;previous_yaw=yaw;return;}
    MotionSample s{stamp,std::abs(vx)*dt,distance(robot,previous_position),std::abs(wz)*dt,std::abs(angle(yaw-previous_yaw))};
    motion.push_back(s);commanded_linear+=s.commanded_linear;actual_linear+=s.actual_linear;
    commanded_angular+=s.commanded_angular;actual_angular+=s.actual_angular;
    previous_motion_stamp=stamp;previous_position=robot;previous_yaw=yaw;
    while(!motion.empty() && stamp-motion.front().stamp>20'000'000'000LL) {
      auto old=motion.front();motion.pop_front();
      commanded_linear-=old.commanded_linear;actual_linear-=old.actual_linear;
      commanded_angular-=old.commanded_angular;actual_angular-=old.actual_angular;
    }
    commanded_linear=std::max(0.,commanded_linear);actual_linear=std::max(0.,actual_linear);
    commanded_angular=std::max(0.,commanded_angular);actual_angular=std::max(0.,actual_angular);
  }
  bool motion_ready() const {return !motion.empty() && motion.back().stamp-motion.front().stamp>=20'000'000'000LL;}
  void record_carrot(Point target,bool extension) {
    if(extension || recovery) return;
    if(!carrot_history.empty() && distance(carrot_history.back(),target)<.2) return;
    carrot_history.push_back(target);
    auto length=[&]() {double n=0;for(size_t i=1;i<carrot_history.size();i++) n+=distance(carrot_history[i-1],carrot_history[i]);return n;};
    while(carrot_history.size()>1 && length()>5.) carrot_history.pop_front();
  }
  void build_backtrack(Point robot) {
    backtrack.clear();
    if(carrot_history.size()<2) return;
    size_t closest_index=0;double d=std::numeric_limits<double>::max();
    for(size_t i=0;i<carrot_history.size();i++) if(distance(robot,carrot_history[i])<d) {d=distance(robot,carrot_history[i]);closest_index=i;}
    if(closest_index==0) return;
    std::vector<Point> newest;newest.push_back(carrot_history[closest_index]);double accumulated=0;
    for(size_t i=closest_index;i>0 && accumulated<.8;i--) {
      Point a=carrot_history[i],b=carrot_history[i-1];double segment=distance(a,b);
      if(segment<1e-6) continue;
      if(accumulated+segment<=.8) {newest.push_back(b);accumulated+=segment;}
      else {double remaining=.8-accumulated;if(remaining>1e-6)newest.push_back(interpolate(a,b,remaining/segment));accumulated=.8;}
    }
    if(newest.size()>=2) backtrack.assign(newest.rbegin(),newest.rend());
  }
  double backtrack_length() const {double n=0;for(size_t i=1;i<backtrack.size();i++)n+=distance(backtrack[i-1],backtrack[i]);return n;}
  void begin_recovery(Point robot,int cause) {
    build_backtrack(robot);recovery=true;boost_until=-1;reset_motion();reset_progress();diag[LINEAR_STARTED]=1;diag[CAUSE]=cause;
  }
  void finish_recovery() {recovery=false;backtrack.clear();closest=0;reset_motion();reset_progress();diag[LINEAR_FINISHED]=1;}
  void fill_common(long long stamp) {
    diag[GOAL]=goal;diag[LINEAR_RECOVERY]=recovery;diag[ANGULAR_RECOVERY]=boost_until>stamp;
    diag[DIRECTION]=recovery?-direction:direction;
    diag[COMMANDED_LINEAR]=commanded_linear;diag[ACTUAL_LINEAR]=actual_linear;
    diag[COMMANDED_ANGULAR]=commanded_angular;diag[ACTUAL_ANGULAR]=actual_angular;
    diag[MOTION_READY]=motion_ready();diag[BEST_REMAINING]=best_remaining;
    diag[PROGRESS_AGE]=progress_stamp<0?0:double(stamp-progress_stamp)*1e-9;
    diag[BACKTRACK_POINTS]=backtrack.size();diag[BACKTRACK_LENGTH]=backtrack_length();
    diag[STAMP_AGE]=first_stamp<0?0:double(stamp-first_stamp)*1e-9;
  }
  void fill_pre_motion() {
    diag[PRE_MOTION_SAMPLES]=motion.size();
    diag[PRE_MOTION_OLDEST_NS]=motion.empty()?0:motion.front().stamp;
    diag[PRE_MOTION_NEWEST_NS]=motion.empty()?0:motion.back().stamp;
    diag[PRE_MOTION_READY]=motion_ready();
    diag[PRE_COMMANDED_LINEAR]=commanded_linear;
    diag[PRE_ACTUAL_LINEAR]=actual_linear;
    diag[PRE_MOTION_RATIO]=commanded_linear>1e-6?actual_linear/commanded_linear:0;
  }
  int step(double x,double y,double yaw,long long stamp,double *cmd) {
    if(!cmd || !std::isfinite(x) || !std::isfinite(y) || !std::isfinite(yaw) || stamp<0) return 1;
    if(last_stamp>=0 && stamp<=last_stamp) return 2;
    if(first_stamp<0) first_stamp=stamp;
    last_stamp=stamp;
    std::fill(diag,diag+DIAG_COUNT,0.);
    cmd[0]=cmd[1]=cmd[2]=0;
    fill_pre_motion();
    if(goal) {fill_common(stamp);return 0;}
    Point robot{x,y};const auto &source=recovery && !backtrack.empty()?backtrack:path;
    auto p=project(robot,source,recovery?0:closest);
    if(!p.found) return 3;
    double line_yaw=std::atan2(path.back().y-y,path.back().x-x);
    double tangent_yaw=0;
    if(tangent(p.segment,tangent_yaw)) {
      double error=std::abs(angle(tangent_yaw-yaw));
      if(!direction_known) { direction=error<=pi/2 ? 1:-1; direction_known=true; }
      else if(error<=1.3962634015954636) direction=1;
      else if(error>=1.7453292519943295) direction=-1;
    }
    closest=p.segment;
    bool use_extension=!recovery && p.remaining<=.5;
    bool has_tangent=tangent(path.size()-1,line_yaw);
    double goal_line=(x-path.back().x)*std::cos(line_yaw)+(y-path.back().y)*std::sin(line_yaw);
    diag[GOAL_DISTANCE]=distance(robot,path.back());
    if(distance(robot,path.back())<.2 || (use_extension && has_tangent && goal_line>=0)) {
      goal=true;fill_common(stamp);return 0;
    }
    if(!recovery) update_progress(stamp,p.remaining);
    Point target=use_extension ? extension(robot,line_yaw) : carrot(p,source,recovery);
    record_carrot(target,use_extension);
    diag[CARROT_X]=target.x;diag[CARROT_Y]=target.y;
    diag[PROJECTION_X]=p.point.x;diag[PROJECTION_Y]=p.point.y;diag[REMAINING]=p.remaining;diag[USE_EXTENSION]=use_extension;
    diag[PROJECTION_SEGMENT]=p.segment;
    double desired=std::atan2(target.y-y,target.x-x);
    double active_direction=recovery?-direction:direction;
    double control=active_direction<0 ? angle(yaw+pi):yaw;
    double heading=angle(desired-control);
    diag[HEADING]=heading;
    cmd[2]=std::clamp(.5*heading,-1.6,1.6);
    diag[TURN_IN_PLACE]=std::abs(heading)>=.6;
    if(std::abs(heading)<.6) cmd[0]=active_direction*.3*std::max(0.,1-std::abs(heading)/pi);
    if(!recovery && progress_stamp>=0 && p.remaining>.2 && stamp-progress_stamp>=20'000'000'000LL) {
      begin_recovery(robot,1);cmd[0]=-direction*.3;
    }
    if(!recovery && motion_ready() && commanded_linear>=1e-3 && actual_linear/commanded_linear<=.25) {
      begin_recovery(robot,2);cmd[0]=-direction*.3;
    }
    if(!recovery && motion_ready() && commanded_angular>=1e-3 && actual_angular/commanded_angular<=.25 && stamp>=boost_until) {
      boost_until=stamp+2'000'000'000LL;reset_motion();diag[ANGULAR_STARTED]=1;
    }
    if(stamp<boost_until) cmd[0]=active_direction*std::min(.3,std::abs(cmd[0])+.1);
    update_motion(stamp,robot,yaw,cmd[0],cmd[2]);
    if(recovery && (actual_linear>=recovery_distance || distance(robot,backtrack.empty()?path.front():backtrack.front())<=.2)) finish_recovery();
    fill_common(stamp);
    return 0;
  }
};
}

extern "C" {
void *p1_follower_create(const double *xy,size_t count) {
  try {return new Follower(xy,count);} catch(...) {return nullptr;}
}
int p1_follower_step(void *instance,double x,double y,double yaw,long long stamp,double *cmd) {
  if(!instance) return 4;
  try {return static_cast<Follower*>(instance)->step(x,y,yaw,stamp,cmd);} catch(...) {return 5;}
}
int p1_follower_step_diagnostic(void *instance,double x,double y,double yaw,long long stamp,double *cmd,double *out,size_t count) {
  if(!instance || !out || count<DIAG_COUNT) return 6;
  try {
    auto *f=static_cast<Follower*>(instance);int rc=f->step(x,y,yaw,stamp,cmd);
    if(rc==0) std::copy(f->diag,f->diag+DIAG_COUNT,out);
    return rc;
  } catch(...) {return 5;}
}
size_t p1_follower_diagnostic_count() {return DIAG_COUNT;}
void *p1_follower_clone(void *instance) {
  if(!instance) return nullptr;
  try {return new Follower(*static_cast<Follower*>(instance));} catch(...) {return nullptr;}
}
int p1_follower_get_state(void *instance,double *out) {
  if(!instance || !out) return 1;
  const auto &f=*static_cast<Follower*>(instance);
  out[0]=static_cast<double>(f.closest);
  out[1]=f.goal ? 1.0 : 0.0;
  out[2]=f.direction_known ? 1.0 : 0.0;
  out[3]=f.direction;
  return 0;
}
int p1_follower_set_state(void *instance,const double *state) {
  if(!instance || !state) return 1;
  auto &f=*static_cast<Follower*>(instance);
  for(int i=0;i<4;i++) if(!std::isfinite(state[i])) return 2;
  if(state[0]<0 || state[0]>=static_cast<double>(f.path.size()) ||
     std::floor(state[0])!=state[0] ||
     (state[1]!=0 && state[1]!=1) || (state[2]!=0 && state[2]!=1) ||
     (state[3]!=1 && state[3]!=-1)) return 3;
  f.closest=static_cast<size_t>(state[0]);
  f.goal=state[1]!=0;
  f.direction_known=state[2]!=0;
  f.direction=state[3];
  return 0;
}
void p1_follower_destroy(void *instance) {delete static_cast<Follower*>(instance);}
}
