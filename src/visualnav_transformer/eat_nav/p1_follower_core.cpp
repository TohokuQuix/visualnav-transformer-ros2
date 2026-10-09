// ROS-independent P1 control core. Semantic reference: simple_trajectory_follower.cpp
// SHA 75249a834a90feb25fc2967e2e253eeaeffa498491c68692434bd08099a8d2be.
// The production node and binary are not modified. This core deliberately supports
// the single-Path, six-second P1 protocol; recovery states outside it are errors.
#include <algorithm>
#include <cmath>
#include <cstddef>
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

struct Follower {
  std::vector<Point> path;
  size_t closest=0;
  bool goal=false, direction_known=false;
  double direction=1;
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
  Projection project(Point robot) const {
    Projection best; double best_dist=std::numeric_limits<double>::max();
    double best_remaining=std::numeric_limits<double>::max();
    size_t start=std::min(closest,path.size()-1);
    for(size_t i=start;i+1<path.size();i++) {
      auto a=path[i],b=path[i+1]; double dx=b.x-a.x,dy=b.y-a.y,seg=std::hypot(dx,dy);
      if(seg<1e-6) continue;
      double ratio=std::clamp(((robot.x-a.x)*dx+(robot.y-a.y)*dy)/(seg*seg),0.,1.);
      Point point=interpolate(a,b,ratio);
      double dist=distance(robot,point),remaining=(1-ratio)*seg;
      for(size_t j=i+1;j+1<path.size();j++) remaining+=distance(path[j],path[j+1]);
      if(!best.found || dist<best_dist || (std::abs(dist-best_dist)<1e-9 && remaining<best_remaining)) {
        best={true,i,ratio,remaining,point}; best_dist=dist; best_remaining=remaining;
      }
    }
    return best;
  }
  Point carrot(Projection p) const {
    Point current=p.point; double remaining=.5;
    for(size_t i=p.segment+1;i<path.size();i++) {
      Point target=path[i]; double seg=distance(current,target);
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
  int step(double x,double y,double yaw,long long stamp,double *cmd) {
    if(!cmd || !std::isfinite(x) || !std::isfinite(y) || !std::isfinite(yaw) || stamp<0) return 1;
    if(stamp > 9'000'000'000LL) return 2; // P1 uses 2s start + 6s trace.
    cmd[0]=cmd[1]=cmd[2]=0;
    if(goal) return 0;
    Point robot{x,y}; auto p=project(robot);
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
    bool use_extension=p.remaining<=.5;
    bool has_tangent=tangent(path.size()-1,line_yaw);
    double goal_line=(x-path.back().x)*std::cos(line_yaw)+(y-path.back().y)*std::sin(line_yaw);
    if(distance(robot,path.back())<.2 || (use_extension && has_tangent && goal_line>=0)) {
      goal=true; return 0;
    }
    Point target=use_extension ? extension(robot,line_yaw) : carrot(p);
    double desired=std::atan2(target.y-y,target.x-x);
    double control=direction<0 ? angle(yaw+pi):yaw;
    double heading=angle(desired-control);
    cmd[2]=std::clamp(.5*heading,-1.6,1.6);
    if(std::abs(heading)<.6) cmd[0]=direction*.3*std::max(0.,1-std::abs(heading)/pi);
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
