"""Simulation-only exact-stamp command acceptance and reason accounting."""
from __future__ import annotations
import argparse, collections, math, time
from pathlib import Path
import rclpy
from eat_nav_sim_clock_bridge import SimClockNode, Server, stamp_ns
from visualnav_transformer.eat_nav.exact_stamp_protocol import ExactClockProtocol

class ExactNode(SimClockNode):
    def __init__(self, branches):
        super().__init__(branches)
        self.protocol = ExactClockProtocol()
        self.reject_reasons = collections.Counter()

    def on_command(self, i, msg):
        stamp = stamp_ns(msg.header.stamp)
        command = (float(msg.twist.linear.x), float(msg.twist.linear.y), float(msg.twist.angular.z))
        steady = time.monotonic_ns()
        with self.cv:
            reason = self.protocol.reject_reason(i, stamp)
            if reason is None and (self.tf_published_steady_ns is None or steady < self.tf_published_steady_ns):
                reason = 'pre_tf_publish'
            if reason is None and (self.path_published_steady_ns is None or steady < self.path_published_steady_ns):
                reason = 'pre_path_publish'
            if reason is None and not all(math.isfinite(x) for x in command):
                reason = 'nonfinite'
            accepted = reason is None and self.protocol.accept(i, stamp, command)
            if not accepted:
                reason = reason or 'protocol_rejected'
                self.reject_reasons[reason] += 1
                self.rejected[i] += 1
            else:
                self.commands[i] = list(command)
                self.sequence[i] += 1
                self.command_count[i] += 1
            self.events.append({'branch':i,'stamp_ns':stamp,'steady_ns':steady,
                'command':command,'accepted':accepted,'reject_reason':reason,
                'tf_stamp_ns':self.protocol.tf_stamp_ns,'path_stamp_ns':self.protocol.path_stamp_ns})
            self.cv.notify_all()

class ExactServer(Server):
    def __init__(self, args):
        self.args=args;self.log_dir=args.diagnostic_dir;self.log_dir.mkdir(parents=True,exist_ok=True)
        rclpy.init();self.node=ExactNode(args.branches)
        import threading
        self.spin=threading.Thread(target=rclpy.spin,args=(self.node,),daemon=True);self.spin.start()
        self.followers=[];self.logs=[];self.reset_number=0

    def snapshot(self):
        result=super().snapshot()
        with self.node.cv:result['reject_reasons']=dict(self.node.reject_reasons)
        return result

    def reset(self, request):
        self.node.reject_reasons.clear()
        return super().reset(request)

    def handle(self, request):
        if request['action'] == 'probe_delayed_tf':
            ns = int(request['stamp_ns'])
            self.node.clock(ns)
            # Three wall-timer callbacks observe a new clock and the previous TF.
            time.sleep(0.18)
            with self.node.cv:
                before = {'accepted_count': sum(self.node.command_count),
                          'events_at_stamp': [x for x in self.node.events if x['stamp_ns'] == ns]}
            self.node.poses(request['poses'], ns)
            result = self.wait_command(ns, float(request.get('timeout_sec', 3.)))
            result['before_tf_delivery'] = before
            return result
        return super().handle(request)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--host',default='127.0.0.1');parser.add_argument('--port',type=int,default=47819)
    parser.add_argument('--branches',type=int,default=8)
    parser.add_argument('--follower',type=Path,required=True)
    parser.add_argument('--params',type=Path,required=True)
    parser.add_argument('--diagnostic-dir',type=Path,required=True)
    ExactServer(parser.parse_args()).serve()

if __name__=='__main__':main()
