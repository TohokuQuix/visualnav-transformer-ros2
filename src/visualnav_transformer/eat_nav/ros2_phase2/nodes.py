"""CPU-only ROS2 Phase 2 nodes. All actuator outputs use /eat_nav/mock/."""

import math
import json
from pathlib import Path

import numpy as np
import rclpy
from rclpy.clock import Clock
from rclpy.clock_type import ClockType
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.time import Time
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Empty
from tf2_ros import Buffer, TransformListener

from eat_nav_interfaces.msg import (DemFrame, ActorState22, CandidateBank,
                                    CandidateEvaluation, GoalState,
                                    NavigationStatus, PathSelection)
from .gateway import MockFollowerGateway
from .history import DemHistory
from .persistent_worker import ModelWorkerClient
from .predictor_client import (mock14_predict, legacy_worker_request,
                               prepare_model_input, worker_request, validate_mode,
                               sha_file, validate_adoption_artifacts)
from .selection import score_global_goal
from .supervisor import Supervisor
from .wire import (BANK_SHA, WIRE_SHA, CANDIDATE_IDS, PATHS,
                   bank_to_message, set_stamp, stamp_ns)


QOS = QoSProfile(depth=8, reliability=ReliabilityPolicy.RELIABLE,
                 durability=DurabilityPolicy.VOLATILE)
BANK_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL)


def build_mock_evaluation(history, actor, now_ns, generation,
                          robot_pose_map, goal_map):
    history.assemble(actor, now_ns)
    if generation <= 0:
        raise ValueError('inactive goal')
    prediction = mock14_predict(PATHS)
    result = score_global_goal(prediction,robot_pose_map,goal_map,
                               np.ones(14,dtype=bool))
    msg = CandidateEvaluation()
    msg.header.frame_id = 'chassis/body'
    msg.header.stamp = actor.header.stamp
    msg.generation = generation
    msg.predictor_mode = 'mock14_transport'
    msg.bank_source_sha256 = BANK_SHA
    msg.checkpoint_sha256 = ''
    msg.predicted_body_xy_m = prediction.reshape(-1).tolist()
    msg.predicted_yaw_valid = False
    msg.predicted_delta_yaw_rad = [0.0] * 14
    msg.scores_m = result['scores_m'].tolist()
    msg.mask = [True]*14
    msg.selected_index = result['index']
    return msg


def build_model_evaluation(history, actor, now_ns, generation,
                           robot_pose_map, goal_map, *, mode, checkpoint,
                           python, results=None, normalization=None, worker=None,
                           predictor_zero_velocity_command=True,
                           predict_yaw=False):
    dem,state,_=history.assemble(actor,now_ns)
    validate_mode(mode,14 if mode=='learned14_adopted' else 7,checkpoint)
    yaw=np.zeros(14,np.float32)
    if predict_yaw and mode!='learned14_adopted':
        raise ValueError('yaw prediction requires adopted fourteen-Path predictor')
    if mode=='legacy6_demo':
        motion7,scores7,checkpoint_sha=(
            worker.predict_legacy(dem,state,robot_pose_map,goal_map)
            if worker else legacy_worker_request(
                python,checkpoint,dem,state,robot_pose_map,goal_map))
        # Legacy GUI order is Direct, Left, Right; frozen 14-bank forward
        # order is Direct, Right, Left. Reverse slots remain explicitly masked.
        reorder=(0,2,1,4,3,6,5)
        prediction=np.zeros((14,2),np.float32)
        prediction[:7]=motion7[list(reorder)]
        scores=np.full(14,-1e9,np.float32)
        scores[:7]=scores7[list(reorder)]
        mask=[True]*7+[False]*7
    else:
        if not all((results,normalization)):
            raise ValueError('adopted model requires results and normalization')
        # Stage C used zero velocity commands. Copy only the predictor input;
        # the original observation remains available to the RL Actor.
        predictor_state = (np.asarray(state, np.float32).copy()
                           if predictor_zero_velocity_command else state)
        if predictor_zero_velocity_command:
            predictor_state[6:9] = 0.0
        if predict_yaw:
            if worker is None:
                raise ValueError('yaw prediction requires persistent worker')
            prediction,yaw,checkpoint_sha=worker.predict_heading(dem,predictor_state)
            if np.asarray(yaw).shape!=(14,) or not np.isfinite(yaw).all():
                raise ValueError('predicted yaw shape/nonfinite')
        elif worker:
            prediction,checkpoint_sha=worker.predict_adopted(dem,predictor_state)
        else:
            result=json.loads(Path(results).read_text())
            norm=json.loads(Path(normalization).read_text())
            values=prepare_model_input(dem,predictor_state,PATHS,
                                       np.asarray([1]*7+[-1]*7,np.float32),norm)
            prediction,checkpoint_sha=worker_request(python,checkpoint,results,
                                                     result['source'],values,norm)
        scores=score_global_goal(prediction,robot_pose_map,goal_map,
                                 np.ones(14,bool))['scores_m']
        mask=[True]*14
    msg=CandidateEvaluation()
    msg.header.frame_id='chassis/body'; msg.header.stamp=actor.header.stamp
    msg.generation=generation; msg.predictor_mode=mode
    msg.bank_source_sha256=BANK_SHA
    msg.checkpoint_sha256=checkpoint_sha
    msg.predicted_body_xy_m=prediction.reshape(-1).tolist()
    msg.predicted_yaw_valid=bool(predict_yaw)
    msg.predicted_delta_yaw_rad=yaw.tolist()
    msg.scores_m=scores.tolist(); msg.mask=mask
    msg.selected_index=int(np.argmax(scores))
    return msg


class InferenceNode(Node):
    """Transport/mock path; real 14-model selection uses separate gated worker."""
    def __init__(self, *, mode='mock14_transport', checkpoint=None,
                 python=None, results=None, normalization=None, sim_time=False,
                 evaluation_period_ns=None, predictor_zero_velocity_command=True,
                 heading_checkpoint=None):
        super().__init__('eat_nav_phase2_inference')
        if sim_time:
            self.set_parameters([rclpy.parameter.Parameter('use_sim_time', value=True)])
        validate_mode(mode,14 if mode!='legacy6_demo' else 7,checkpoint)
        if mode!='mock14_transport' and python is None:
            raise ValueError('model worker interpreter required')
        self.mode=mode
        self.predictor_zero_velocity_command=bool(predictor_zero_velocity_command)
        self.predict_yaw=heading_checkpoint is not None
        if self.predict_yaw and mode!='learned14_adopted':
            raise ValueError('heading checkpoint requires adopted predictor')
        self.get_logger().info(
            'Motion Predictor command-zero %s; predictor input only, Actor indices 6–8'
            % ('ON' if self.predictor_zero_velocity_command else 'OFF'))
        self.evaluation_period_ns=evaluation_period_ns
        if evaluation_period_ns is not None and evaluation_period_ns <= 0:
            raise ValueError('evaluation period')
        self.checkpoint=checkpoint
        self.python=python
        self.results=results
        self.normalization=normalization
        self.worker=(None if mode=='mock14_transport' else ModelWorkerClient(
            python,mode,checkpoint,results=results,normalization=normalization,
            heading_checkpoint=heading_checkpoint))
        self.history = DemHistory()
        self.goal = None
        self.goal_generation = None
        self.last_eval_stamp = -1
        self.pending_actor = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer,self)
        self.bank_pub = self.create_publisher(CandidateBank,'/eat_nav/bank',BANK_QOS)
        self.bank_pub.publish(bank_to_message(self.get_clock().now().nanoseconds))
        self.eval_pub = self.create_publisher(CandidateEvaluation,'/eat_nav/evaluation',QOS)
        self.create_subscription(DemFrame,'/eat_nav/dem',self._dem,QOS)
        self.create_subscription(ActorState22,'/eat_nav/actor22',self._actor,QOS)
        self.create_subscription(GoalState,'/eat_nav/goal_state',self._goal,QOS)
        if sim_time:
            # TF, DEM, and Actor arrive on independent ROS topics. Retry the
            # same measured stamp while Isaac is paused for the selection.
            self.create_timer(.01, self._try_evaluate,
                              clock=Clock(clock_type=ClockType.STEADY_TIME))

    def _goal(self,msg):
        if self.goal_generation != int(msg.generation):
            self.history.clear()
            self.last_eval_stamp = -1
            self.pending_actor = None
            self.goal_generation = int(msg.generation)
        self.goal = msg if msg.active else None
        if not msg.active:
            self.history.clear()
            self.pending_actor = None

    def _dem(self,msg):
        try:
            self.history.push(msg)
            if self.pending_actor is not None:
                self._try_evaluate()
        except ValueError as error:
            self.get_logger().warn(f'DEM rejected: {error}')

    def _actor(self,msg):
        self.pending_actor = msg
        self._try_evaluate()

    def _try_evaluate(self):
        msg = self.pending_actor
        if msg is None:
            return
        if self.goal is None:
            return
        stamp = stamp_ns(msg.header.stamp)
        if self.evaluation_period_ns is not None and \
                stamp % self.evaluation_period_ns:
            return
        if stamp <= self.last_eval_stamp or \
                self.get_clock().now().nanoseconds-stamp > 300_000_000:
            return
        try:
            transform = self.tf_buffer.lookup_transform('map','chassis/body',
                                                        Time.from_msg(msg.header.stamp))
            tr = transform.transform.translation
            q = transform.transform.rotation
            yaw = math.atan2(2*(q.w*q.z+q.x*q.y),
                             1-2*(q.y*q.y+q.z*q.z))
            args=(self.history,msg,self.get_clock().now().nanoseconds,
                  int(self.goal.generation),(tr.x,tr.y,yaw),
                  (self.goal.goal_map.x,self.goal.goal_map.y))
            evaluation=(build_mock_evaluation(*args) if self.mode=='mock14_transport'
                else build_model_evaluation(*args,mode=self.mode,
                    checkpoint=self.checkpoint,python=self.python,
                    results=self.results,normalization=self.normalization,
                    worker=self.worker,
                    predictor_zero_velocity_command=self.predictor_zero_velocity_command,
                    predict_yaw=self.predict_yaw))
            self.eval_pub.publish(evaluation)
            self.last_eval_stamp = stamp
            self.pending_actor = None
        except Exception as error:
            self.get_logger().warn(f'inference deferred: {error}')

    def destroy_node(self):
        if self.worker is not None:
            self.worker.close()
        return super().destroy_node()


class SupervisorGatewayNode(Node):
    """Mock interface only. No production Path or velocity publisher exists."""
    def __init__(self, *, mode='mock14_transport', checkpoint=None, results=None,
                 sim_time=False, simulator_event_driven=False):
        super().__init__('eat_nav_phase2_supervisor')
        if sim_time:
            self.set_parameters([rclpy.parameter.Parameter('use_sim_time', value=True)])
        self.simulator_event_driven = bool(simulator_event_driven)
        validate_mode(mode,7 if mode=='legacy6_demo' else 14,checkpoint)
        if mode=='learned14_adopted':
            validate_adoption_artifacts(results,checkpoint)
        if mode=='legacy6_demo' and sha_file(checkpoint) != \
                'ff6cd2e03ee2c4832e35e2d4a935815bbd4ecb0717a8a34d445fc48e6d77bb2d':
            raise ValueError('legacy checkpoint SHA mismatch')
        self.mode=mode
        self.expected_checkpoint_sha=sha_file(checkpoint) if checkpoint else ''
        self.supervisor = Supervisor()
        self.gateway = MockFollowerGateway()
        self.follower_command = (0.,0.)
        self.follower_command_stamp_ns = None
        self.goal_pub = self.create_publisher(GoalState,'/eat_nav/goal_state',QOS)
        self.selection_pub = self.create_publisher(PathSelection,'/eat_nav/mock/selection',QOS)
        self.status_pub = self.create_publisher(NavigationStatus,'/eat_nav/status',QOS)
        self.cmd_pub = self.create_publisher(TwistStamped,'/eat_nav/mock/final_cmd',QOS)
        self.create_subscription(PoseStamped,'/goal_pose',self._goal,QOS)
        self.create_subscription(Empty,'/eat_nav/cancel',self._cancel,QOS)
        self.create_subscription(DemFrame,'/eat_nav/dem',self._dem,QOS)
        self.create_subscription(CandidateEvaluation,'/eat_nav/evaluation',self._evaluation,QOS)
        self.create_subscription(TwistStamped,'/eat_nav/mock/follower_cmd',
                                 self._follower_command,QOS)
        self.create_timer(.05,self._tick)

    def _follower_command(self,msg):
        if msg.header.frame_id != 'chassis/body':
            return
        self.follower_command=(msg.twist.linear.x,msg.twist.angular.z)
        self.follower_command_stamp_ns=stamp_ns(msg.header.stamp)
        # Phase 3 simulation asks for a gate decision at the same logical
        # stamp. Phase 2 retains timer-only behavior by default.
        if self.simulator_event_driven:
            self._tick()

    def _publish_goal(self,active,now):
        m=GoalState()
        m.header.frame_id='map'; set_stamp(m.header.stamp,now)
        m.generation=self.supervisor.generation; m.active=active
        if active:
            m.goal_map=self.current_goal.pose.position
        self.goal_pub.publish(m)

    def _goal(self,msg):
        now=self.get_clock().now().nanoseconds
        if msg.header.frame_id != 'map' or abs(now-stamp_ns(msg.header.stamp)) > 300_000_000:
            self.get_logger().warn('goal rejected: frame/stamp')
            return
        if self.gateway.active_generation is not None:
            self.gateway.invalidate(self.gateway.active_generation)
        self.supervisor.start_goal(now,timeout_ns=30_000_000_000)
        self.current_goal=msg
        self._publish_goal(True,now)
        self._tick()

    def _cancel(self,_msg):
        if self.supervisor.state not in ('IDLE','CANCELLED'):
            prior=self.supervisor.generation
            self.supervisor.cancel(prior)
            if self.gateway.active_generation is not None:
                self.gateway.invalidate(self.gateway.active_generation)
        self._publish_goal(False,self.get_clock().now().nanoseconds)
        self._tick()

    def _dem(self,msg):
        try:
            self.supervisor.observe_dem(stamp_ns(msg.header.stamp),
                                        stamp_ns(msg.sensor_stamp))
        except ValueError:
            pass

    def _evaluation(self,msg):
        now=self.get_clock().now().nanoseconds
        expected_mask=([True]*7+[False]*7 if self.mode=='legacy6_demo'
                       else [True]*14)
        if msg.predictor_mode != self.mode or \
                msg.bank_source_sha256 != BANK_SHA or \
                msg.checkpoint_sha256 != self.expected_checkpoint_sha or \
                list(msg.mask) != expected_mask or \
                msg.header.frame_id != 'chassis/body' or \
                len(msg.predicted_body_xy_m) != 28 or len(msg.scores_m) != 14 or \
                not 0 <= msg.selected_index < 14 or not msg.mask[msg.selected_index] or \
                not np.isfinite(msg.predicted_body_xy_m).all() or \
                not np.isfinite(msg.scores_m).all() or \
                int(np.argmax(np.where(msg.mask,msg.scores_m,-np.inf))) != \
                    msg.selected_index or \
                not 0 <= now-stamp_ns(msg.header.stamp) <= 300_000_000:
            return
        try:
            self.supervisor.accept_path(msg.generation,now)
            self.gateway.activate(msg.generation,now,now+500_000_000)
        except ValueError:
            return
        selected=PathSelection()
        selected.header=msg.header
        selected.generation=msg.generation
        selected.candidate_id=CANDIDATE_IDS[msg.selected_index]
        selected.direction=1 if msg.selected_index < 7 else -1
        selected.wire_path_sha256=WIRE_SHA
        selected.body_xy_m=PATHS[msg.selected_index].reshape(-1).tolist()
        set_stamp(selected.expires_at,now+500_000_000)
        self.selection_pub.publish(selected)

    def _tick(self):
        now=self.get_clock().now().nanoseconds
        decision=self.supervisor.tick(now, self.gateway.follower_healthy(now))
        if decision == 'STOP' and self.gateway.active_generation is not None:
            self.gateway.invalidate(self.gateway.active_generation)
            self._publish_goal(False,now)
        command=self.gateway.tick(now,self.follower_command,
            command_stamp_ns=self.follower_command_stamp_ns,
            supervisor_heartbeat_ns=now)
        twist=TwistStamped(); twist.header.frame_id='chassis/body'
        set_stamp(twist.header.stamp,now)
        twist.twist.linear.x=command[0]; twist.twist.angular.z=command[1]
        self.cmd_pub.publish(twist)
        status=NavigationStatus(); status.header=twist.header
        status.generation=self.supervisor.generation
        status.state=self.supervisor.state; status.reason=decision
        status.command_gate_open=self.gateway.active_generation is not None
        self.status_pub.publish(status)


def main(role, **kwargs):
    rclpy.init()
    node = InferenceNode(**kwargs) if role == 'inference' else \
        SupervisorGatewayNode(mode=kwargs.get('mode','mock14_transport'),
                              checkpoint=kwargs.get('checkpoint'),
                              results=kwargs.get('results'),
                              sim_time=kwargs.get('sim_time',False),
                              simulator_event_driven=(role=='supervisor_gui'))
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node(); rclpy.shutdown()
