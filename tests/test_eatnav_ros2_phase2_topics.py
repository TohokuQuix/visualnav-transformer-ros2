import os
import unittest

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from eat_nav_interfaces.msg import DemFrame, ActorState22

from visualnav_transformer.eat_nav.ros2_phase2.nodes import QOS
from visualnav_transformer.eat_nav.ros2_phase2.wire import (
    dem_from_message, actor_from_message, dem_to_message, actor_to_message)


class TopicTest(unittest.TestCase):
    def test_transport_over_ros_topic(self):
        rclpy.init()
        pub=Node('eatnav_phase2_cpu_wire_pub')
        sub=Node('eatnav_phase2_cpu_wire_sub')
        got=[]
        dem_pub=pub.create_publisher(DemFrame,'/eat_nav/test/dem',QOS)
        act_pub=pub.create_publisher(ActorState22,'/eat_nav/test/actor',QOS)
        sub.create_subscription(DemFrame,'/eat_nav/test/dem',lambda m:got.append(('dem',m)),QOS)
        sub.create_subscription(ActorState22,'/eat_nav/test/actor',lambda m:got.append(('actor',m)),QOS)
        ex=SingleThreadedExecutor(); ex.add_node(pub); ex.add_node(sub)
        try:
            frame=np.zeros((2,20,20),np.float32); frame[1]=1
            frame[0,3,2]=-.37; frame[0,4,7]=0; frame[1,4,7]=0
            state=np.arange(22,dtype=np.float32)/10
            stamps={key:1_000_000_000 for key in ('imu','joint','command')}
            for _ in range(20):
                dem_pub.publish(dem_to_message(frame,1_000_000_000,1_000_000_000,1))
                act_pub.publish(actor_to_message(state,1_000_000_000,1,stamps))
                ex.spin_once(timeout_sec=.05)
                if {kind for kind,_ in got} == {'dem','actor'}:
                    break
            self.assertEqual({kind for kind,_ in got},{'dem','actor'})
            dem=next(msg for kind,msg in got if kind=='dem')
            actor=next(msg for kind,msg in got if kind=='actor')
            np.testing.assert_array_equal(dem_from_message(dem)[0],frame)
            np.testing.assert_array_equal(actor_from_message(actor)[0],state)
        finally:
            ex.shutdown(); pub.destroy_node(); sub.destroy_node(); rclpy.shutdown()


if __name__ == '__main__': unittest.main()
