import time
from constrained_solver import solve
import rospy
from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import JointPositionSensorMessage, ShouldTerminateSensorMessage
from franka_interface_msgs.msg import SensorDataGroup

from frankapy.utils import min_jerk
if __name__ == '__main__':
    fa = FrankaArm()
    rospy.loginfo('Initializing Sensor Publisher')
    pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)
    joint_state = fa.get_joints().astype('float32')
    rate = rospy.Rate(10)

    ee_translation_goal = fa.get_pose().translation.astype('float32')
    ee_quaternion_goal = fa.get_pose().quaternion.astype('float32')
    print(ee_translation_goal)
    print(ee_quaternion_goal)
    ee_translation_goal[1] +=0.05
    
    print(joint_state)
    joints_traj = solve(
        joint_state,
        ee_translation_goal,
        ee_quaternion_goal,
    )
    print(len(joints_traj))
    input()
    fa.goto_joints(joint_state, duration=5, dynamic=True, buffer_time=10, ignore_virtual_walls=True)
    init_time = rospy.Time.now().to_time()
    for i in range(2, len(joints_traj)):
        traj_gen_proto_msg = JointPositionSensorMessage(
            id=i, timestamp=rospy.Time.now().to_time() - init_time, 
            joints=joints_traj[i]
        )
        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                traj_gen_proto_msg, SensorDataMessageType.JOINT_POSITION)
        )
        
        rospy.loginfo('Publishing: ID {}'.format(traj_gen_proto_msg.id))
        pub.publish(ros_msg)
        rate.sleep()