import time
from constrained_solver import solve
import rospy
from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import JointPositionSensorMessage, ShouldTerminateSensorMessage
from franka_interface_msgs.msg import SensorDataGroup
from scipy.spatial.transform import Rotation as R
from frankapy.utils import min_jerk
import argparse
import pyrealsense2 as rs
import cv2
import numpy as np
import os
import threading
from tqdm import tqdm
POSE = [
    [-0.35471786,  0.67136656, -0.12932039, -1.96341863,  0.79877985,  2.14604293,  1.27579102],#left

]
def init_realsense(fps=30):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, fps)  # 配置彩色流
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, fps)  # 配置彩色流
    profile = pipeline.start(config)
    return pipeline, profile

def control_thread(fa, joint_state, joints_traj, init_time, args):
    joints_cmd=[]
    pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)
    time.sleep(1.0)
    fa.goto_joints(joint_state, duration=5, dynamic=True, buffer_time=10, ignore_virtual_walls=True, 
                   k_gains= [500.0, 500.0, 500.0, 500.0, 700.0, 200.0, 80.0],
                                      d_gains=[100.0, 100.0, 80.0, 80.0, 30.0, 20.0, 15.0],
                   )
    rate = rospy.Rate(10)
    tss = []
    joints_traj.append(joints_traj[-1])
    joints_traj.append(joints_traj[-1])
    joints_traj.append(joints_traj[-1])
    # joints_traj.append(joints_traj[-1])
    # joints_traj.append(joints_traj[-1])
    # joints_traj.append(joints_traj[-1])
    for i in range(0, len(joints_traj)):
        pose = fa.get_pose().translation
        if pose[1]>-0.05:
            flag=1
        else:
            flag=0
        timestamp = rospy.Time.now().to_time() - init_time
        tss.append(timestamp)
        #import ipdb; ipdb.set_trace()
        cmd = joints_traj[len(joints_traj)//2] if flag else joints_traj[-1]
        traj_gen_proto_msg = JointPositionSensorMessage(
            id=i, timestamp=rospy.Time.now().to_time() - init_time, 
            joints=cmd,
        )
        joints_cmd.append(cmd)
        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                traj_gen_proto_msg, SensorDataMessageType.JOINT_POSITION)
        )
        
        pub.publish(ros_msg)

        rospy.loginfo(f"Published control command ID {traj_gen_proto_msg.id}")
        rate.sleep()
    time.sleep(1)
    cmd_timestamps = []
    for i, (ts, cmd) in enumerate(zip(tss, joints_cmd)):
        
        cmd_timestamps.append({
            "id": i,
            "ros_timestamp": ts,
            "cmd": cmd
        })
    save_timestamps(args, "control.json", cmd_timestamps)
        
def save_timestamps(args, name, ts):
    import json
    with open(f"{args.save_dir}/{name}", "w") as f:
        json.dump(ts, f, indent=4)
    #print("Timestamps saved to timestamps.json")
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pushing with Franka')
    parser.add_argument('--robot', type=str, default="franka.yml", help="robot configuration to load")
    parser.add_argument('--debug', type=int, default=0, help='debug level')
    parser.add_argument('--cmd_num', type=int, default=20, help='')
    parser.add_argument('--save_dir', type=str, default='wood/traj14', help='index of the saved data')
    args = parser.parse_args()
    dir_name = args.save_dir
    os.makedirs(dir_name, exist_ok=True)
    depth_dir = os.path.join(dir_name, "depth")
    color_dir = os.path.join(dir_name, "rgb")
    vis_dir = os.path.join(dir_name, "vis")
            
    os.makedirs(depth_dir,exist_ok=True)
    os.makedirs(color_dir,exist_ok=True)
    os.makedirs(vis_dir,exist_ok=True)

    fa = FrankaArm()
    fa.goto_joints(POSE[0], ignore_virtual_walls=True)
    rospy.loginfo('Initializing Sensor Publisher')
    #rate = rospy.Rate(30)
    #pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)
    fa.close_gripper()
    joint_state = fa.get_joints().astype('float32')
    ee_translation = fa.get_pose().translation.astype('float32')
    ee_quaternion = fa.get_pose().quaternion.astype('float32')
    rotation = R.from_quat([ee_quaternion[3], ee_quaternion[0], ee_quaternion[1], ee_quaternion[2]])
    rotation_matrix = rotation.as_matrix()
    #print("rotation_matrix: ", rotation_matrix)
    print("")
    print("ee_translation: ", ee_translation)
    print("ee_quaternion: ", ee_quaternion)
    #ee_translation[1] +=0.1
    z_axis = rotation_matrix[:, 2]
    z_proj = -z_axis
    z_proj[2] = 0.0
    z_proj = z_proj / np.linalg.norm(z_proj)
    print(z_proj)
    #ee_translation_goal += z_proj * 0.2
    print("joint_state: ", joint_state)
    trans_goals = []
    rot_goals = []
    # trans_goals.append(ee_translation)
    # rot_goals.append(ee_quaternion)
    X = 0.05
    for i in range(7):
        trans_goals.append(ee_translation+z_proj* X*i)
        rot_goals.append(ee_quaternion)
    new_joint_state = joint_state
    joints_traj = []
    ee_translation_traj = []
    for i in range(0, len(trans_goals)-1):
        
        j_traj, ee_traj, new_joint_state = solve(
            new_joint_state,
            ee_translation=trans_goals[i],
            ee_orientation=rot_goals[i], 
            ee_translation_goal=trans_goals[i+1],
            ee_orientation_goal=rot_goals[i+1],
            args=args
        )
        joints_traj+= j_traj
        ee_translation_traj += ee_traj


    joints_traj = joints_traj[::20]
    print(len(joints_traj))
    input()
    init_time = rospy.Time.now().to_time()
    #control_thread(fa, joint_state, joints_traj, init_time, args)
    

    timestamps=[]
    pipeline, profile = init_realsense()
    # Get the stream profiles for depth and color
    depth_profile = profile.get_stream(rs.stream.depth)
    color_profile = profile.get_stream(rs.stream.color)

    # Get the intrinsics for depth and color streams
    depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
    color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
    cam_K = np.array([
        [color_intrinsics.fx, 0, color_intrinsics.ppx],
        [0, color_intrinsics.fy, color_intrinsics.ppy],
        [0, 0, 1]
    ])
    with open(os.path.join(dir_name, "cam_K.txt"), "w") as f:
        for row in cam_K:
            f.write(" ".join(f"{x:.10f}" for x in row) + "\n")
    align_to = rs.stream.color
    align = rs.align(align_to)
    save_index = 0
    #start controlling
    control_thread_obj = threading.Thread(target=control_thread, args=( fa, joint_state, joints_traj, init_time, args))
    control_thread_obj.start()
    #start recording
    depth_images = []
    color_images = []
    for i in range(150):
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        depth_frame = None
        color_frame = None
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            print(123)
            continue
        camera_timestamp = frames.get_timestamp() / 1000.0  # 转换为秒
        ros_timestamp = rospy.Time.now().to_time() - init_time
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        # np.savez_compressed(os.path.join(depth_dir, f"{save_index:05d}.npz"), depth=depth_image.astype(np.float32)/1000.0)#
        # cv2.imwrite(os.path.join(color_dir, f"{save_index:05d}.png"), color_image)

        depth_images.append((depth_image.astype(np.float32) / 1000.0).copy())  # 深度图以米为单位
        color_images.append(color_image.copy())
        timestamps.append({
                "id": i,
                "ros_timestamp": ros_timestamp,
                "camera_timestamp": camera_timestamp
            })
        save_index += 1
    control_thread_obj.join()
    for idx, (depth_image, color_image) in enumerate(tqdm(zip(depth_images, color_images), desc='saving...')):
        np.savez_compressed(os.path.join(depth_dir, f"{idx:05d}.npz"), depth=depth_image)
        cv2.imwrite(os.path.join(color_dir, f"{idx:05d}.png"), color_image)
    #fa.goto_joints(POSE[0])
    save_timestamps(args, "frame.json", timestamps)