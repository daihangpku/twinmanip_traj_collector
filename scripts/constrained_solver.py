import torch
import json
a = torch.zeros(4, device="cuda:0")

# Third Party
import numpy as np
import torch

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.types.state import JointState
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    PoseCostMetric,
)
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.util_file import get_robot_path
import argparse

parser = argparse.ArgumentParser()


parser.add_argument("--robot", type=str, default="franka.yml", help="robot configuration to load")

args = parser.parse_args()


def solve(joint_state, ee_translation_goal, ee_orientation_goal):
    #joint_state = [-0.95555313,  1.11025978 , 0.66254836, -1.85974693,  0.83744874,  1.87575201, -1.77852278]
    js_names = ["panda_joint1","panda_joint2","panda_joint3","panda_joint4", "panda_joint5",
      "panda_joint6","panda_joint7"]
 # Set EE teleop goals, use cube for simple non-vr init:
    # ee_translation_goal =[0.54131516, 0.08129586, 0.05194307] #[ 0.52246403 -0.07636242  0.27630054]
    # ee_orientation_goal = [-0.48969682 , 0.51585968 , 0.48066197 , 0.51288387]
    collision_checker_type = CollisionCheckerType.BLOX

    tensor_args = TensorDeviceType()

    config_file = load_yaml(join_path(get_robot_path(), "franka.yml"))
    print(join_path(get_robot_path(), "franka.yml"))
    urdf_file = config_file["robot_cfg"]["kinematics"][
        "urdf_path"
    ]  
    base_link = config_file["robot_cfg"]["kinematics"]["base_link"]
    ee_link = config_file["robot_cfg"]["kinematics"]["ee_link"]

    robot_cfg = RobotConfig.from_basic(urdf_file, base_link, ee_link, tensor_args)
    kin_model = CudaRobotModel(robot_cfg.kinematics)
    # compute forward kinematics:
    q = torch.tensor(joint_state).to(device="cuda:0")
    out = kin_model.get_state(q)
    print(out)
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        None,
        tensor_args,
        interpolation_dt=0.02,
        ee_link_name="ee_link",
    )
    motion_gen = MotionGen(motion_gen_config)
    print("warming up..")
    motion_gen.warmup(warmup_js_trajopt=False)

    # world_model = motion_gen.world_collision
    tensor_args = TensorDeviceType()
    cmd_plan = None

    plan_config = MotionGenPlanConfig(
        enable_graph=False,
        enable_graph_attempt=4,
        max_attempts=2,
        enable_finetune_trajopt=True,
        time_dilation_factor=0.5,
    )

    #print("Constrained: Holding tool linear-y")
    pose_cost_metric = PoseCostMetric(
        hold_partial_pose=True,
        hold_vec_weight=motion_gen.tensor_args.to_device([1, 1, 1, 1, 1, 0]),
    )

    plan_config.pose_cost_metric = pose_cost_metric

    # motion generation:
    
    cu_js = JointState(
        position=tensor_args.to_device(joint_state),
        velocity=tensor_args.to_device(joint_state) * 0.0,
        acceleration=tensor_args.to_device(joint_state) * 0.0,
        jerk=tensor_args.to_device(joint_state) * 0.0,
        joint_names=js_names,
    )
    cu_js = cu_js.get_ordered_joint_state(motion_gen.kinematics.joint_names)

   

    # compute curobo solution:
    ik_goal = Pose(
        position=tensor_args.to_device(ee_translation_goal),
        quaternion=tensor_args.to_device(ee_orientation_goal),
    )
    result = motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, plan_config)
    # ik_result = ik_solver.solve_single(ik_goal, cu_js.position.view(1,-1), cu_js.position.view(1,1,-1))

    succ = result.success.item()  # ik_result.success.item()
    if succ:
        print("success")
        cmd_plan = result.get_interpolated_plan()
        cmd_plan = motion_gen.get_full_js(cmd_plan)
        # get only joint names that are in both:
        q = torch.tensor(cmd_plan.position[-1]).to(device="cuda:0")
        out = kin_model.get_state(q)
        print(out)
        cmd_plan_dict = {
            "position": cmd_plan.position.cpu().numpy().tolist(),
            "velocity": cmd_plan.velocity.cpu().numpy().tolist(),
            "acceleration": cmd_plan.acceleration.cpu().numpy().tolist(),
            "jerk": cmd_plan.jerk.cpu().numpy().tolist(),
            "joint_names": cmd_plan.joint_names,
        }
        with open("cmd_plan.json", "w") as json_file:
            json.dump(cmd_plan_dict, json_file, indent=4)
        print("Saved cmd_plan to cmd_plan.json")
        #print(cmd_plan)
        #cmd_plan = cmd_plan.get_ordered_joint_state(common_js_names)
        return cmd_plan.position.cpu().numpy().tolist()

    else:
        print("failed")
        pass
        
    print("finished program")