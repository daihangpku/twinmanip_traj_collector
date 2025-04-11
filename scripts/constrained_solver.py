import torch

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

import argparse

from curobo.util.usd_helper import UsdHelper

parser = argparse.ArgumentParser()


parser.add_argument("--robot", type=str, default="franka.yml", help="robot configuration to load")

args = parser.parse_args()


if __name__ == "__main__":

    n_obstacle_cuboids = 10
    n_obstacle_mesh = 10

    # target_orient = [0,0,0.707,0.707]
    target_orient = [0.5, -0.5, 0.5, 0.5]


    collision_checker_type = CollisionCheckerType.BLOX

    tensor_args = TensorDeviceType()

    robot_cfg = load_yaml(join_path(get_robot_configs_path(), args.robot))["robot_cfg"]

    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        None,
        tensor_args,
        collision_checker_type=CollisionCheckerType.MESH,
        collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
        interpolation_dt=0.02,
        ee_link_name="right_gripper",
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

    print("Constrained: Holding tool linear-y")
    pose_cost_metric = PoseCostMetric(
        hold_partial_pose=True,
        hold_vec_weight=motion_gen.tensor_args.to_device([0, 0, 0, 0, 1, 0]),
    )

    plan_config.pose_cost_metric = pose_cost_metric

    # motion generation:
    sim_js = robot.get_joints_state()
    sim_js_names = robot.dof_names
    cu_js = JointState(
        position=tensor_args.to_device(sim_js.positions),
        velocity=tensor_args.to_device(sim_js.velocities) * 0.0,
        acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
        jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
        joint_names=sim_js_names,
    )
    cu_js = cu_js.get_ordered_joint_state(motion_gen.kinematics.joint_names)

    # Set EE teleop goals, use cube for simple non-vr init:
    ee_translation_goal = cube_position
    ee_orientation_teleop_goal = cube_orientation

    # compute curobo solution:
    ik_goal = Pose(
        position=tensor_args.to_device(ee_translation_goal),
        quaternion=tensor_args.to_device(ee_orientation_teleop_goal),
    )
    result = motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, plan_config)
    # ik_result = ik_solver.solve_single(ik_goal, cu_js.position.view(1,-1), cu_js.position.view(1,1,-1))

    succ = result.success.item()  # ik_result.success.item()
    if succ:
        cmd_plan = result.get_interpolated_plan()
        cmd_plan = motion_gen.get_full_js(cmd_plan)
        # get only joint names that are in both:

        cmd_plan = cmd_plan.get_ordered_joint_state(common_js_names)

    else:
        pass
        
    print("finished program")