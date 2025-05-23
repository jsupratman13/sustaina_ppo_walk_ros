#!/usr/bin/env pipenv-shebang

import os
import pickle

import numpy as np
import rospy
import torch
from rospkg.rospack import RosPack
from rsl_rl.runners import OnPolicyRunner
from sensor_msgs.msg import Imu
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


def inv_quat(quat):
    if isinstance(quat, torch.Tensor):
        scaling = torch.tensor([1, -1, -1, -1], device=quat.device)
        return quat * scaling
    elif isinstance(quat, np.ndarray):
        scaling = np.array([1, -1, -1, -1], dtype=quat.dtype)
        return quat * scaling
    raise Exception(f"the input must be either torch.Tensor or np.ndarray. got: {type(quat)=}")


def quat_to_xyz(quat):
    if isinstance(quat, torch.Tensor):
        # Extract quaternion components
        qw, qx, qy, qz = quat.unbind(-1)
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = torch.atan2(sinr_cosp, cosr_cosp)
        # Pitch (y-axis rotation)
        sinp = 2 * (qw * qy - qz * qx)
        pitch = torch.where(
            torch.abs(sinp) >= 1,
            torch.sign(sinp) * torch.tensor(torch.pi / 2),
            torch.asin(sinp),
        )
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        return torch.stack([roll, pitch, yaw], dim=-1) * 180.0 / torch.tensor(np.pi)
    elif isinstance(quat, np.ndarray):
        return Rotation.from_quat(wxyz_to_xyzw(quat)).as_euler("xyz", degrees=True)
    raise Exception(f"the input must be either torch.Tensor or np.ndarray. got: {type(quat)=}")


def transform_by_quat(v, quat):
    if isinstance(v, torch.Tensor) and isinstance(quat, torch.Tensor):
        qvec = quat[..., 1:]
        t = qvec.cross(v, dim=-1) * 2
        return v + quat[..., :1] * t + qvec.cross(t, dim=-1)
    elif isinstance(v, np.ndarray) and isinstance(quat, np.ndarray):
        return transform_by_R(v, quat_to_R(quat))
    raise Exception(f"the input must be either torch.Tensor or np.ndarray. got: {type(quat)=}")


def transform_quat_by_quat(v, u):
    if isinstance(v, torch.Tensor) and isinstance(u, torch.Tensor):
        assert v.shape == u.shape, f"{v.shape} != {u.shape}"
        w1, x1, y1, z1 = u[..., 0], u[..., 1], u[..., 2], u[..., 3]
        w2, x2, y2, z2 = v[..., 0], v[..., 1], v[..., 2], v[..., 3]
        ww = (z1 + x1) * (x2 + y2)
        yy = (w1 - y1) * (w2 + z2)
        zz = (w1 + y1) * (w2 - z2)
        xx = ww + yy + zz
        qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
        w = qq - ww + (z1 - y1) * (y2 - z2)
        x = qq - xx + (x1 + w1) * (x2 + w2)
        y = qq - yy + (w1 - x1) * (y2 + z2)
        z = qq - zz + (z1 + y1) * (w2 - x2)
        quat = torch.stack([w, x, y, z], dim=-1)
        return quat
    elif isinstance(v, np.ndarray) and isinstance(u, np.ndarray):
        assert v.shape == u.shape, f"{v.shape} != {u.shape}"
        w1, x1, y1, z1 = u[..., 0], u[..., 1], u[..., 2], u[..., 3]
        w2, x2, y2, z2 = v[..., 0], v[..., 1], v[..., 2], v[..., 3]
        # This method transforms quat_v by quat_u
        # This is equivalent to quatmul(quat_u, quat_v) or R_u @ R_v
        ww = (z1 + x1) * (x2 + y2)
        yy = (w1 - y1) * (w2 + z2)
        zz = (w1 + y1) * (w2 - z2)
        xx = ww + yy + zz
        qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
        w = qq - ww + (z1 - y1) * (y2 - z2)
        x = qq - xx + (x1 + w1) * (x2 + w2)
        y = qq - yy + (w1 - x1) * (y2 + z2)
        z = qq - zz + (z1 + y1) * (w2 - x2)
        quat = np.stack([w, x, y, z], axis=-1)
        return quat
    raise Exception(f"the input must be either torch.Tensor or np.ndarray. got: {type(quat)=}")


class SustainaEnv(object):
    def __init__(self,
                 env_cfg: dict,
                 obs_cfg: dict,
                 command_cfg: dict,
                 device: str = 'cuda'):
        self.device = torch.device(device)

        self.num_envs = 1
        self.num_obs = obs_cfg['num_obs']
        self.num_privileged_obs = None
        self.num_actions = env_cfg['num_actions']
        self.num_commands = command_cfg['num_commands']

        self.obs_scales = obs_cfg['obs_scales']
        self.clip_action = env_cfg['clip_actions']
        self.action_scale = env_cfg['action_scale']
        self.default_dof_pos = torch.tensor(
            [env_cfg['default_joint_angles'][name] for name in env_cfg['dof_names']],
            device=self.device,
            dtype=torch.float,
        )
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=torch.long)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float)
        self.dof_names = env_cfg['dof_names']

        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=torch.float)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=torch.float)
        self.last_actions = torch.zeros_like(self.actions)
        self.base_init_quat = torch.tensor(env_cfg['base_init_quat'], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float)
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device,
                                           dtype=torch.float).repeat(self.num_envs, 1)
        self.dof_pos = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=torch.float)
        self.dof_vel = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=torch.float)
        self.commands_scale = torch.tensor(
            [self.obs_scales['lin_vel'], self.obs_scales['lin_vel'], self.obs_scales['ang_vel']],
            device=self.device, dtype=torch.float)

        # ROS
        self.cmd_pub = rospy.Publisher('joint_group_position_controller/command', Float64MultiArray, queue_size=1)
        # self.imu_sub = rospy.Subscriber('imu/data', Imu, self.imu_callback)
        self.joint_state_sub = rospy.Subscriber('joint_states', JointState, self.joint_state_callback)
        # self.cmd_sub = rospy.Subscriber('command', Float64MultiArray, self.cmd_callback)

        self.latest_imu_msg = None
        self.latest_joint_state_msg = None

    def imu_callback(self, msg):
        self.latest_imu_msg = msg

        # Update orientation: ROS gives (x,y,z,w)
        quat = torch.tensor(
            [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
            device=self.device, dtype=torch.float
        )
        self.base_quat[0, :] = quat

        # Update angular velocity:
        ang_vel = torch.tensor(
            [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
            device=self.device, dtype=torch.float
        )
        self.base_ang_vel[0, :] = ang_vel

    def joint_state_callback(self, msg):
        self.latest_joint_state_msg = msg

        # Assume msg.name is a list of joint names and that env_cfg['dof_names'] gives the desired order.
        pos = []
        vel = []
        for joint in self.dof_names:
            if joint in msg.name:
                idx = msg.name.index(joint)
                pos.append(msg.position[idx])
                vel.append(msg.velocity[idx])
            else:
                # If a joint is missing, fall back to default.
                default_val = self.default_dof_pos[self.dof_names.index(joint)].item()
                pos.append(default_val)
                vel.append(0.0)
        self.dof_pos[0, :] = torch.tensor(pos, device=self.device, dtype=torch.float)
        self.dof_vel[0, :] = torch.tensor(vel, device=self.device, dtype=torch.float)

    def cmd_callback(self, msg):
        cmd = torch.tensor(msg.data, device=self.device, dtype=torch.float)
        if cmd.ndim == 1:
            cmd = cmd.unsqueeze(0)
        self.commands.copy_(cmd)

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return None

    def reset(self):
        self.latest_imu_msg = None
        self.latest_joint_state_msg = None
        return self.obs_buf, None

    def step(self, actions):
        actions = torch.clip(actions, - self.clip_action, self.clip_action)
        exec_actions = self.last_actions
        target_dof_pos = exec_actions * self.action_scale + self.default_dof_pos
        cmd_msg = Float64MultiArray()
        cmd_msg.data = target_dof_pos.detach().cpu().numpy().flatten().tolist()
        self.cmd_pub.publish(cmd_msg)

        # update buffers
        # self.base_quat[:]  # TODO: get quat
        # self.base_euler = quat_to_xyz(
        #     transform_quat_by_quat(torch.ones_like(self.base_quat * self.inv_base_init_quat), self.base_quat))
        # inv_base_quat = inv_quat(self.base_quat)
        # self.base_ang_vel  # TODO: get angular velocity
        # self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        # self.dof_pos[:]  # TODO: get joint pos
        # self.dof_vel[:]  # TODO: get joint vel

        msg = rospy.wait_for_message('imu/data', Imu)
        self.imu_callback(msg)
        # if self.latest_imu_msg is not None:
        #     quat = torch.tensor(
        #         [self.latest_imu_msg.orientation.x, self.latest_imu_msg.orientation.y,
        #          self.latest_imu_msg.orientation.z, self.latest_imu_msg.orientation.w],
        #         device=self.device, dtype=torch.float
        #     )
        #     self.base_quat[0, :] = quat
        #     ang_vel = torch.tensor(
        #         [self.latest_imu_msg.angular_velocity.x, self.latest_imu_msg.angular_velocity.y,
        #          self.latest_imu_msg.angular_velocity.z],
        #         device=self.device, dtype=torch.float
        #     )
        #     self.base_ang_vel[0, :] = ang_vel

        msg = rospy.wait_for_message('joint_states', JointState)
        self.joint_state_callback(msg)
        if self.latest_joint_state_msg is not None:
            pos = []
            vel = []
            for joint in self.dof_names:
                if joint in self.latest_joint_state_msg.name:
                    idx = self.latest_joint_state_msg.name.index(joint)
                    pos.append(self.latest_joint_state_msg.position[idx])
                    vel.append(self.latest_joint_state_msg.velocity[idx])
                else:
                    # If a joint is missing, fall back to default.
                    default_val = self.default_dof_pos[self.dof_names.index(joint)].item()
                    pos.append(default_val)
                    vel.append(0.0)
            self.dof_pos[0, :] = torch.tensor(pos, device=self.device, dtype=torch.float)
            self.dof_vel[0, :] = torch.tensor(vel, device=self.device, dtype=torch.float)

        relative_quat = transform_quat_by_quat(
            torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        self.base_euler = quat_to_xyz(relative_quat)
        inv_base_quat = inv_quat(self.base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)

        # compute observations
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales['ang_vel'],
                self.projected_gravity,
                self.commands * self.commands_scale,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales['dof_pos'],
                self.dof_vel * self.obs_scales['dof_vel'],
                self.actions,
            ],
            axis=-1,
        )

        self.last_actions[:] = actions[:]

        return self.obs_buf, None, self.rew_buf, self.reset_buf, {}


class EvalWalk(object):
    def __init__(self):
        self.__weight = rospy.get_param('~weight_path')
        self.__config = rospy.get_param('~train_config')
        env_cfg, obs_cfg, _, command_cfg, train_cfg, _ = pickle.load(open(self.__config, 'rb'))
        log_dir = RosPack().get_path('sustaina_walk_ros') + '/config/dummy'

        env = SustainaEnv(env_cfg, obs_cfg, command_cfg)

        runner = OnPolicyRunner(env, train_cfg, log_dir, device='cuda:0')
        runner.load(self.__weight)
        policy = runner.get_inference_policy(device='cuda:0')
        obs, _ = env.reset()
        rate = rospy.Rate(50)
        while torch.no_grad():
            while not rospy.is_shutdown():
                actions = policy(obs)
                obs, _, _, _, _ = env.step(actions)
                rate.sleep()


if __name__ == '__main__':
    rospy.init_node('sustaina_walk_ros')
    evaluate = EvalWalk()

    rospy.spin()
