#!/usr/bin/env pipenv-shebang

import pickle

import numpy as np
import rospy
import torch
from rospkg.rospack import RosPack
from rsl_rl.runners import OnPolicyRunner
from sensor_msgs.msg import Imu
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

# Joint order (ROS and RL are the same)
joint_order = [
    'left_shoulder_pitch_joint',
    'left_shoulder_roll_joint',
    'left_elbow_pitch_joint',
    'left_waist_yaw_joint',
    'left_waist_roll_joint',
    'left_waist_pitch_joint',
    'left_knee_pitch_joint',
    'left_ankle_pitch_joint',
    'left_ankle_roll_joint',
    'right_shoulder_pitch_joint',
    'right_shoulder_roll_joint',
    'right_elbow_pitch_joint',
    'right_waist_yaw_joint',
    'right_waist_roll_joint',
    'right_waist_pitch_joint',
    'right_knee_pitch_joint',
    'right_ankle_pitch_joint',
    'right_ankle_roll_joint',
]


class WalkPolicyRosBridge:
    def __init__(self, exp_name, ckpt):
        # Load configs and policy
        log_dir = RosPack().get_path('sustaina_ppo_walk_ros') + '/logs'
        # log_dir = f"logs/{exp_name}"
        # env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg, domain_rand_cfg = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg, domain_rand_cfg = pickle.load(
            open(rospy.get_param('~train_config'), "rb"))

        class DummyEnv:
            def __init__(self):
                self.num_privileged_obs = None
                self.num_obs = obs_cfg['num_obs']
                self.num_actions = env_cfg['num_actions']
                self.num_envs = 1
                self.reset = lambda **kwargs: (None, None)

        runner = OnPolicyRunner(DummyEnv(), train_cfg, log_dir, device="cuda:0")
        resume_path = f"{log_dir}/model_{ckpt}.pt"
        # runner.load(resume_path)
        runner.load(rospy.get_param('~weight_path'))
        self.policy = runner.get_inference_policy(device="cuda:0")

        # Prepare obs and state arrays
        self.num_obs = obs_cfg['num_obs']
        self.joint_pos = np.zeros(18)
        self.joint_vel = np.zeros(18)
        self.last_actions = np.zeros(18)
        self.imu_ang_vel = np.zeros(3)
        self.ready = False

        # Scales and defaults
        self.obs_scales = obs_cfg['obs_scales']
        self.default_dof_pos = np.array(
            [env_cfg['default_joint_angles'][jn] for jn in joint_order], dtype=np.float32
        )
        self.clip_actions = env_cfg['clip_actions']
        self.action_scale = env_cfg['action_scale']

        # ROS
        rospy.Subscriber("joint_states", JointState, self.joint_state_callback)
        rospy.Subscriber("imu/data", Imu, self.imu_callback)
        self.joint_cmd_pub = rospy.Publisher(
            "joint_group_position_controller/command",
            Float64MultiArray,
            queue_size=1)

    def joint_state_callback(self, msg):
        name_to_idx = {name: i for i, name in enumerate(msg.name)}
        for i, jn in enumerate(joint_order):
            if jn in msg.name:
                idx = msg.name.index(jn)
                self.joint_pos[i] = msg.position[idx]
                self.joint_vel[i] = 0.0  # msg.velocity[idx] if len(msg.velocity) > idx else 0.0

        self.ready = True

    def imu_callback(self, msg):
        # Only use angular velocity for this example; expand if needed
        self.imu_ang_vel[0] = msg.angular_velocity.x
        self.imu_ang_vel[1] = msg.angular_velocity.y
        self.imu_ang_vel[2] = msg.angular_velocity.z

    def build_observation(self):
        obs = []
        obs += list(self.imu_ang_vel * self.obs_scales['ang_vel'])  # 3
        # obs += [0.0, 0.0, -1.0]   # Projected gravity (replace with real value if needed)
        obs += [0.0, 0.0, 0.0]    # Command (set to zero, or change if you want walking)
        obs += list((self.joint_pos - self.default_dof_pos) * self.obs_scales['dof_pos'])  # 18
        # obs += list(self.joint_vel * self.obs_scales['dof_vel'] * 0.0)  # 18
        obs += list(self.last_actions)  # 18
        # Pad if less than num_obs
        obs = np.array(obs, dtype=np.float32)
        if obs.shape[0] < self.num_obs:
            obs = np.pad(obs, (0, self.num_obs - obs.shape[0]))
        return obs.reshape(1, -1)  # batch of one

    def step(self):
        if not self.ready:
            return
        obs = self.build_observation()
        obs_torch = torch.from_numpy(obs).to("cuda:0")
        with torch.no_grad():
            actions = self.policy(obs_torch)  # .cpu().numpy().flatten()
            actions = torch.clip(actions, -self.clip_actions, self.clip_actions)
            actions = actions.cpu().numpy().flatten()
        self.last_actions = actions
        msg = Float64MultiArray()
        msg.data = (actions * self.action_scale + self.default_dof_pos).tolist()
        self.joint_cmd_pub.publish(msg)


def main():
    rospy.init_node('walk_policy_ros_bridge')
    exp_name = rospy.get_param('~exp_name', 'sustaina-walking')
    ckpt = rospy.get_param('~ckpt', 200)
    bridge = WalkPolicyRosBridge(exp_name, ckpt)
    rate = rospy.Rate(20)  # 50 Hz
    while not rospy.is_shutdown():
        bridge.step()
        rate.sleep()


if __name__ == "__main__":
    main()
