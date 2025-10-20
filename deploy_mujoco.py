import time
import argparse
import os
import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml
from collections import deque


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='g1 deploy mujoco')
    parser.add_argument('--policy', type=str, default="checkpoint/policy.pt",
                       help='Direct path to policy file (overrides config file)')
    parser.add_argument('--cmd', type=float, nargs=3, default=None,
                       help='Override desired command [vx vy yaw]')
    parser.add_argument('--log-vel', action='store_true',
                       help='Print base linear velocity (m/s) and base height (m) at control updates')
    
    args = parser.parse_args()
    
    with open("configs/g1_29dof_walk.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
        if args.policy is not None:
            policy_path = args.policy
        else:
            policy_path = config["policy_path"]

        if os.path.exists(policy_path):
            print(f"Using policy from command line: {policy_path}")
        else:
            raise FileNotFoundError(f"Policy file not found: {policy_path}")

        xml_path = config["xml_path"]

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)
        policy_joints = config["policy_joints"]

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        
        if args.cmd is not None:
            cmd = np.array(args.cmd, dtype=np.float32)
        else:
            cmd = np.array(config["cmd_init"], dtype=np.float32)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    policy_to_xml = []
    for i in range(1, m.njnt):
        jname = mujoco.mj_id2name(m, 3, i)
        idx = policy_joints.index(jname)
        policy_to_xml.append(idx)

    xml_to_policy = []
    for i in range(len(policy_to_xml)):
        idx = policy_to_xml.index(i)
        xml_to_policy.append(idx)

    default_angles = default_angles[policy_to_xml]
    target_dof_pos = default_angles.copy()

    frame_stack = deque(maxlen=5)
    for _ in range(5):
        frame_stack.append(obs.copy())
        mujoco.mj_step(m, d) 


    # load policy
    policy = torch.jit.load(policy_path)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # Apply control signal here.

                # create observation
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]

                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * ang_vel_scale

                count = counter * simulation_dt

                obs[:3] = omega
                obs[3:6] = gravity_orientation
                obs[6:9] = cmd * cmd_scale
                obs[9 : 9 + num_actions] = qj[xml_to_policy]
                obs[9 + num_actions : 9 + 2 * num_actions] = dqj[xml_to_policy]
                obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action[xml_to_policy]

                frame_stack.append(obs.copy())
                stacked_obs = np.concatenate(frame_stack, axis=0)
                
                obs_omega = np.asarray(stacked_obs).reshape(5, 96)[:, 0:3].reshape(-1)
                obs_gravity_orientation = np.asarray(stacked_obs).reshape(5, 96)[:, 3:6].reshape(-1)
                obs_cmd = np.asarray(stacked_obs).reshape(5, 96)[:, 6:9].reshape(-1)
                obs_pos = np.asarray(stacked_obs).reshape(5, 96)[:, 9:9 + num_actions].reshape(-1)
                obs_vel = np.asarray(stacked_obs).reshape(5, 96)[:, 9 + num_actions : 9 + 2 * num_actions].reshape(-1)
                obs_action = np.asarray(stacked_obs).reshape(5, 96)[:, 9 + 2 * num_actions : 9 + 3 * num_actions].reshape(-1)
                big_group_major = np.concatenate([
                    obs_omega,
                    obs_gravity_orientation,
                    obs_cmd,
                    obs_pos,
                    obs_vel,
                    obs_action,
                ], axis=0)
                obs_tensor = torch.from_numpy(big_group_major).unsqueeze(0)

                # policy inference
                action = policy(obs_tensor).detach().numpy().squeeze()
                action = action[policy_to_xml]
                # transform action to target_dof_pos
                target_dof_pos = action * action_scale + default_angles

                if args.log_vel:
                    base_lin_vel = d.qvel[0:3]
                    base_height = d.qpos[2]
                    print(f"vel: [{base_lin_vel[0]:+.3f}, {base_lin_vel[1]:+.3f}, {base_lin_vel[2]:+.3f}] m/s, height: {base_height:.3f} m, cmd: [{cmd[0]:.2f}, {cmd[1]:.2f}, {cmd[2]:.2f}]")

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
