#!/usr/bin/env python
import gym
import comet_ml
import safety_gym
import safe_rl
from safe_rl.utils.run_utils import setup_logger_kwargs


def main(robot, task, algo, seed, exp_name, cpu, use_vision):

    # Verify experiment
    robot_list = ['point', 'car', 'doggo']
    task_list = ['goal1', 'goal2', 'button1', 'button2', 'push1', 'push2']
    algo_list = ['ppo', 'ppo_lagrangian', 'trpo', 'trpo_lagrangian', 'cpo']

    algo = algo.lower()
    task = task.capitalize()
    robot = robot.capitalize()
    assert algo in algo_list, "Invalid algo"
    assert task.lower() in task_list, "Invalid task"
    assert robot.lower() in robot_list, "Invalid robot"

    k = 5
    pi_iters = int(80 / k)
    vf_iters = int(80 / k)

    # Hyperparameters
    if robot=='Doggo':
        num_steps = 1e8
        steps_per_epoch = 60000
    else:
        num_steps = 1e7
        steps_per_epoch = int(30000 / k)
    epochs = int(num_steps / steps_per_epoch)
    save_freq = 50
    target_kl = 0.01
    cost_lim = 25

    # Prepare Logger
    exp_name = exp_name or (algo + '_' + robot.lower() + task.lower())
    logger_kwargs = setup_logger_kwargs(exp_name, seed)

    # Algo and Env
    algo = getattr(safe_rl, algo)
    env_name = 'Safexp-'+robot+task+'-v0'

    log_params = {"pi_iters": pi_iters}
    algo(pi_iters=pi_iters,
         env_fn=lambda: gym.make(env_name),
         ac_kwargs=dict(
             hidden_sizes=(256, 256),
            ),
         epochs=epochs,
         steps_per_epoch=steps_per_epoch,
         save_freq=save_freq,
         target_kl=target_kl,
         cost_lim=cost_lim,
         seed=seed,
         logger_kwargs=logger_kwargs,
         env_name=env_name,
         use_vision=use_vision,
         vf_iters=vf_iters,
         log_params=log_params,
         )



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', type=str, default='Point')
    parser.add_argument('--task', type=str, default='Goal1')
    parser.add_argument('--algo', type=str, default='ppo')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument("--vision", action="store_true")
    args = parser.parse_args()
    main(args.robot, args.task, args.algo, args.seed, args.exp_name, args.cpu, args.vision)
