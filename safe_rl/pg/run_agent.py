import numpy as np
import comet_ml
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# import tensorflow as tf
import gym
import time
import safe_rl.pg.trust_region as tro
from safe_rl.pg.agents import PPOAgent, TRPOAgent, CPOAgent
from safe_rl.pg.buffer import CPOBuffer
from safe_rl.pg.network import count_vars, \
                               get_vars, \
                               actor_critic,\
                               placeholders, \
                               placeholders_from_spaces
from safe_rl.pg.utils import values_as_sorted_list
from safe_rl.utils.logx import EpochLogger
from tqdm import trange

"""
TODO before starting all experiments!!

* Set the threshold for target # unsafe actions per episode (set it to 0)
* Do the penalties for being unsafe actually factor into the reward at all? If not,
  do we want to add that or no?

* search for other TODOs here
"""

# Multi-purpose agent runner for policy optimization algos
# (PPO, TRPO, their primal-dual equivalents, CPO)
def run_polopt_agent(env_fn,
                     agent=PPOAgent(),
                     actor_critic=actor_critic,
                     ac_kwargs=dict(),
                     seed=0,
                     render=False,
                     # Experience collection:
                     steps_per_epoch=4000,
                     epochs=50,
                     max_ep_len=1000,
                     # Discount factors:
                     gamma=0.99,
                     lam=0.97,
                     cost_gamma=0.99,
                     cost_lam=0.97,
                     # Policy learning:
                     ent_reg=0.,
                     # Cost constraints / penalties:
                     cost_lim=25,
                     penalty_init=1.,
                     penalty_lr=5e-2,
                     # KL divergence:
                     target_kl=0.01,
                     # Value learning:
                     vf_lr=1e-3,
                     vf_iters=80,
                     # Logging:
                     logger=None,
                     logger_kwargs=dict(),
                     save_freq=1,
                     use_vision=False,
                     env_name="",
                     verbose=False,
                     log_params=None,
                     ):


    #=========================================================================#
    #  Prepare logger, seed, and environment in this process                  #
    #=========================================================================#

    logger = EpochLogger(**logger_kwargs) if logger is None else logger
    logger.save_config(locals())

    tf.set_random_seed(seed)
    np.random.seed(seed)

    import ray
    ray.init()

    from symbolic_safe_rl.safety_gym_utils import VisionWrapper


    @ray.remote
    class RemoteEnv(object):
        def __init__(self, env):
            self.env = VisionWrapper(env, 128, 128)

        def reset(self):
            obs = self.env.reset()
            return obs

        def step(self, action):
            return self.env.step(action)

    n_envs = 6
    envs = [env_fn() for _ in range(n_envs)]
    envs = [RemoteEnv.remote(env) for env in envs]

    # one extra to more easily get shapes, etc.
    env = env_fn()

    range_ = lambda *args, **kwargs: trange(*args, leave=False, **kwargs)
    exp = comet_ml.Experiment(log_code=False, log_env_gpu=False, log_env_cpu=False)
    exp.add_tag("crl")

    if exp:
        if "Point" in env_name:
           robot_type = "Point"
        elif "Car" in env_name:
            robot_type = "Car"
        elif "Doggo" in env_name:
            robot_type = "Doggo"
        else:
            assert False
        task = env_name.replace("-v0", "").replace("Safexp-", "").replace(robot_type, "")
        task, difficulty = task[:-1], task[-1]

        exp.log_parameters({
            "robot": robot_type,
            "task": task,
            "difficulty": difficulty,
            "model": "cnn0" if use_vision else "mlp",
            "use_vision": use_vision,
            "steps_per_epoch": steps_per_epoch,
            "vf_iters": vf_iters,
        })
        if log_params:
            exp.log_parameters(log_params)

    if use_vision:
        # TODO - cmd line arg for shape
        img_width = img_height = 128
        env = VisionWrapper(env, img_width, img_height)

    agent.set_logger(logger)

    #=========================================================================#
    #  Create computation graph for actor and critic (not training routine)   #
    #=========================================================================#

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    if use_vision:
        ac_kwargs["net_type"] = "cnn"

    # Inputs to computation graph from environment spaces
    x_ph, a_ph = placeholders_from_spaces(env.observation_space, env.action_space)

    # Inputs to computation graph for batch data
    adv_ph, cadv_ph, ret_ph, cret_ph, logp_old_ph = placeholders(*(None for _ in range(5)))

    # Inputs to computation graph for special purposes
    surr_cost_rescale_ph = tf.placeholder(tf.float32, shape=())
    cur_cost_ph = tf.placeholder(tf.float32, shape=())

    # Outputs from actor critic
    ac_outs = actor_critic(x_ph, a_ph, **ac_kwargs)
    pi, logp, logp_pi, pi_info, pi_info_phs, d_kl, ent, v, vc = ac_outs

    # Organize placeholders for zipping with data from buffer on updates
    buf_phs = [x_ph, a_ph, adv_ph, cadv_ph, ret_ph, cret_ph, logp_old_ph]
    buf_phs += values_as_sorted_list(pi_info_phs)

    # Organize symbols we have to compute at each step of acting in env
    get_action_ops = dict(pi=pi,
                          v=v,
                          logp_pi=logp_pi,
                          pi_info=pi_info)

    # If agent is reward penalized, it doesn't use a separate value function
    # for costs and we don't need to include it in get_action_ops; otherwise we do.
    if not(agent.reward_penalized):
        get_action_ops['vc'] = vc

    # Count variables
    var_counts = tuple(count_vars(scope) for scope in ['pi', 'vf', 'vc'])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d, \t vc: %d\n'%var_counts)

    # Make a sample estimate for entropy to use as sanity check
    approx_ent = tf.reduce_mean(-logp)


    #=========================================================================#
    #  Create replay buffer                                                   #
    #=========================================================================#

    # Obs/act shapes
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    # Experience buffer
    local_steps_per_epoch = int(steps_per_epoch / n_envs)
    pi_info_shapes = {k: v.shape.as_list()[1:] for k,v in pi_info_phs.items()}
    bufs = [CPOBuffer(local_steps_per_epoch,
                    obs_shape,
                    act_shape,
                    pi_info_shapes,
                    gamma,
                    lam,
                    cost_gamma,
                    cost_lam) for _ in range(n_envs)]


    #=========================================================================#
    #  Create computation graph for penalty learning, if applicable           #
    #=========================================================================#

    if agent.use_penalty:
        with tf.variable_scope('penalty'):
            # param_init = np.log(penalty_init)
            param_init = np.log(max(np.exp(penalty_init)-1, 1e-8))
            penalty_param = tf.get_variable('penalty_param',
                                          initializer=float(param_init),
                                          trainable=agent.learn_penalty,
                                          dtype=tf.float32)
        # penalty = tf.exp(penalty_param)
        penalty = tf.nn.softplus(penalty_param)

    if agent.learn_penalty:
        if agent.penalty_param_loss:
            penalty_loss = -penalty_param * (cur_cost_ph - cost_lim)
        else:
            penalty_loss = -penalty * (cur_cost_ph - cost_lim)
        # train_penalty = MpiAdamOptimizer(learning_rate=penalty_lr).minimize(penalty_loss)
        train_penalty = tf.train.AdamOptimizer(learning_rate=penalty_lr).minimize(penalty_loss)


    #=========================================================================#
    #  Create computation graph for policy learning                           #
    #=========================================================================#

    # Likelihood ratio
    ratio = tf.exp(logp - logp_old_ph)

    # Surrogate advantage / clipped surrogate advantage
    if agent.clipped_adv:
        min_adv = tf.where(adv_ph>0,
                           (1+agent.clip_ratio)*adv_ph,
                           (1-agent.clip_ratio)*adv_ph
                           )
        surr_adv = tf.reduce_mean(tf.minimum(ratio * adv_ph, min_adv))
    else:
        surr_adv = tf.reduce_mean(ratio * adv_ph)

    # Surrogate cost
    surr_cost = tf.reduce_mean(ratio * cadv_ph)

    # Create policy objective function, including entropy regularization
    pi_objective = surr_adv + ent_reg * ent

    # Possibly include surr_cost in pi_objective
    if agent.objective_penalized:
        pi_objective -= penalty * surr_cost
        pi_objective /= (1 + penalty)

    # Loss function for pi is negative of pi_objective
    pi_loss = -pi_objective

    # Optimizer-specific symbols
    if agent.trust_region:

        # Symbols needed for CG solver for any trust region method
        pi_params = get_vars('pi')
        flat_g = tro.flat_grad(pi_loss, pi_params)
        v_ph, hvp = tro.hessian_vector_product(d_kl, pi_params)
        if agent.damping_coeff > 0:
            hvp += agent.damping_coeff * v_ph

        # Symbols needed for CG solver for CPO only
        flat_b = tro.flat_grad(surr_cost, pi_params)

        # Symbols for getting and setting params
        get_pi_params = tro.flat_concat(pi_params)
        set_pi_params = tro.assign_params_from_flat(v_ph, pi_params)

        training_package = dict(flat_g=flat_g,
                                flat_b=flat_b,
                                v_ph=v_ph,
                                hvp=hvp,
                                get_pi_params=get_pi_params,
                                set_pi_params=set_pi_params)

    elif agent.first_order:

        # Optimizer for first-order policy optimization
        # train_pi = MpiAdamOptimizer(learning_rate=agent.pi_lr).minimize(pi_loss)
        train_pi = tf.train.AdamOptimizer(learning_rate=agent.pi_lr).minimize(pi_loss)

        # Prepare training package for agent
        training_package = dict(train_pi=train_pi)

    else:
        raise NotImplementedError

    # Provide training package to agent
    training_package.update(dict(pi_loss=pi_loss,
                                 surr_cost=surr_cost,
                                 d_kl=d_kl,
                                 target_kl=target_kl,
                                 cost_lim=cost_lim))
    agent.prepare_update(training_package)

    #=========================================================================#
    #  Create computation graph for value learning                            #
    #=========================================================================#

    # Value losses
    v_loss = tf.reduce_mean((ret_ph - v)**2)
    vc_loss = tf.reduce_mean((cret_ph - vc)**2)

    # If agent uses penalty directly in reward function, don't train a separate
    # value function for predicting cost returns. (Only use one vf for r - p*c.)
    if agent.reward_penalized:
        total_value_loss = v_loss
    else:
        total_value_loss = v_loss + vc_loss

    # Optimizer for value learning
    # train_vf = MpiAdamOptimizer(learning_rate=vf_lr).minimize(total_value_loss)
    train_vf = tf.train.AdamOptimizer(learning_rate=vf_lr).minimize(total_value_loss)


    #=========================================================================#
    #  Create session, sync across procs, and set up saver                    #
    #=========================================================================#

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # Sync params across processes
    # sess.run(sync_all_params())

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph}, outputs={'pi': pi, 'v': v, 'vc': vc})


    #=========================================================================#
    #  Provide session to agent                                               #
    #=========================================================================#
    agent.prepare_session(sess)


    #=========================================================================#
    #  Create function for running update (called at end of each epoch)       #
    #=========================================================================#

    def update():
        # TODO!!! - need correct epcost...
        cur_cost = logger.get_stats('EpCost')[0]
        c = cur_cost - cost_lim
        if c > 0 and agent.cares_about_cost:
            if verbose:
                logger.log('Warning! Safety constraint is already violated.', 'red')

        #=====================================================================#
        #  Prepare feed dict                                                  #
        #=====================================================================#

        inputs = {}
        buf_inputs = [buf.get() for buf in bufs]
        for i, ph in enumerate(buf_phs):
            inputs[ph] = np.concatenate([buf_input[i] for buf_input in buf_inputs])

        # inputs = {k:v for k,v in zip(buf_phs, buf.get())}
        inputs[surr_cost_rescale_ph] = logger.get_stats('EpLen')[0]
        inputs[cur_cost_ph] = cur_cost

        #=====================================================================#
        #  Make some measurements before updating                             #
        #=====================================================================#

        measures = dict(LossPi=pi_loss,
                        SurrCost=surr_cost,
                        LossV=v_loss,
                        Entropy=ent)
        if not(agent.reward_penalized):
            measures['LossVC'] = vc_loss
        if agent.use_penalty:
            measures['Penalty'] = penalty

        pre_update_measures = sess.run(measures, feed_dict=inputs)
        logger.store(**pre_update_measures)

        #=====================================================================#
        #  Update penalty if learning penalty                                 #
        #=====================================================================#
        if agent.learn_penalty:
            sess.run(train_penalty, feed_dict={cur_cost_ph: cur_cost})

        #=====================================================================#
        #  Update policy                                                      #
        #=====================================================================#
        agent.update_pi(inputs)

        #=====================================================================#
        #  Update value function                                              #
        #=====================================================================#
        for _ in range(vf_iters):
            sess.run(train_vf, feed_dict=inputs)

        #=====================================================================#
        #  Make some measurements after updating                              #
        #=====================================================================#

        del measures['Entropy']
        measures['KL'] = d_kl

        post_update_measures = sess.run(measures, feed_dict=inputs)
        deltas = dict()
        for k in post_update_measures:
            if k in pre_update_measures:
                deltas['Delta'+k] = post_update_measures[k] - pre_update_measures[k]
        logger.store(KL=post_update_measures['KL'], **deltas)




    #=========================================================================#
    #  Run main environment interaction loop                                  #
    #=========================================================================#

    start_time = time.time()
    rs = np.zeros(n_envs)
    ds = [False] * n_envs
    cs = np.zeros(n_envs)
    ep_rets = np.zeros(n_envs)
    ep_costs = np.zeros(n_envs)
    ep_lens = np.zeros(n_envs)
    vc_t0 = np.zeros(n_envs)
    os = np.stack(ray.get([env.reset.remote() for env in envs]))
    cur_penalty = 0
    cum_cost = 0

    for epoch in range_(epochs):

        if agent.use_penalty:
            cur_penalty = sess.run(penalty)

        for t in range_(local_steps_per_epoch):

            # Possibly render
            # if render and rank == 0 and t < 1000:
            #     env.render()

            # Get outputs from policy
            get_action_outs = sess.run(get_action_ops,
                                       feed_dict={x_ph: os})
            a = get_action_outs['pi']
            v_t = get_action_outs['v']
            vc_t = get_action_outs.get('vc', vc_t0)  # Agent may not use cost value func
            logp_t = get_action_outs['logp_pi']
            pi_info_t = get_action_outs['pi_info']
            pi_info_t = [{"mu": pi_info_t["mu"][i:i+1], "log_std": pi_info_t["log_std"]} for i in range(n_envs)]

            # Step in environment
            o2s, rs, ds, infos = zip(*ray.get([env.step.remote(a_i) for env, a_i in zip(envs, a)]))
            o2s = np.stack(o2s)
            rs = np.array(rs)
            # o2, r, d, info = env.step(a)

            # Include penalty on cost
            cs = np.array([info.get('cost', 0) for info in infos])

            # Track cumulative cost over training
            cum_cost += cs.sum()

            # save and log
            if agent.reward_penalized:
                r_totals = rs - cur_penalty * cs
                r_totals = r_totals / (1 + cur_penalty)
                for i, buf in enumerate(bufs):
                    buf.store(os[i], a[i], r_totals[i], v_t[i], 0, 0, logp_t[i], pi_info_t[i])
            else:
                for i, buf in enumerate(bufs):
                    buf.store(os[i], a[i], rs[i], v_t[i], cs[i], vc_t[i], logp_t[i], pi_info_t[i])
            # TODO - what values to use here??
            logger.store(VVals=v_t[0], CostVVals=vc_t[0])

            os = o2s
            ep_rets += rs
            ep_costs += cs
            ep_lens += 1

            for i, buf in enumerate(bufs):
                ep_len = ep_lens[i]
                d = ds[i]
                terminal = d or (ep_len == max_ep_len)
                if terminal or (t==local_steps_per_epoch-1):

                    # If trajectory didn't reach terminal state, bootstrap value target(s)
                    if d and not(ep_len == max_ep_len):
                        # Note: we do not count env time out as true terminal state
                        last_val, last_cval = 0, 0
                    else:
                        feed_dict={x_ph: os[i:i+1]}
                        if agent.reward_penalized:
                            last_val = sess.run(v, feed_dict=feed_dict)
                            last_cval = 0
                        else:
                            last_val, last_cval = sess.run([v, vc], feed_dict=feed_dict)
                    buf.finish_path(last_val, last_cval)

                    # Only save EpRet / EpLen if trajectory finished
                    if terminal:
                        ep_ret = ep_rets[i]
                        ep_cost = ep_costs[i]
                        logger.store(EpRet=ep_ret, EpLen=ep_len, EpCost=ep_cost)
                        if exp:
                            exp.log_metrics({
                                "return": ep_ret,
                                "episode_length": ep_len,
                                "cost": ep_cost,
                            }, step=epoch * steps_per_epoch + t)
                    else:
                        if verbose:
                            print('Warning: trajectory cut off by epoch at %d steps.'%ep_len)

                    # Reset environment
                    os[i] = ray.get(envs[i].reset.remote())
                    rs[i] = 0
                    # ds[i] = False
                    cs[i] = 0
                    ep_rets[i] = 0
                    ep_lens[i] = 0
                    ep_costs[i] = 0

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)

        #=====================================================================#
        #  Run RL update                                                      #
        #=====================================================================#
        update()

        #=====================================================================#
        #  Cumulative cost calculations                                       #
        #=====================================================================#
        # TODO
        # cumulative_cost = mpi_sum(cum_cost)
        # cost_rate = cumulative_cost / ((epoch+1)*steps_per_epoch)

        #=====================================================================#
        #  Log performance and stats                                          #
        #=====================================================================#

        logger.log_tabular('Epoch', epoch)

        # Performance stats
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpCost', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        # logger.log_tabular('CumulativeCost', cumulative_cost)
        # logger.log_tabular('CostRate', cost_rate)

        # Value function values
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('CostVVals', with_min_and_max=True)

        # Pi loss and change
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)

        # Surr cost and change
        logger.log_tabular('SurrCost', average_only=True)
        logger.log_tabular('DeltaSurrCost', average_only=True)

        # V loss and change
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)

        # Vc loss and change, if applicable (reward_penalized agents don't use vc)
        if not(agent.reward_penalized):
            logger.log_tabular('LossVC', average_only=True)
            logger.log_tabular('DeltaLossVC', average_only=True)

        if agent.use_penalty or agent.save_penalty:
            logger.log_tabular('Penalty', average_only=True)
            logger.log_tabular('DeltaPenalty', average_only=True)
        else:
            logger.log_tabular('Penalty', 0)
            logger.log_tabular('DeltaPenalty', 0)

        # Anything from the agent?
        agent.log()

        # Policy stats
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)

        # Time and steps elapsed
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('Time', time.time()-start_time)

        # Show results!
        if verbose:
            logger.dump_tabular()
        else:
            logger.log_current_row.clear()
            logger.first_row = False

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default='ppo')
    parser.add_argument('--env', type=str, default='Safexp-PointGoal1-v0')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--cost_gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--len', type=int, default=1000)
    parser.add_argument('--cost_lim', type=float, default=10)
    parser.add_argument('--exp_name', type=str, default='runagent')
    parser.add_argument('--kl', type=float, default=0.01)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--reward_penalized', action='store_true')
    parser.add_argument('--objective_penalized', action='store_true')
    parser.add_argument('--learn_penalty', action='store_true')
    parser.add_argument('--penalty_param_loss', action='store_true')
    parser.add_argument('--entreg', type=float, default=0.)
    args = parser.parse_args()

    try:
        import safety_gym
    except:
        print('Make sure to install Safety Gym to use constrained RL environments.')

    mpi_fork(args.cpu)  # run parallel code with mpi

    # Prepare logger
    from safe_rl.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    # Prepare agent
    agent_kwargs = dict(reward_penalized=args.reward_penalized,
                        objective_penalized=args.objective_penalized,
                        learn_penalty=args.learn_penalty,
                        penalty_param_loss=args.penalty_param_loss)
    if args.agent=='ppo':
        agent = PPOAgent(**agent_kwargs)
    elif args.agent=='trpo':
        agent = TRPOAgent(**agent_kwargs)
    elif args.agent=='cpo':
        agent = CPOAgent(**agent_kwargs)

    run_polopt_agent(lambda : gym.make(args.env),
                     agent=agent,
                     actor_critic=actor_critic,
                     ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
                     seed=args.seed,
                     render=args.render,
                     # Experience collection:
                     steps_per_epoch=args.steps,
                     epochs=args.epochs,
                     max_ep_len=args.len,
                     # Discount factors:
                     gamma=args.gamma,
                     cost_gamma=args.cost_gamma,
                     # Policy learning:
                     ent_reg=args.entreg,
                     # KL Divergence:
                     target_kl=args.kl,
                     cost_lim=args.cost_lim,
                     # Logging:
                     logger_kwargs=logger_kwargs,
                     save_freq=1,
                     env_name=args.env,
                     )
