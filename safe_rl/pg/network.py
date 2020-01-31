import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# import tensorflow as tf
from gym.spaces import Box, Discrete
from safe_rl.pg.utils import combined_shape, EPS


"""
Network utils
"""

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=combined_shape(None,dim))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

def placeholder_from_space(space):
    if isinstance(space, Box):
        return placeholder(space.shape)
    elif isinstance(space, Discrete):
        return tf.placeholder(dtype=tf.int32, shape=(None,))
    raise NotImplementedError('bad space {}'.format(space))

def placeholders_from_spaces(*args):
    return [placeholder_from_space(space) for space in args]

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

def get_img_features(x, config):
    activation = tf.nn.leaky_relu
    x = tf.layers.conv2d(x, 32, 3, strides=2, padding="same", activation=activation)
    # x = tf.layers.batch_normalization(x)
    x = tf.layers.conv2d(x, 64, 3, strides=2, padding="same", activation=activation)
    x = tf.layers.conv2d(x, 128, 3, strides=2, padding="same", activation=activation)
    x = tf.layers.conv2d(x, 256, 3, strides=2, padding="same", activation=activation)
    x = tf.layers.flatten(x)
    x = tf.layers.dense(x, units=512, activation=activation)
    return x

def get_output_layer(x, config):
    assert not config["output_activation"]
    return tf.layers.dense(x, units=config["output_size"])

def make_network(x, config):
    """
    :param x: placeholder for the input tensor
    :param config: network parameters
    """
    net_type = config.pop("type")
    if net_type == "mlp":
        config["hidden_sizes"].append(config.pop("output_size"))
        net = mlp(x, **config)
    elif net_type == "img_features":
        net = get_img_features(x, config)
    elif net_type == "output_layer":
        net = get_output_layer(x, config)
    else:
        assert False, f"Network type {net_type} not recognized."
    config["type"] = net_type
    return net

def get_vars(scope=''):
    return [x for x in tf.trainable_variables() if scope in x.name]

def count_vars(scope=''):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

"""
Gaussian distributions
"""

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def gaussian_kl(mu0, log_std0, mu1, log_std1):
    """Returns average kl divergence between two batches of dists"""
    var0, var1 = tf.exp(2 * log_std0), tf.exp(2 * log_std1)
    pre_sum = 0.5*(((mu1- mu0)**2 + var0)/(var1 + EPS) - 1) +  log_std1 - log_std0
    all_kls = tf.reduce_sum(pre_sum, axis=1)
    return tf.reduce_mean(all_kls)

def gaussian_entropy(log_std):
    """Returns average entropy over a batch of dists"""
    pre_sum = log_std + 0.5 * np.log(2*np.pi*np.e)
    all_ents = tf.reduce_sum(pre_sum, axis=-1)
    return tf.reduce_mean(all_ents)

"""
Categorical distributions
"""

def categorical_kl(logp0, logp1):
    """Returns average kl divergence between two batches of dists"""
    all_kls = tf.reduce_sum(tf.exp(logp1) * (logp1 - logp0), axis=1)
    return tf.reduce_mean(all_kls)

def categorical_entropy(logp):
    """Returns average entropy over a batch of dists"""
    all_ents = -tf.reduce_sum(logp * tf.exp(logp), axis=1)
    return tf.reduce_mean(all_ents)


"""
Policies
"""

def categorical_policy(x, a, config, action_space):
    act_dim = action_space.n
    config["output_size"] = act_dim
    logits = make_network(x, config)
    logp_all = tf.nn.log_softmax(logits)
    pi = tf.squeeze(tf.multinomial(logits,1), axis=1)
    logp = tf.reduce_sum(tf.one_hot(a, depth=act_dim) * logp_all, axis=1)
    logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)

    old_logp_all = placeholder(act_dim)
    d_kl = categorical_kl(logp_all, old_logp_all)
    ent = categorical_entropy(logp_all)

    pi_info = {'logp_all': logp_all}
    pi_info_phs = {'logp_all': old_logp_all}

    return pi, logp, logp_pi, pi_info, pi_info_phs, d_kl, ent


def gaussian_policy(x, a, config, action_space):
    act_dim = a.shape.as_list()[-1]
    config["output_size"] = act_dim
    mu = make_network(x, config)
    log_std = tf.get_variable(name='log_std', initializer=-0.5*np.ones(act_dim, dtype=np.float32))
    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    logp = gaussian_likelihood(a, mu, log_std)
    logp_pi = gaussian_likelihood(pi, mu, log_std)

    old_mu_ph, old_log_std_ph = placeholders(act_dim, act_dim)
    d_kl = gaussian_kl(mu, log_std, old_mu_ph, old_log_std_ph)
    ent = gaussian_entropy(log_std)

    pi_info = {'mu': mu, 'log_std': log_std}
    pi_info_phs = {'mu': old_mu_ph, 'log_std': old_log_std_ph}

    return pi, logp, logp_pi, pi_info, pi_info_phs, d_kl, ent


LOG_STD_MAX = 2
LOG_STD_MIN = -20

def mlp_squashed_gaussian_policy(x, a, hidden_sizes, activation, output_activation, action_space):
    """
    Experimental code for squashed gaussian policies, not yet tested
    """
    act_dim = a.shape.as_list()[-1]
    net = mlp(x, list(hidden_sizes), activation, activation)
    mu = tf.layers.dense(net, act_dim, activation=output_activation)
    log_std = tf.layers.dense(net, act_dim, activation=None)
    log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)

    std = tf.exp(log_std)
    u = mu + tf.random_normal(tf.shape(mu)) * std
    pi = tf.tanh(u)

    old_mu_ph, old_log_std_ph, u_ph = placeholders(act_dim, act_dim, act_dim)
    d_kl = gaussian_kl(mu, log_std, old_mu_ph, old_log_std_ph)  # kl is invariant to squashing transform

    def apply_squashing_func(log_prob, raw_action):
        # Adjustment to log prob
        act = tf.tanh(raw_action)
        log_prob -= tf.reduce_sum(2*(np.log(2) - act - tf.nn.softplus(-2*act)), axis=1)
        return log_prob

    # Base log probs
    logp = gaussian_likelihood(u_ph, mu, log_std)
    logp_pi = gaussian_likelihood(u, mu, log_std)

    # Squashed log probs
    logp = apply_squashing_func(logp, u_ph)
    logp_pi = apply_squashing_func(logp_pi, u)

    # Approximate entropy
    ent = -tf.reduce_mean(logp_pi)  # approximate! hacky!

    pi_info = {'mu': mu, 'log_std': log_std, 'raw_action': u}
    pi_info_phs = {'mu': old_mu_ph, 'log_std': old_log_std_ph, 'raw_action': u_ph}

    return pi, logp, logp_pi, pi_info, pi_info_phs, d_kl, ent



"""
Actor-Critics
"""
def actor_critic(x, a, hidden_sizes=(64,64), activation=tf.tanh,
                     output_activation=None, policy=None, action_space=None, net_type=""):

    # default policy builder depends on action space
    config = {
        "type": "mlp",
        "hidden_sizes": list(hidden_sizes),
        "activation": activation,
        "output_activation": output_activation,
    }
    if net_type == "cnn":
        config["type"] = "img_features"
        # does it matter that this isn't in one of the variable scopes?
        img_features = get_img_features(x, config)
        x = img_features
        config["type"] = "output_layer"

    if policy is None and isinstance(action_space, Box):
        policy = gaussian_policy
    elif policy is None and isinstance(action_space, Discrete):
        policy = categorical_policy

    with tf.variable_scope('pi'):
        policy_outs = policy(x, a, config, action_space)
        pi, logp, logp_pi, pi_info, pi_info_phs, d_kl, ent = policy_outs

    with tf.variable_scope('vf'):
        if config["type"] == "mlp":
            v = tf.squeeze(mlp(x, list(hidden_sizes)+[1], activation, None), axis=1)
        else:
            config["output_size"] = 1
            v = tf.squeeze(get_output_layer(x, config), axis=1)

    with tf.variable_scope('vc'):
        if config["type"] == "mlp":
            vc = tf.squeeze(mlp(x, list(hidden_sizes)+[1], activation, None), axis=1)
        else:
            config["output_size"] = 1
            vc = tf.squeeze(get_output_layer(x, config), axis=1)

    return pi, logp, logp_pi, pi_info, pi_info_phs, d_kl, ent, v, vc
