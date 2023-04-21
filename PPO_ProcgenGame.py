import torch

from agents import TYPE
from agents.PPOProcgenAgent import PPOProcgenSNDAgent, PPOProcgenAgent, PPOProcgenRNDAgent, PPOProcgenSPAgent, PPOProcgenICMAgent
from experiment.ppo_nenv_experiment import ExperimentNEnvPPO
from utils.MultiEnvWrapper import MultiEnvParallel
from utils.ProcgenWrapper import WrapperProcgenExploration


def encode_state(state):
    return torch.tensor(state, dtype=torch.float32)


def test(config, path, env_name):
    env = WrapperProcgenExploration(env_name)
    input_shape = env.observation_space.shape
    action_dim = env.action_space.n

    experiment = ExperimentNEnvPPO(env_name, env, config)
    experiment.add_preprocess(encode_state)

    agent = PPOProcgenSNDAgent(input_shape, action_dim, config, TYPE.discrete)
    agent.load(path)
    experiment.test(agent)

    env.close()


def run_baseline(config, trial, env_name):
    print('Creating {0:d} environments'.format(config.n_env))
    env = MultiEnvParallel([WrapperProcgenExploration(env_name) for _ in range(config.n_env)], config.n_env, config.num_threads)

    input_shape = env.observation_space.shape
    action_dim = env.action_space.n

    print('Start training')
    experiment = ExperimentNEnvPPO(env_name, env, config)

    experiment.add_preprocess(encode_state)
    agent = PPOProcgenAgent(input_shape, action_dim, config, TYPE.discrete)
    experiment.run_baseline(agent, trial)

    env.close()


def run_rnd_model(config, trial, env_name):
    print('Creating {0:d} environments'.format(config.n_env))
    env = MultiEnvParallel([WrapperProcgenExploration(env_name) for _ in range(config.n_env)], config.n_env, config.num_threads)

    input_shape = env.observation_space.shape
    action_dim = env.action_space.n

    print('Start training')
    experiment = ExperimentNEnvPPO(env_name, env, config)

    experiment.add_preprocess(encode_state)
    agent = PPOProcgenRNDAgent(input_shape, action_dim, config, TYPE.discrete)
    experiment.run_rnd_model(agent, trial)

    env.close()


def run_snd_model(config, trial, env_name):
    print('Creating {0:d} environments'.format(config.n_env))
    env = MultiEnvParallel([WrapperProcgenExploration(env_name) for _ in range(config.n_env)], config.n_env, config.num_threads)

    input_shape = env.observation_space.shape
    action_dim = env.action_space.n

    print('Start training')
    experiment = ExperimentNEnvPPO(env_name, env, config)

    experiment.add_preprocess(encode_state)
    agent = PPOProcgenSNDAgent(input_shape, action_dim, config, TYPE.discrete)
    experiment.run_snd_model(agent, trial)

    env.close()


def run_sp_model(config, trial, env_name):
    print('Creating {0:d} environments'.format(config.n_env))
    env = MultiEnvParallel([WrapperProcgenExploration(env_name) for _ in range(config.n_env)], config.n_env, config.num_threads)

    input_shape = env.observation_space.shape
    action_dim = env.action_space.n

    print('Start training')
    experiment = ExperimentNEnvPPO(env_name, env, config)

    experiment.add_preprocess(encode_state)
    agent = PPOProcgenSPAgent(input_shape, action_dim, config, TYPE.discrete)
    experiment.run_sp_model(agent, trial)

    env.close()


def run_icm_model(config, trial, env_name):
    print('Creating {0:d} environments'.format(config.n_env))
    env = MultiEnvParallel([WrapperProcgenExploration(env_name) for _ in range(config.n_env)], config.n_env, config.num_threads)

    input_shape = env.observation_space.shape
    action_dim = env.action_space.n

    print('Start training')
    experiment = ExperimentNEnvPPO(env_name, env, config)

    experiment.add_preprocess(encode_state)
    agent = PPOProcgenICMAgent(input_shape, action_dim, config, TYPE.discrete)
    experiment.run_icm_model(agent, trial)

    env.close()
