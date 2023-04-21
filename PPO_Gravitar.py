import PPO_HardAtariGame

env_name = 'GravitarNoFrameskip-v4'


def test(config, path):
    PPO_HardAtariGame.test(config, path, env_name)


def run_baseline(config, trial):
    PPO_HardAtariGame.run_baseline(config, trial, env_name)


def run_rnd_model(config, trial):
    PPO_HardAtariGame.run_rnd_model(config, trial, env_name)


def run_qrnd_model(config, trial):
    PPO_HardAtariGame.run_qrnd_model(config, trial, env_name)


def run_forward_model(config, trial):
    PPO_HardAtariGame.run_forward_model(config, trial, env_name)


def run_sp_model(config, trial):
    PPO_HardAtariGame.run_sp_model(config, trial, env_name)


def run_icm_model(config, trial):
    PPO_HardAtariGame.run_icm_model(config, trial, env_name)


def run_snd_model(config, trial):
    PPO_HardAtariGame.run_snd_model(config, trial, env_name)
