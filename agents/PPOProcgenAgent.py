from agents.PPOAgent import PPOAgent
from algorithms.PPO import PPO
from algorithms.ReplayBuffer import GenericTrajectoryBuffer
from modules.PPO_ProcgenModules import PPOProcgenNetwork, PPOProcgenNetworkRND, PPOProcgenNetworkSP, PPOProcgenNetworkICM, PPOProcgenNetworkSND
from motivation.ForwardModelMotivation import ForwardModelMotivation
from motivation.RNDMotivation import RNDMotivation
from motivation.SNDMotivation import SNDMotivationFactory


class PPOProcgenAgent(PPOAgent):
    def __init__(self, input_shape, action_dim, config, action_type):
        super().__init__(input_shape, action_dim, action_type, config)
        self.network = PPOProcgenNetwork(input_shape, action_dim, config, head=action_type).to(config.device)
        self.algorithm = PPO(self.network, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.batch_size, config.trajectory_size,
                             config.beta, config.gamma, ppo_epochs=config.ppo_epochs, n_env=config.n_env,
                             device=config.device, motivation=False)


class PPOProcgenRNDAgent(PPOAgent):
    def __init__(self, input_shape, action_dim, config, action_type):
        super().__init__(input_shape, action_dim, action_type, config)
        self.network = PPOProcgenNetworkRND(input_shape, action_dim, config, head=action_type).to(config.device)
        self.motivation = RNDMotivation(self.network.rnd_model, config.motivation_lr, config.motivation_eta, config.device)
        self.algorithm = PPO(self.network, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.batch_size, config.trajectory_size,
                             config.beta, config.gamma, ext_adv_scale=2, int_adv_scale=1, ppo_epochs=config.ppo_epochs, n_env=config.n_env,
                             device=config.device, motivation=True)

    def train(self, state0, value, action0, probs0, state1, reward, mask):
        self.memory.add(state=state0.cpu(), value=value.cpu(), action=action0.cpu(), prob=probs0.cpu(), reward=reward.cpu(), mask=mask.cpu())
        indices = self.memory.indices()
        self.algorithm.train(self.memory, indices)
        self.motivation.train(self.memory, indices)
        if indices is not None:
            self.memory.clear()


class PPOProcgenSPAgent(PPOAgent):
    def __init__(self, input_shape, action_dim, config, action_type):
        super().__init__(input_shape, action_dim, action_type, config)
        self.network = PPOProcgenNetworkSP(input_shape, action_dim, config, head=action_type).to(config.device)
        self.motivation = ForwardModelMotivation(self.network.forward_model, config.motivation_lr, config.motivation_eta, config.forward_model_variant, config.device)
        self.algorithm = PPO(self.network, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.batch_size, config.trajectory_size,
                             config.beta, config.gamma, ext_adv_scale=2, int_adv_scale=1, ppo_epochs=config.ppo_epochs, n_env=config.n_env,
                             device=config.device, motivation=True)

    def train(self, state0, value, action0, probs0, state1, reward, mask):
        self.memory.add(state=state0.cpu(), value=value.cpu(), action=action0.cpu(), prob=probs0.cpu(), next_state=state1.cpu(), reward=reward.cpu(), mask=mask.cpu())
        indices = self.memory.indices()
        self.algorithm.train(self.memory, indices)
        self.motivation.train(self.memory, indices)
        if indices is not None:
            self.memory.clear()


class PPOProcgenICMAgent(PPOAgent):
    def __init__(self, input_shape, action_dim, config, action_type):
        super().__init__(input_shape, action_dim, action_type, config)
        self.network = PPOProcgenNetworkICM(input_shape, action_dim, config, head=action_type).to(config.device)
        self.motivation = ForwardModelMotivation(self.network.forward_model, config.motivation_lr, config.motivation_eta, config.forward_model_variant, config.device)
        self.algorithm = PPO(self.network, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.batch_size, config.trajectory_size,
                             config.beta, config.gamma, ext_adv_scale=2, int_adv_scale=1, ppo_epochs=config.ppo_epochs, n_env=config.n_env,
                             device=config.device, motivation=True)

    def train(self, state0, value, action0, probs0, state1, reward, mask):
        self.memory.add(state=state0.cpu(), value=value.cpu(), action=action0.cpu(), prob=probs0.cpu(), next_state=state1.cpu(), reward=reward.cpu(), mask=mask.cpu())
        indices = self.memory.indices()

        self.algorithm.train(self.memory, indices)
        self.motivation.train(self.memory, indices)
        if indices is not None:
            self.memory.clear()

    def get_action(self, state):
        value, action, probs = self.network(state)
        features = self.network.forward_model.encoder(state)

        return features.detach(), value.detach(), action, probs.detach()


class PPOProcgenSNDAgent(PPOAgent):
    def __init__(self, input_shape, action_dim, config, action_type):
        super().__init__(input_shape, action_dim, action_type, config)
        self.network = PPOProcgenNetworkSND(input_shape, action_dim, config, head=action_type).to(config.device)
        self.motivation_memory = GenericTrajectoryBuffer(config.trajectory_size, config.batch_size, config.n_env)
        self.motivation = SNDMotivationFactory.get_motivation(config.type, self.network.cnd_model, config.motivation_lr, config.motivation_eta, config.device)
        self.algorithm = PPO(self.network, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.batch_size, config.trajectory_size,
                             config.beta, config.gamma, ext_adv_scale=2, int_adv_scale=1, ppo_epochs=config.ppo_epochs, n_env=config.n_env,
                             device=config.device, motivation=True)

    def train(self, state0, value, action0, probs0, state1, reward, mask):
        self.memory.add(state=state0.cpu(), value=value.cpu(), action=action0.cpu(), prob=probs0.cpu(), reward=reward.cpu(), mask=mask.cpu())
        self.motivation_memory.add(state=state0.cpu(), next_state=state1.cpu(), action=action0.cpu())

        indices = self.memory.indices()
        motivation_indices = self.motivation_memory.indices()

        if indices is not None:
            self.algorithm.train(self.memory, indices)
            self.memory.clear()

        if motivation_indices is not None:
            self.motivation.train(self.motivation_memory, motivation_indices)
            self.motivation_memory.clear()

    def get_action(self, state):
        value, action, probs = self.network(state)
        features = self.network.cnd_model.target_model(self.network.cnd_model.preprocess(state))

        return features.detach(), value.detach(), action, probs.detach()
