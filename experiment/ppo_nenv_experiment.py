import numpy
import torch
from gym.wrappers.monitoring.video_recorder import VideoRecorder

from analytic.ResultCollector import ResultCollector
from utils.RunningAverage import RunningAverageWindow, StepCounter
from utils.TimeEstimator import PPOTimeEstimator


class ExperimentNEnvPPO:
    def __init__(self, env_name, env, config):
        self._env_name = env_name
        self._env = env
        self._config = config
        self._preprocess = None

        print('Total steps: {0:.2f}M'.format(self._config.steps))
        print('Total batch size: {0:d}'.format(self._config.batch_size))
        print('Total trajectory size: {0:d}'.format(self._config.trajectory_size))

    def add_preprocess(self, preprocess):
        self._preprocess = preprocess

    def process_state(self, state):
        if self._preprocess is None:
            processed_state = torch.tensor(state, dtype=torch.float32).to(self._config.device)
        else:
            processed_state = self._preprocess(state).to(self._config.device)

        return processed_state

    def test(self, agent):
        config = self._config

        for i in range(1):
            video_path = 'ppo_{0}_{1}_{2:d}.mp4'.format(config.name, config.model, i)
            video_recorder = VideoRecorder(self._env, video_path)
            state0 = torch.tensor(self._env.reset(), dtype=torch.float32).unsqueeze(0).to(config.device)
            done = False

            while not done:
                video_recorder.capture_frame()
                # features0 = agent.get_features(state0)
                _, _, action0, probs0 = agent.get_action(state0)
                # actor_state, value, action0, probs0, head_value, head_action, head_probs, all_values, all_action, all_probs = agent.get_action(state0)
                # action0 = probs0.argmax(dim=1)
                next_state, reward, done, info = self._env.step(agent.convert_action(action0.cpu())[0])
                state0 = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(config.device)
            video_recorder.close()

    def run_baseline(self, agent, trial):
        config = self._config
        n_env = config.n_env
        trial = trial + config.shift
        step_counter = StepCounter(int(config.steps * 1e6))
        reward_avg = RunningAverageWindow(100)
        time_estimator = PPOTimeEstimator(step_counter.limit)

        analytic = ResultCollector()
        analytic.init(n_env, re=(1,), ext_value=(1,))

        s = numpy.zeros((n_env,) + self._env.observation_space.shape, dtype=numpy.float32)
        for i in range(n_env):
            s[i] = self._env.reset(i)

        state0 = self.process_state(s)

        while step_counter.running():
            with torch.no_grad():
                value, action0, probs0 = agent.get_action(state0)

            next_state, reward, done, info = self._env.step(agent.convert_action(action0.cpu()))

            reward = torch.tensor(reward, dtype=torch.float32)
            if info is not None:
                if 'normalised_score' in info:
                    analytic.add(normalised_score=(1,))
                    score = torch.tensor(info['normalised_score']).unsqueeze(-1)
                    analytic.update(normalised_score=score)
                if 'raw_score' in info:
                    analytic.add(score=(1,))
                    score = torch.tensor(info['raw_score']).unsqueeze(-1)
                    analytic.update(score=score)

            analytic.update(re=reward,
                            ext_value=value[:, 0].unsqueeze(-1).cpu())

            env_indices = numpy.nonzero(numpy.squeeze(done, axis=1))[0]
            stats = analytic.reset(env_indices)
            step_counter.update(n_env)

            for i, index in enumerate(env_indices):
                reward_avg.update(stats['re'].sum[i])

                print('Run {0:d} step {1:d}/{2:d} training [ext. reward {3:f} steps {4:d}  mean reward {5:f} score {6:f})]'.format(
                    trial, step_counter.steps, step_counter.limit, stats['re'].sum[i], int(stats['re'].step[i]), reward_avg.value().item(), stats['score'].sum[i]))
                print(time_estimator)

                next_state[i] = self._env.reset(index)

            analytic.end_step()
            state1 = self.process_state(next_state)
            done = torch.tensor(done, dtype=torch.float32)

            agent.train(state0, value, action0, probs0, state1, reward, done)

            state0 = state1
            time_estimator.update(n_env)

        agent.save('./models/{0:s}_{1}_{2:d}'.format(self._env_name, config.model, trial))

        print('Saving data...')
        analytic.reset(numpy.array(range(n_env)))
        save_data = analytic.finalize()
        numpy.save('ppo_{0}_{1}_{2:d}'.format(config.name, config.model, trial), save_data)
        analytic.clear()

    def run_rnd_model(self, agent, trial):
        config = self._config
        n_env = config.n_env
        trial = trial + config.shift
        step_counter = StepCounter(int(config.steps * 1e6))

        analytic = ResultCollector()
        analytic.init(n_env, re=(1,), score=(1,), ri=(1,), error=(1,), ext_value=(1,), int_value=(1,))

        reward_avg = RunningAverageWindow(100)
        time_estimator = PPOTimeEstimator(step_counter.limit)

        s = numpy.zeros((n_env,) + self._env.observation_space.shape, dtype=numpy.float32)
        for i in range(n_env):
            s[i] = self._env.reset(i)

        state0 = self.process_state(s)

        while step_counter.running():
            agent.motivation.update_state_average(state0)
            with torch.no_grad():
                value, action0, probs0 = agent.get_action(state0)
            next_state, reward, done, info = self._env.step(agent.convert_action(action0.cpu()))

            ext_reward = torch.tensor(reward, dtype=torch.float32)
            int_reward = agent.motivation.reward(state0).cpu().clip(0.0, 1.0)

            if info is not None:
                if 'normalised_score' in info:
                    analytic.add(normalised_score=(1,))
                    score = torch.tensor(info['normalised_score']).unsqueeze(-1)
                    analytic.update(normalised_score=score)
                if 'raw_score' in info:
                    analytic.add(score=(1,))
                    score = torch.tensor(info['raw_score']).unsqueeze(-1)
                    analytic.update(score=score)

            error = agent.motivation.error(state0).cpu()
            analytic.update(re=ext_reward,
                            ri=int_reward,
                            ext_value=value[:, 0].unsqueeze(-1).cpu(),
                            int_value=value[:, 1].unsqueeze(-1).cpu(),
                            error=error)

            env_indices = numpy.nonzero(numpy.squeeze(done, axis=1))[0]
            stats = analytic.reset(env_indices)
            step_counter.update(n_env)

            for i, index in enumerate(env_indices):
                reward_avg.update(stats['re'].sum[i])

                print('Run {0:d} step {1:d}/{2:d} training [ext. reward {3:f} int. reward (max={4:f} mean={5:f} std={6:f}) steps {7:d}  mean reward {8:f} score {9:f})]'.format(
                    trial, step_counter.steps, step_counter.limit, stats['re'].sum[i], stats['ri'].max[i], stats['ri'].mean[i],
                    stats['ri'].std[i],
                    int(stats['re'].step[i]), reward_avg.value().item(), stats['score'].sum[i]))
                print(time_estimator)
                next_state[i] = self._env.reset(index)

            state1 = self.process_state(next_state)

            reward = torch.cat([ext_reward, int_reward], dim=1)
            done = torch.tensor(1 - done, dtype=torch.float32)
            analytic.end_step()

            agent.train(state0, value, action0, probs0, state1, reward, done)

            state0 = state1
            time_estimator.update(n_env)

        agent.save('./models/{0:s}_{1}_{2:d}'.format(config.name, config.model, trial))

        print('Saving data...')
        analytic.reset(numpy.array(range(n_env)))
        save_data = analytic.finalize()
        numpy.save('ppo_{0}_{1}_{2:d}'.format(config.name, config.model, trial), save_data)
        analytic.clear()

    def run_snd_model(self, agent, trial):
        config = self._config
        n_env = config.n_env
        trial = trial + config.shift
        step_counter = StepCounter(int(config.steps * 1e6))

        analytic = ResultCollector()
        analytic.init(n_env, re=(1,), score=(1,), ri=(1,), error=(1,), feature_space=(1,), state_space=(1,), ext_value=(1,), int_value=(1,))

        reward_avg = RunningAverageWindow(100)
        time_estimator = PPOTimeEstimator(step_counter.limit)

        s = numpy.zeros((n_env,) + self._env.observation_space.shape, dtype=numpy.float32)
        for i in range(n_env):
            s[i] = self._env.reset(i)

        state0 = self.process_state(s)

        while step_counter.running():
            agent.motivation.update_state_average(state0)
            with torch.no_grad():
                features, value, action0, probs0 = agent.get_action(state0)
            next_state, reward, done, info = self._env.step(agent.convert_action(action0.cpu()))

            ext_reward = torch.tensor(reward, dtype=torch.float32)
            int_reward = agent.motivation.reward(state0).cpu().clip(0.0, 1.0)

            if info is not None:
                if 'normalised_score' in info:
                    analytic.add(normalised_score=(1,))
                    score = torch.tensor(info['normalised_score']).unsqueeze(-1)
                    analytic.update(normalised_score=score)
                if 'raw_score' in info:
                    analytic.add(score=(1,))
                    score = torch.tensor(info['raw_score']).unsqueeze(-1)
                    analytic.update(score=score)

            error = agent.motivation.error(state0).cpu()
            cnd_state = agent.network.cnd_model.preprocess(state0)
            analytic.update(re=ext_reward,
                            ri=int_reward,
                            ext_value=value[:, 0].unsqueeze(-1).cpu(),
                            int_value=value[:, 1].unsqueeze(-1).cpu(),
                            error=error,
                            state_space=cnd_state.norm(p=2, dim=[1, 2, 3]).unsqueeze(-1).cpu(),
                            feature_space=features.norm(p=2, dim=1, keepdim=True).cpu())

            env_indices = numpy.nonzero(numpy.squeeze(done, axis=1))[0]
            stats = analytic.reset(env_indices)
            step_counter.update(n_env)

            for i, index in enumerate(env_indices):
                # step_counter.update(int(stats['ext_reward'].step[i]))
                reward_avg.update(stats['re'].sum[i])

                print(
                    'Run {0:d} step {1:d}/{2:d} training [ext. reward {3:f} int. reward (max={4:f} mean={5:f} std={6:f}) steps {7:d}  mean reward {8:f} score {9:f} feature space (max={10:f} mean={11:f} std={12:f})]'.format(
                        trial, step_counter.steps, step_counter.limit, stats['re'].sum[i], stats['ri'].max[i], stats['ri'].mean[i], stats['ri'].std[i],
                        int(stats['re'].step[i]), reward_avg.value().item(), stats['score'].sum[i], stats['feature_space'].max[i], stats['feature_space'].mean[i],
                        stats['feature_space'].std[i]))
                print(time_estimator)
                next_state[i] = self._env.reset(index)

            state1 = self.process_state(next_state)

            reward = torch.cat([ext_reward, int_reward], dim=1)
            done = torch.tensor(1 - done, dtype=torch.float32)
            analytic.end_step()

            agent.train(state0, value, action0, probs0, state1, reward, done)

            state0 = state1
            time_estimator.update(n_env)

        agent.save('./models/{0:s}_{1}_{2:d}'.format(config.name, config.model, trial))

        print('Saving data...')
        analytic.reset(numpy.array(range(n_env)))
        save_data = analytic.finalize()
        numpy.save('ppo_{0}_{1}_{2:d}'.format(config.name, config.model, trial), save_data)
        analytic.clear()

    def run_sp_model(self, agent, trial):
        config = self._config
        n_env = config.n_env
        trial = trial + config.shift
        step_counter = StepCounter(int(config.steps * 1e6))

        analytic = ResultCollector()
        analytic.init(n_env, re=(1,), score=(1,), ri=(1,), error=(1,), ext_value=(1,), int_value=(1,))

        reward_avg = RunningAverageWindow(100)
        time_estimator = PPOTimeEstimator(step_counter.limit)

        s = numpy.zeros((n_env,) + self._env.observation_space.shape, dtype=numpy.float32)
        for i in range(n_env):
            s[i] = self._env.reset(i)

        state0 = self.process_state(s)

        while step_counter.running():
            with torch.no_grad():
                value, action0, probs0 = agent.get_action(state0)
            next_state, reward, done, info = self._env.step(agent.convert_action(action0.cpu()))

            ext_reward = torch.tensor(reward, dtype=torch.float32)
            int_reward = agent.motivation.reward(state0, action0, self.process_state(next_state)).cpu().clip(0.0, 1.0)

            if info is not None:
                if 'normalised_score' in info:
                    analytic.add(normalised_score=(1,))
                    score = torch.tensor(info['normalised_score']).unsqueeze(-1)
                    analytic.update(normalised_score=score)
                if 'raw_score' in info:
                    analytic.add(score=(1,))
                    score = torch.tensor(info['raw_score']).unsqueeze(-1)
                    analytic.update(score=score)

            env_indices = numpy.nonzero(numpy.squeeze(done, axis=1))[0]
            stats = analytic.reset(env_indices)
            step_counter.update(n_env)

            for i, index in enumerate(env_indices):
                reward_avg.update(stats['re'].sum[i])

                print('Run {0:d} step {1:d}/{2:d} training [ext. reward {3:f} int. reward (max={4:f} mean={5:f} std={6:f}) steps {7:d}  mean reward {8:f} score {9:f})]'.format(
                    trial, step_counter.steps, step_counter.limit, stats['re'].sum[i], stats['ri'].max[i], stats['ri'].mean[i],
                    stats['ri'].std[i],
                    int(stats['re'].step[i]), reward_avg.value().item(), stats['score'].sum[i]))
                print(time_estimator)
                next_state[i] = self._env.reset(index)

            state1 = self.process_state(next_state)

            error = agent.motivation.error(state0, action0, state1).cpu()
            analytic.update(re=ext_reward,
                            ri=int_reward,
                            ext_value=value[:, 0].unsqueeze(-1).cpu(),
                            int_value=value[:, 1].unsqueeze(-1).cpu(),
                            error=error)

            reward = torch.cat([ext_reward, int_reward], dim=1)
            done = torch.tensor(1 - done, dtype=torch.float32)
            analytic.end_step()

            agent.train(state0, value, action0, probs0, state1, reward, done)

            state0 = state1
            time_estimator.update(n_env)

        agent.save('./models/{0:s}_{1}_{2:d}'.format(config.name, config.model, trial))

        print('Saving data...')
        analytic.reset(numpy.array(range(n_env)))
        save_data = analytic.finalize()
        numpy.save('ppo_{0}_{1}_{2:d}'.format(config.name, config.model, trial), save_data)
        analytic.clear()

    def run_icm_model(self, agent, trial):
        config = self._config
        n_env = config.n_env
        trial = trial + config.shift
        step_counter = StepCounter(int(config.steps * 1e6))

        analytic = ResultCollector()
        analytic.init(n_env, re=(1,), score=(1,), ri=(1,), error=(1,), ext_value=(1,), int_value=(1,), feature_space=(1,))

        reward_avg = RunningAverageWindow(100)
        time_estimator = PPOTimeEstimator(step_counter.limit)

        s = numpy.zeros((n_env,) + self._env.observation_space.shape, dtype=numpy.float32)
        for i in range(n_env):
            s[i] = self._env.reset(i)

        state0 = self.process_state(s)

        while step_counter.running():
            with torch.no_grad():
                features, value, action0, probs0 = agent.get_action(state0)
            next_state, reward, done, info = self._env.step(agent.convert_action(action0.cpu()))

            ext_reward = torch.tensor(reward, dtype=torch.float32)
            int_reward = agent.motivation.reward(state0, action0, self.process_state(next_state)).cpu().clip(0.0, 1.0)

            if info is not None:
                if 'normalised_score' in info:
                    analytic.add(normalised_score=(1,))
                    score = torch.tensor(info['normalised_score']).unsqueeze(-1)
                    analytic.update(normalised_score=score)
                if 'raw_score' in info:
                    analytic.add(score=(1,))
                    score = torch.tensor(info['raw_score']).unsqueeze(-1)
                    analytic.update(score=score)

            env_indices = numpy.nonzero(numpy.squeeze(done, axis=1))[0]
            stats = analytic.reset(env_indices)
            step_counter.update(n_env)

            for i, index in enumerate(env_indices):
                reward_avg.update(stats['re'].sum[i])

                print(
                    'Run {0:d} step {1:d}/{2:d} training [ext. reward {3:f} int. reward (max={4:f} mean={5:f} std={6:f}) steps {7:d}  mean reward {8:f} score {9:f} feature space (max={10:f} mean={11:f} std={12:f}))]'.format(
                        trial, step_counter.steps, step_counter.limit, stats['re'].sum[i], stats['ri'].max[i], stats['ri'].mean[i],
                        stats['ri'].std[i], int(stats['re'].step[i]), reward_avg.value().item(), stats['score'].sum[i],
                        stats['feature_space'].max[i], stats['feature_space'].mean[i], stats['feature_space'].std[i]))
                print(time_estimator)
                next_state[i] = self._env.reset(index)

            state1 = self.process_state(next_state)

            error = agent.motivation.error(state0, action0, state1).cpu()
            analytic.update(re=ext_reward,
                            ri=int_reward,
                            ext_value=value[:, 0].unsqueeze(-1).cpu(),
                            int_value=value[:, 1].unsqueeze(-1).cpu(),
                            error=error,
                            feature_space=features.norm(p=2, dim=1, keepdim=True).cpu())

            reward = torch.cat([ext_reward, int_reward], dim=1)
            done = torch.tensor(1 - done, dtype=torch.float32)
            analytic.end_step()

            agent.train(state0, value, action0, probs0, state1, reward, done)

            state0 = state1
            time_estimator.update(n_env)

        agent.save('./models/{0:s}_{1}_{2:d}'.format(config.name, config.model, trial))

        print('Saving data...')
        analytic.reset(numpy.array(range(n_env)))
        save_data = analytic.finalize()
        numpy.save('ppo_{0}_{1}_{2:d}'.format(config.name, config.model, trial), save_data)
        analytic.clear()
