# ライブラリのインポート
from mpi4py import MPI
from solutions.es.operator import ESOperator
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.environment import ActionTuple, UnityEnvironment
from solutions.base_solution import BaseSolution
from solutions.utils import create_logger

import os
import time
import pickle
import numpy as np
import torch

torch.set_num_threads(1)


class BaseTorchSolution(BaseSolution):
    def __init__(self, device):
        self.modules_to_learn = []
        self.device = torch.device(device)

    def get_action(self, obs):
        with torch.no_grad():
            return self._get_action(obs)

    def get_params(self):
        params = []
        with torch.no_grad():
            for layer in self.modules_to_learn:
                for p in layer.parameters():
                    params.append(p.cpu().numpy().ravel())
        return np.concatenate(params)

    def set_params(self, params):
        params = np.array(params)
        assert isinstance(params, np.ndarray)
        ss = 0
        for layer in self.modules_to_learn:
            for p in layer.parameters():
                ee = ss + np.prod(p.shape)
                p.data = torch.from_numpy(
                    params[ss:ee].reshape(p.shape)
                ).float().to(self.device)
                ss = ee
        assert ss == params.size

    def save(self, filename):
        params = self.get_params()
        np.savez(filename, params=params)

    def load(self, filename):
        with np.load(filename) as data:
            params = data['params']
            self.set_params(params)

    def get_num_params(self):
        return self.get_params().size

    def _get_action(self, obs):
        raise NotImplementedError()

    def reset(self):
        pass

    def get_fitness(self, worker_id, params, seed, num_rollouts):
        self.set_params(params)
        total_scores = []
        for _ in range(num_rollouts):
            self.env.reset()
            behavior_names = list(self.env.behavior_specs.keys())
            decision_steps, terminal_steps = self.env.get_steps(behavior_names[0])
            done = False
            reward = 0
            while not done:
                start = time.time()
                for i in decision_steps.agent_id:
                    obs = decision_steps.obs[0][i] * 255
                    action = self.get_action(obs)
                    action_tuple = ActionTuple(continuous=np.expand_dims(action, axis=0))
                    self.env.set_action_for_agent(behavior_names[0], i, action_tuple)
                self.env.step()
                end = time.time()
                if worker_id == 0:
                    print('1step:', end - start, '[s]')
                decision_steps, terminal_steps = self.env.get_steps(behavior_names[0])
                reward += sum(decision_steps.reward)
                done = len(terminal_steps.interrupted) > 0
                if done:
                    reward += sum(terminal_steps.reward)

            total_scores.append(reward)
        return np.mean(total_scores)

    @staticmethod
    def save_params(solver, solution, model_path):
        solution.set_params(solver.best_param())
        solution.save(model_path)

    def train(self,
              t: int = 1,
              timesteps: int = 1000,
              max_iter: int = 1000,
              reps: int = 1,
              log_dir: str = None,
              save_interval: int = 10,
              seed: int = 42,
              load_model = None,
              init_sigma: float = 0.1,
              algo_number: int = 0
              ):
        ii32 = np.iinfo(np.int32)
        rnd = np.random.RandomState(seed=seed)

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        if rank == 0:
            self.popsize = size * t
            self.max_iter = max_iter
            self.reps = reps
            self.seed = seed
            self.algo_number = algo_number
            self.init_sigma = init_sigma
            logger = create_logger(solution=self, name='train_log', log_dir=log_dir)
            best_so_far = -np.Inf

            num_params = self.get_num_params()
            print('#params={}'.format(num_params))
            if load_model is not None:
                self.load(load_model)
                print('Loaded model from {}'.format(load_model))
                init_params = self.get_params()
            else:
                init_params = num_params * [0]

            solver = ESOperator(x0=init_params, dim=num_params, popsize=size * t, sigma0=init_sigma, algo_number=algo_number).get_solver()
        comm.barrier()

        # Unity環境の生成
        channel1 = EngineConfigurationChannel()
        channel1.set_configuration_parameters(time_scale=10, width=300, height=300, capture_frame_rate=50)
        channel2 = EnvironmentParametersChannel()
        channel2.set_float_parameter("time_steps", float(timesteps))
        self.env = UnityEnvironment(
            file_name=self.file_name,
            no_graphics=False,
            side_channels=[channel1, channel2],
            worker_id=rank,
            seed=seed
        )

        for n_iter in range(max_iter):
            task_seed = rnd.randint(0, ii32.max)
            comm.barrier()
            if rank == 0:
                params_sets = solver.ask()
                c_rank = 0
                for i in range(0, len(params_sets), t):
                    if c_rank == 0:
                        params_set = params_sets[i:i + t]
                    else:
                        data = params_sets[i:i + t]
                        comm.send(data, dest=c_rank, tag=c_rank)
                    c_rank += 1
            else:
                data = comm.recv(source=0, tag=rank)
                params_set = data

            fitness = []
            for i in range(t):
                f = self.get_fitness(rank, params_set[i], task_seed, reps)
                fitness.append(f)

            fitnesses = comm.gather(fitness, root=0)
            if rank == 0:
                fitnesses = np.concatenate(np.array(fitnesses))

                solver.tell(fitnesses)
                logger.info(
                    'Iter={0}, '
                    'max={1:.2f}, avg={2:.2f}, min={3:.2f}, std={4:.2f}'.format(
                        n_iter + 1, np.max(fitnesses), np.mean(fitnesses), np.min(fitnesses), np.std(fitnesses)))

                best_fitness = max(fitnesses)
                if best_fitness > best_so_far:
                    best_so_far = best_fitness
                    model_path = os.path.join(log_dir, 'best.npz')
                    self.save_params(solver=solver, solution=self, model_path=model_path)
                    logger.info('Best model updated, score={}'.format(best_fitness))
                if (n_iter + 1) % save_interval == 0:
                    model_path = os.path.join(log_dir, 'Iter_{}.npz'.format(n_iter + 1))
                    self.save_params(solver=solver, solution=self, model_path=model_path)
            self.reset()
        self.env.close()