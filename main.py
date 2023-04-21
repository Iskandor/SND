import argparse
import math
import os
import platform
import subprocess

import psutil
import ray
import torch
from torch import multiprocessing
from torch.multiprocessing import Pool

import PPO_Caveflyer
import PPO_Climber
import PPO_Coinrun
import PPO_Gravitar
import PPO_Jumper
import PPO_Montezuma
import PPO_Pitfall
import PPO_PrivateEye
import PPO_Solaris
import PPO_Venture
from config import load_config_file
from config.Config import Config

envs = {
    'ppo': {
        'gravitar': PPO_Gravitar,
        'montezuma': PPO_Montezuma,
        'pitfall': PPO_Pitfall,
        'private_eye': PPO_PrivateEye,
        'solaris': PPO_Solaris,
        'venture': PPO_Venture,
        'caveflyer': PPO_Caveflyer,
        'coinrun': PPO_Coinrun,
        'climber': PPO_Climber,
        'jumper': PPO_Jumper,
    },
}


def run_ray_parallel(args, experiment):
    @ray.remote(num_gpus=1/args.num_processes, max_calls=1)
    def run_thread_ray(p_thread_params):
        run_thread(p_thread_params)

    for i in range(math.ceil(experiment.trials / args.num_processes)):
        thread_params = []
        for j in range(args.num_processes):
            index = i * args.num_processes + j
            if index < experiment.trials:
                thread_params.append((args.algorithm, args.env, experiment, index))

        ray.get([run_thread_ray.remote(tp) for tp in thread_params])


def run_thread(thread_params):
    algorithm, env, experiment, i = thread_params

    if experiment.gpus:
        torch.cuda.set_device(experiment.gpus[0])

    run(i, algorithm, env, experiment)


def run(id, algorithm, env, experiment):
    print('Starting experiment {0}_{1} on env {2} learning algorithm {3} model {4} {5}'.format(experiment.name, id + experiment.shift, env, algorithm, experiment.model, experiment.type if hasattr(experiment, 'type') else ''))

    env_class = envs[algorithm][env]

    if experiment.model == 'baseline':
        env_class.run_baseline(experiment, id)
    if experiment.model == 'rnd':
        env_class.run_rnd_model(experiment, id)
    if experiment.model == 'snd':
        env_class.run_snd_model(experiment, id)
    if experiment.model == 'icm':
        env_class.run_icm_model(experiment, id)
    if experiment.model == 'sp':
        env_class.run_sp_model(experiment, id)

def write_command_file(args, experiment):
    print(multiprocessing.cpu_count())
    thread_per_env = max(multiprocessing.cpu_count() // experiment.trials, 1)
    if platform.system() == 'Windows':
        file = open("run.bat", "w")
        file.write('set OMP_NUM_THREADS={0}\n'.format(thread_per_env))
        for i in range(experiment.trials):
            file.write('start "" python main.py --env {0} --config {1} -t -s {2} \n'.format(args.env, args.config, i + args.shift))
        file.close()

    if platform.system() == 'Linux':
        file = open("run.sh", "w")
        for i in range(experiment.trials):
            file.write(
                'OMP_NUM_THREADS={0} python3 main.py --env {1} --config {2} -t -s {3} & \n'.format(thread_per_env, args.env, args.config, i + args.shift))
        file.close()


def run_command_file():
    if platform.system() == 'Windows':
        subprocess.call([r'run.bat'])
        if os.path.exists('run.bat'):
            os.remove('run.bat')
    if platform.system() == 'Linux':
        os.chmod('run.sh', 777)
        subprocess.run(['bash', './run.sh'])
        if os.path.exists('./run.sh'):
            os.remove('./run.sh')


def run_torch_parallel(args, experiment):
    multiprocessing.set_start_method('spawn')

    thread_params = []
    for i in range(experiment.trials):
        thread_params.append((args.algorithm, args.env, experiment, i))

    with Pool(args.num_processes) as p:
        p.map(run_thread, thread_params)


def update_config(args, experiment):
    experiment.device = args.device
    experiment.gpus = args.gpus
    experiment.shift = args.shift
    if args.num_threads == 0:
        experiment.num_threads = psutil.cpu_count(logical=True)
    else:
        experiment.num_threads = args.num_threads
    # if args.algorithm == 'ppo':
    #     experiment.steps *= experiment.n_env
    #     experiment.batch_size *= experiment.n_env
    #     experiment.trajectory_size *= experiment.n_env


if __name__ == '__main__':
    print(platform.system())
    print(torch.__version__)
    print(torch.__config__.show())
    print(torch.__config__.parallel_info())
    # torch.autograd.set_detect_anomaly(True)

    for i in range(torch.cuda.device_count()):
        print('{0:d}. {1:s}'.format(i, torch.cuda.get_device_name(i)))

    parser = argparse.ArgumentParser(description='Motivation models learning platform.')

    if not os.path.exists('./models'):
        os.mkdir('./models')

    parser.add_argument('--env', type=str, help='environment name')
    parser.add_argument('-a', '--algorithm', type=str, help='training algorithm', choices=['ppo', 'ddpg', 'a2c', 'dqn'])
    parser.add_argument('--config', type=int, help='id of config')
    parser.add_argument('--device', type=str, help='device type', default='cpu')
    parser.add_argument('--gpus', help='device ids', default=None)
    parser.add_argument('--load', type=str, help='path to saved agent', default='')
    parser.add_argument('-s', '--shift', type=int, help='shift result id', default=0)
    parser.add_argument('-p', '--parallel', action="store_true", help='run envs in parallel mode')
    parser.add_argument('-pb', '--parallel_backend', type=str, default='torch', choices=['ray', 'torch'], help='parallel backend')
    parser.add_argument('--num_processes', type=int, help='number of parallel processes started in parallel mode (0=automatic number of cpus)', default=0)
    parser.add_argument('--num_threads', type=int, help='number of parallel threads running in PPO (0=automatic number of cpus)', default=0)
    parser.add_argument('-t', '--thread', action="store_true", help='do not use: technical parameter for parallel run')

    args = parser.parse_args()
    if args.gpus:
        args.gpus = [int(s) for s in args.gpus.split(',')]
        torch.cuda.set_device(args.gpus[0])
    config = load_config_file(args.algorithm)

    experiment = Config(config[args.env][str(args.config)], "{0}_{1}".format(args.env, str(args.config)))
    update_config(args, experiment)

    if args.load != '':
        env_class = envs[args.algorithm][args.env]
        env_class.test(experiment, args.load)
    else:
        if args.thread:
            experiment.trials = 1

        if args.parallel:
            if args.num_processes == 0:
                num_cpus = psutil.cpu_count(logical=True)
            else:
                num_cpus = min(psutil.cpu_count(logical=True), args.num_processes)
            print('Running parallel {0} trainings'.format(min(experiment.trials, num_cpus)))
            print('Using {0} parallel backend'.format(args.parallel_backend))

            if args.parallel_backend == 'ray':
                if args.gpus:
                    experiment.gpus = None
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus[0])
                ray.shutdown()
                ray.init(num_cpus=num_cpus, num_gpus=1)
                torch.set_num_threads(max(1, num_cpus // experiment.trials))

                run_ray_parallel(args, experiment)
                # write_command_file(args, experiment)
                # run_command_file()
            elif args.parallel_backend == 'torch':
                torch.set_num_threads(1)
                run_torch_parallel(args, experiment)
        else:
            for i in range(experiment.trials):
                run(i, args.algorithm, args.env, experiment)
