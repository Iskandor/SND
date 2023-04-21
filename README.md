# Self-supervised Network Destilation
Self-supervised Network Destilation (SND) is class of intrinsic motivation algorithms based on the distillation error as a novelty indicator, where the target model is trained using self-supervised learning.

## Methods

[SND Vanilla original implementation](https://github.com/michalnand/reinforcement_learning)

## Results


### Replication
```
python main.py -a ppo --env montezuma --config 2 --device cuda --gpus 0 --num_threads 4
```

## Citation
```
@article{pechac2023exploration,
  title={Exploration by self-supervised exploitation},
  author={Pech{\'a}{\v{c}}, Matej and Chovanec, Michal and Farka{\v{s}}, Igor},
  journal={arXiv preprint arXiv:2302.11563},
  year={2023}
}
```
