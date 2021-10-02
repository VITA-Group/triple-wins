## Description
The page contains reproducible code for Cifar-10 experiments.
## Train Network
```
CUDA_VISIBLE_DEVICES=gid python3 train_base.py train cifar10_resnet_38 -d cifar10 --attack_algo ifgm_main --defend_algo ifgm_max_v2 --save-folder /path/to/folder
```
defense type = Max-Avg attack; Reported defense type = main branch attack.

### Evaluate Network
```
CUDA_VISIBLE_DEVICES=gid python3 train_base.py test cifar10_resnet_38 -d cifar10 --resume checkpoint_60000.pth.tar
```
We test all attacks by default.
