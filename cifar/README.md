## Description
The page contains reproducible code for Cifar-10 experiments.
## Train Network
```
CUDA_VISIBLE_DEVICES=gid python3 train_base.py train cifar10_resnet_38 -d cifar10 --attack_algo pgd_avg --defend_algo pgd_max --save-folder branch_adv_max
```

### Evaluate Network
```
CUDA_VISIBLE_DEVICES=gid python3 train_base.py test cifar10_resnet_38 -d cifar10 --resume checkpoint_60000.pth.tar
```
