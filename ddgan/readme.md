# DDGAN


## Training Denoising Diffusion GANs ##
We use the following commands on each dataset for training denoising diffusion GANs.

#### CIFAR-10 ####

We train Denoising Diffusion GANs on CIFAR-10 using 4 32-GB V100 GPU. 
```
python3 train_ddgan.py --dataset cifar10 --exp ddgan_cifar10_exp1 --num_channels 3 --num_channels_dae 128 --num_timesteps 4 \
--num_res_blocks 2 --batch_size 64 --num_epoch 1800 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
--use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 --num_process_per_node 4 \
--ch_mult 1 2 2 2 --save_content
```

#### LSUN Church Outdoor 256 ####

We train Denoising Diffusion GANs on LSUN Church Outdoor 256 using 8 32-GB V100 GPU. 
```
python3 train_ddgan.py --dataset lsun --image_size 256 --exp ddgan_lsun_exp1 --num_channels 3 --num_channels_dae 64 --ch_mult 1 1 2 2 4 4 --num_timesteps 4 \
--num_res_blocks 2 --batch_size 8 --num_epoch 500 --ngf 64 --embedding_type positional --use_ema --ema_decay 0.999 --r1_gamma 1. \
--z_emb_dim 256 --lr_d 1e-4 --lr_g 1.6e-4 --lazy_reg 10 --num_process_per_node 8 --save_content
```

#### CelebA HQ 256 ####

We train Denoising Diffusion GANs on CelebA HQ 256 using 8 32-GB V100 GPUs. 
```
python3 train_ddgan.py --dataset celeba_256 --image_size 256 --exp ddgan_celebahq_exp1 --num_channels 3 --num_channels_dae 64 --ch_mult 1 1 2 2 4 4 --num_timesteps 2 \
--num_res_blocks 2 --batch_size 4 --num_epoch 800 --ngf 64 --embedding_type positional --use_ema --r1_gamma 2. \
--z_emb_dim 256 --lr_d 1e-4 --lr_g 2e-4 --lazy_reg 10  --num_process_per_node 8 --save_content
```


## Evaluation ##
After training, samples can be generated by calling ```test_ddgan.py```. We evaluate the models with single V100 GPU.
Below, we use `--epoch_id` to specify the checkpoint saved at a particular epoch.
Specifically, for models trained by above commands, the scripts for generating samples on CIFAR-10 is
```
python3 test_ddgan.py --dataset cifar10 --exp ddgan_cifar10_exp1 --num_channels 3 --num_channels_dae 128 --num_timesteps 4 \
--num_res_blocks 2 --nz 100 --z_emb_dim 256 --n_mlp 4 --ch_mult 1 2 2 2 --epoch_id 1200
```
