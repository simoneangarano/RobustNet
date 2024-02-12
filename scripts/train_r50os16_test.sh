#!/usr/bin/env bash
    # Example on Cityscapes
    exp = cityscapes
    python train.py \
        --dataset $exp --val_dataset $exp \
        --arch network.deepv3.DeepR50V3PlusD \
        --city_mode train \
        --sgd \
        --lr_schedule poly \
        --lr 0.01 \
        --poly_exp 0.9 \
        --max_cu_epoch 10000 \
        --class_uniform_pct 0.0 \
        --class_uniform_tile 1024 \
        --crop_size 768 \
        --scale_min 0.5 \
        --scale_max 2.0 \
        --rrotate 0 \
        --max_iter 40000 \
        --bs_mult 8 \
        --gblur \
        --color_aug 0.5 \
        --date 0212 \
        --exp $exp \
        --ckpt ./logs/ \
        --tb_path ./logs/ \
        --wt_reg_weight 0.0 \
        --relax_denom 0.0 \
        --cov_stat_epoch 0 \
        --wt_layer 0 0 0 0 0 0 0 2>&1 | tee logs/$exp.txt