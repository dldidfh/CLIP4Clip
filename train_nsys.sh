#!/bin/bash

/home/ubuntu/anaconda3/envs/clip4clip_train/bin/python -m torch.distributed.launch --nproc_per_node=1 --standalone --nnodes=1 \
main_task_retrieval.py --do_train \
--num_thread_reader=0 \
--epochs=2 \
--batch_size=32 \
--n_display=1 \
--train_csv  './datas/test.csv' \
--val_csv  './datas/test.csv' \
--data_path './datas/test.json' \
--features_path './datas/videos/piadata_224_fps3_cutting/' \
--output_dir './ckpts/nsight_system_test' \
--lr 1e-4 \
--max_words 32 \
--max_frames 12 \
--batch_size_val 64 \
--datatype msrvtt \
--expand_msrvtt_sentences  \
--feature_framerate 1 \
--coef_lr 1e-3 \
--freeze_layer_num 0  \
--slice_framepos 2 \
--loose_type \
--linear_patch 2d \
--sim_header meanP \
--pretrained_clip_name ViT-B/32 \
--use_ram
