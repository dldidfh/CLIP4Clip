#!/bin/bash


DATA_PATH='./datas'
VIDEO_PATH=${DATA_PATH}'/videos/piadata_224_fps3_cutting/'

# start_time case 3
JSON_PATH=${DATA_PATH}'/test.json'
OUTPUT_DIR='./ckpts/nsight_system_test'
TRAIN_CSV=${DATA_PATH}'/test.csv'
TEST_CSV=${DATA_PATH}'/test.csv'

python3 -m torch.distributed.launch --nproc_per_node=1 --standalone --nnodes=1 \
main_task_retrieval.py --do_train \
--num_thread_reader=0 \
--epochs=3 \
--batch_size=32 \
--n_display=1 \
--train_csv  ${TRAIN_CSV} \
--val_csv  ${TEST_CSV} \
--data_path ${JSON_PATH} \
--features_path ${VIDEO_PATH} \
--output_dir ${OUTPUT_DIR} \
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
