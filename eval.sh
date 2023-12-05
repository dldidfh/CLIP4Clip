#!/bin/bash


DATA_PATH='./datas'
VIDEO_PATH=${DATA_PATH}'/videos/piadata_224_fps3_cutting/'
JSON_PATH=${DATA_PATH}'/test.json'



# --val_csv  ${TEST_CSV} \

# RESUME_MODEL='ckpts/nsight_system_test/pytorch_model.bin.0'
RESUME_MODEL="/home/ubuntu/work/for_train/internvideo/Downstream/Video-Text-Retrieval/checkpoints/pretrained_model/vit_b_hybrid_pt_800e.pth"
TEST_CSV=${DATA_PATH}'/test.csv'
OUTPUT_DIR='./ckpts/test'
python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 \
main_task_retrieval.py \
--do_eval \
--val_csv "/home/ubuntu/work/for_train/internvideo/Downstream/Video-Text-Retrieval/data/video-text-retrieval_data/MSR-VTT/anns/MSRVTT_JSFUSION_test.csv" \
--data_path ${JSON_PATH} \
--features_path "/home/ubuntu/work/datas/msrvtt/MSRVTT/videos/all" \
--output_dir ${OUTPUT_DIR} \
--init_model ${RESUME_MODEL} \
--num_thread_reader=0 --max_words 32 --max_frames 12 --batch_size_val 64 --datatype msrvtt --expand_msrvtt_sentences  --feature_framerate 1 --loose_type --linear_patch 2d --sim_header meanP 
