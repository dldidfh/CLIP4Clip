"""
Used to compress video in: https://github.com/ArrowLuo/CLIP4Clip
Author: ArrowLuo
"""
import os
import argparse
import ffmpeg
import subprocess
import time
import multiprocessing
from multiprocessing import Pool
import shutil
import json 
try:
    from psutil import cpu_count
except:
    from multiprocessing import cpu_count
# multiprocessing.freeze_support()

def compress(paras):
    input_video_path, output_video_path, start_time, end_time = paras
    try:
        command = ['ffmpeg',
                '-y',  # (optional) overwrite output file if it exists
                '-i', input_video_path,
                '-filter:v',
                'scale=\'if(gt(a,1),trunc(oh*a/2)*2,224)\':\'if(gt(a,1),224,trunc(ow*a/2)*2)\'',  # scale to 224
                '-map', '0:v',
                '-r', '3',  # frames per second
                '-ss', f"00:00:{start_time}",
                '-to', f"00:00:{end_time}",
                output_video_path,
                ]
        ffmpeg = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = ffmpeg.communicate()
        retcode = ffmpeg.poll()
        # print something above for debug
    except Exception as e:
        raise e

def prepare_input_output_pairs(input_root, output_root, json_path):
    input_video_path_list = []
    output_video_path_list = []
    start_time_list = []
    end_time_list = []

    with open(json_path, "r") as rj : 
        data = json.load(rj)
    for d in data['videos']:
        input_video_path = os.path.join(input_root, d['video_id'] + '.mp4')
        output_video_path = os.path.join(output_root, d['video_id'] + '.mp4')
        start_time = int(d['start time'])
        end_time = int(d['end time'])

        # if already exist
        if os.path.isfile(output_video_path) and os.path.getsize(output_video_path) > 0:
            continue
        if os.path.isfile(input_video_path) and os.path.getsize(input_video_path) > 0:
            input_video_path_list.append(input_video_path)
            output_video_path_list.append(output_video_path)
            start_time_list.append(start_time)
            end_time_list.append(end_time)
        else: 
            continue
        
    return input_video_path_list, output_video_path_list, start_time_list, end_time_list


    # for root, dirs, files in os.walk(input_root):
    #     for file_name in files:
    #         input_video_path = os.path.join(root, file_name)
    #         output_video_path = os.path.join(output_root, file_name)
    #         if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
    #             pass
    #         else:
    #             input_video_path_list.append(input_video_path)
    #             output_video_path_list.append(output_video_path)

    return input_video_path_list, output_video_path_list, start_time_list, end_time_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compress video for speed-up')
    parser.add_argument('--input_root', type=str, help='input root')
    parser.add_argument('--output_root', type=str, help='output root')
    parser.add_argument('--json_path', type=str, help='start time, end time json file path')

    args = parser.parse_args()

    input_root = args.input_root
    output_root = args.output_root
    json_path = args.json_path

    assert input_root != output_root

    if not os.path.exists(output_root):
        os.makedirs(output_root, exist_ok=True)

    input_video_path_list, output_video_path_list, start_time_list, end_time_list = prepare_input_output_pairs(input_root, output_root, json_path)
    print(f"Total input len : {len(input_video_path_list)}\noutput len : {len(output_video_path_list)}\nstart time len : {len(start_time_list)}\nend time len : {len(end_time_list)}")
    num_works = cpu_count()
    print("Begin with {}-core logical processor.".format(num_works))

    pool = Pool(num_works)
    data_dict_list = pool.map(compress,
                              [(input_video_path, output_video_path, start_time, end_time) for
                               input_video_path, output_video_path, start_time, end_time in
                               zip(input_video_path_list, output_video_path_list, start_time_list, end_time_list)])
    pool.close()
    pool.join()

    print("Compress finished, wait for checking files...")
    for input_video_path, output_video_path in zip(input_video_path_list, output_video_path_list):
        if os.path.exists(input_video_path):
            if os.path.exists(output_video_path) is False or os.path.getsize(output_video_path) < 1.:
                # shutil.copyfile(input_video_path, output_video_path)
                print("Copy and replace file: {}".format(output_video_path))