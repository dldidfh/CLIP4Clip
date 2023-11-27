#!/bin/bash

RAW_VIDEOS_DIR='/data/input'
COMPRESSED_VIDEOS_DIR='/data/output'
JSON_PATH='/data/start_time.json'

# check if video source directory is correct
if [ ! -d "$RAW_VIDEOS_DIR" ]; then
    echo "Directory $RAW_VIDEOS_DIR does not exist. Please download the videos and place them in the directory."
    exit 1
fi

python3 preprocess/cutting_compress_video.py --input_root $RAW_VIDEOS_DIR --output_root $COMPRESSED_VIDEOS_DIR --json_path $JSON_PATH
~                                    