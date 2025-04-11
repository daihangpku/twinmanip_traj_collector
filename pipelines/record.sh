# This script is used to test the get_rgbd.py script

export DIR_NAME="record/franka-track"
export PREHEAT_TIME=2
export RECORD_FRAMES=10
export FPS=15


export PROJECT_ROOT=$(pwd)


python3 twinmanip_realsense_recorder/get_rgbd.py \
--dir_name $DIR_NAME \
--preheat_time $PREHEAT_TIME \
--fps $FPS \
--record_frames $RECORD_FRAMES 