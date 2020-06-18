#!/bin/bash
source bin/venv/bin/activate
source /opt/intel/openvino/bin/setupvars.sh
#python main.py -o "bin/cam_out_1.mp4" --inputType "cam" --mouse_speed "fast" --mouse_precision "medium" --visualize "True"
python src/main.py -i "bin/inputs/demo.mp4" -o "bin/outputs/demo_out.mp4" --inputType "video" --mouse_speed "fast" --mouse_precision "medium" --visualize "False" 

