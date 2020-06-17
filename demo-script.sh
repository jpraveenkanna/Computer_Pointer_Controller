#!/bin/bash
source bin/venv/bin/activate
pip3 install -r requirements.txt
source /opt/intel/openvino/bin/setupvars.sh
#python main.py -i "bin/demo.mp4" -o "bin/cam_out_1.mp4" --inputType "cam" --mouse_speed "fast" --mouse_precision "medium" --visualize "True"
#python main.py -i "bin/demo_img_1.png" -o "bin/test.png" -t "image" --mouse_speed "fast"
python src/main.py -i "bin/inputs/demo.mp4" -o "bin/outputs/demo_out_5.mp4" --inputType "video" --mouse_speed "fast" --mouse_precision "medium" --visualize "True"

