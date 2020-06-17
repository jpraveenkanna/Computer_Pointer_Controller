# Computer Pointer Controller

The application tracks the eye movement of the camera feed and controls the movement of the mouse pointer.


# How it Works
The application uses 4 deep learning models. Each frame is processed to crop the face, identify left and right eye, crop left and right eye, estimate the headpose of the face and finally get x and y axis from gaze estimation model to control the mouse. The design pipeline is as follows.

<pic - design pipeline>

# Project Setup

## Project Setup on Ubuntu machine
* Download the deployement package from `deployement_Ubuntu_18.04.4 LTS/deploy_package.tar.xz`
* Extract the archive.
* Open the terminal from the extracted archive folder
* To install dependency execute the following command
<pre>sudo ./install_dependency.sh</pre>
 
* To check whether it is working properly. Run the demo script.
<pre>./demo-script.sh</pre>

## Project Setup on Windows or Mac
* Clone the repository using Git.
<pre>git clone 'https://github.com/jpraveenkanna/Computer_Pointer_Controller.git'</pre> 

* Install OpenVino 2020. The instructions are available [here]('https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_windows.html')


* Create Virtual environment inside the cloned folder <pre>python3 -m venv bin/venv</pre>

* Install python dependencies <pre>pip3 install -r requirements.txt</pre>


## Demo

* Intialize openvino in the terminal.
* Activate the virtual environment <pre>bin/venv/bin/activate</pre> 
* To control mouse pointer on the sample video. Run the following python script <pre>python src/main.py -i "bin/inputs/demo.mp4" -o "bin/outputs/demo_out.mp4" --inputType "video" --mouse_speed "fast" --mouse_precision "medium" --visualize "True"</pre>
* To control mouse pointer on the camera feed. Run the following python script <pre>python src/main.py  -o "bin/outputs/cam_out.mp4" --inputType "video" --mouse_speed "fast" --mouse_precision "medium" --visualize "True"</pre>

* To stop the application press `esc` key.

# Folder Structure

* `src/face_detection.py` - Python script to handle face detection model.
* `src/gaze_estimation.py` - Python script to handle gaze estimation model.
* `src/head_pose_estimation.py` - Python script to handle headpose estimation model.
* `src/landmark_detection.py` - Python script to handle face landmark detection model.
* `src/mouse_controller.py` - Script to control mouse movement.
* `src/input_feeder.py` - Script to load video frame to the model.
* `src/main.py` - The main python script to run the application.
* `models` - Contains models of different precision.
* `deployement_Ubuntu_18.04.4 LTS` - Deployement package for Linux machine.
* `bin/inputs` - Contains sample input video.
* `demo-script.sh` - Bash script to run the demo.
* `requirements.txt` - Python package dependencies.

## Documentation
Available arguments to  run `main.py` script 

Required argumets:
 --inputType                    Type of video input. Options - [video,cam,image]
 --output_path                  Save path for the video file 

optional arguments:
  --help                         show this help message and exit
  --input_path                   Input path of video file to run the application
  --mouse_precision              Mouse movement precision. Options - [low, medium,high]
  --mouse_speed                  Mouse movement speed. Options -[slow, medium, fast]
  --face_detection_model         Path of face detection model without file extension
  --landmark_detection_model     Path of landmark detection model without file extension
  --head_pose_estimation_model   Path of headpose estimation model without file extension
  --gaze_estimation_model        Path of Gaze estimation model without file extension
  --device                       Target hardware type to run inference on. Options -[CPU, GPU, FPGA, MYRIAD]
  --visualize                    To visualize model intermediate output. Options - [True,False]


## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.


### Edge Cases
* Decent lighting condition is necessary to identify the face.
* If two or more face is available only one face is tracked to control mouse. It depends on the confidence of the face detection model.
* A frame is processed for every 10 frames. so sudden movements cannot be tracked.
