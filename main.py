
from input_feeder import InputFeeder
from mouse_controller import MouseController
import numpy as np
from openvino.inference_engine import IENetwork,IECore
import cv2
from matplotlib import pyplot as plt
from argparse import ArgumentParser


#initialize models
import face_detection
face_detection_model = 'models/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001'
fd = face_detection.face_detection(face_detection_model,device='CPU')
fd.load_model()
fd.get_input_name()

import landmark_detection
landmark_detection_model = 'models/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009'
ld = landmark_detection.landmark_detection(landmark_detection_model,device='GPU')
ld.load_model()
ld.get_input_name()

import head_pose_estimation
head_pose_estimation_model = 'models/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001'
hd = head_pose_estimation.head_pose_estimation(head_pose_estimation_model,device='GPU')
hd.load_model()
hd.get_input_name()

import gaze_estimation
gaze_estimation_model = 'models/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002'
ge = gaze_estimation.gaze_estimation(gaze_estimation_model,device='GPU')
ge.load_model()
ge.get_input_name()


def build_argparser():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_path', default= None)
    parser.add_argument("-o", "--output_path", default= 'bin/demo_out6.mp4',
                        type=str,required=True)
    parser.add_argument("-t", "--inputType", default= 'video',
                        type=str,help='Type either video or image')
    parser.add_argument('--mouse_precision', default='medium',
                        help='Mouse movement precision')
    parser.add_argument('--mouse_speed', default='medium',
                        help='Mouse movement speed')
    
    return parser

def visualize_frame(frame,face,x_coord,y_coord,gaze_vec,boxes,result):
    gaze_x = int(gaze_vec[0]*100)
    gaze_y = int(gaze_vec[1]*100)
    cv2.arrowedLine(face, (x_coord[0], y_coord[0]),
                    (x_coord[0] + gaze_x, y_coord[0] - gaze_y),
                    (255,0,255), 5)
    cv2.arrowedLine(face, (x_coord[1], y_coord[1]),
                    (x_coord[1] + gaze_x, y_coord[1] - gaze_y),
                    (255,0,255), 5)

    frame[boxes[0][1]:boxes[0][3], boxes[0][0]:boxes[0][2]] = face    
    return frame

def process_frame(frame):
    
    #calling face detection
    input_img = fd.pre_process_input(frame)
    result = fd.predict(input_img)
    _,boxes = fd.preprocess_output(result)
    face_location = fd.get_face_location()
    
    if (len(boxes) < 1):
        return "No face detected", frame

    # calling landmark detection
    crop_percentage = 0.5
    pre_processed_img = ld.pre_process_input(face_location[0])
    result = ld.predict(pre_processed_img)
    output_image,x_axis,y_axis = ld.preprocess_output(result)
    
    left_eye_crop = ld.crop_left_eye(crop_percentage)
    right_eye_crop = ld.crop_right_eye(crop_percentage)
    
    # Calling head pose estimation
    pre_processed_img = hd.pre_process_input(face_location[0])
    result = hd.predict(pre_processed_img)
    headpose = hd.preprocess_output(result)
    

    # calling gaze model
    res_left = ge.pre_process_input(left_eye_crop)
    res_right = ge.pre_process_input(right_eye_crop)
    result_ge = ge.predict(headpose,res_left,res_right)
    gaze_vec = result_ge['gaze_vector'][0, :]
    
    '''
    rvec = np.array([0, 0, 0], np.float)
    tvec = np.array([0, 0, 0], np.float)
    camera_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float)

    result, _ = cv2.projectPoints(gaze_vec, rvec, tvec, camera_matrix, None)
    result = result[0][0]
    '''   
    #for visualizing 
    frame = visualize_frame(frame,face_location[0],x_axis,y_axis,gaze_vec,boxes,result)
    return gaze_vec,frame #result,frame


def process_video(input_video, video_output):
    if input_video is None:
        feed = InputFeeder(input_type='cam')
    else:
        feed = InputFeeder(input_type='video', input_file=input_video)

    feed.load_data()

    w = int(feed.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(feed.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(feed.cap.get(cv2.CAP_PROP_FPS))
    fps=int(fps/4)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_output, fourcc, fps, (w, h), True)

    frame_counter = 0
    for frame in feed.next_batch():
        if frame is not None:
            frame_counter += 1
            key = cv2.waitKey(60)
            result, output_frame = process_frame(frame)

            out.write(output_frame)

            print("Frame: {} result: {}".format(frame_counter,result))
            
            esc_code = 27
            if key == esc_code:
                break

            if mouse_controller is not None:
                #print("Moving mouse: ",result[0], result[1])
                try:
                    mouse_controller.move(result[0], result[1])
                    pass
                except Exception as e:
                      print("Mouse controller exception:\n",e)
        else:
            break



    out.release()
    feed.close()
    print("Saved the video")

if __name__ == '__main__':
    args = build_argparser().parse_args()
    input_path = args.input_path
    output_path = args.output_path
    #initialize mouse controller
    mouse_controller = MouseController(args.mouse_precision,args.mouse_speed)
    
    if(args.inputType == 'image'):
        input_image = input_path
        feed = InputFeeder(input_type='image', input_file=input_image)
        feed.load_data()
        frame = feed.cap
        _,output_img = process_frame(frame)
        cv2.imwrite(output_path,output_img)
        
    elif(args.inputType == 'video'):
        process_video(input_path, output_path)
        
    else:
        print("Invalid input type")
