
from input_feeder import InputFeeder
from mouse_controller import MouseController
import numpy as np
from openvino.inference_engine import IENetwork,IECore
import cv2
from matplotlib import pyplot as plt
from argparse import ArgumentParser
import face_detection
import landmark_detection
import head_pose_estimation
import gaze_estimation
import logging

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

    for box in boxes:
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (232, 255, 244), 2)

    cv2.imshow("Preview",frame)
    cv2.waitKey(60)     
    return frame

def process_frame(frame,visualize):
    
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

    #for visualizing 
    if(visualize == 'True'):
        frame = visualize_frame(frame,face_location[0],x_axis,y_axis,gaze_vec,boxes,result)
    return gaze_vec,frame 


def process_video(input_video, video_output,visualize):
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
            result, output_frame = process_frame(frame,visualize)

            out.write(output_frame)

            print("Frame: {} result: {}".format(frame_counter,result))
            logger.info("Frame: {} result: {}".format(frame_counter,result))
            
            esc_code = 27
            if key == esc_code:
                break

            if mouse_controller is not None:
                try:
                    mouse_controller.move(result[0], result[1])
                    pass
                except Exception as e:
                      print("Mouse controller exception:\n",e)
                      logger.info("Mouse controller exception:{}".format(e))
                     
        else:
            break


    cv2.destroyAllWindows() 
    out.release()
    feed.close()
    print("Saved the video")
    logger.info("Saved the video")

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_path', default= None)
    parser.add_argument("-o", "--output_path",
                        type=str,required=True)
    parser.add_argument("-t", "--inputType", default= 'video',
                        type=str,help='Options - [video,cam,image]')
    parser.add_argument('--mouse_precision', default='medium',
                        help='Mouse movement precision. Options - [low, medium, high]')
    parser.add_argument('--mouse_speed', default='medium',
                        help='Mouse movement speed. Options -[slow, medium, fast]')
    parser.add_argument('--face_detection_model', default='models/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001',
                        help='Path of face detection model without file extension')
    parser.add_argument('--landmark_detection_model', default='models/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009',
                        help='Path of landmark detection model without file extension')
    parser.add_argument('--head_pose_estimation_model', default='models/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001',
                        help='Path of headpose estimation model without file extension')
    parser.add_argument('--gaze_estimation_model', default='models/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002',
                        help='Path of Gaze estimation model without file extension')

    parser.add_argument('--device', default='CPU',
                        help='Target hardware type to run inference on. Options - [CPU, GPU, FPGA, VPU]')
    parser.add_argument('--visualize', default='True',
                        help='To visualize model intermediate output. Options - [True,False]')

    return parser

if __name__ == '__main__':
    args = build_argparser().parse_args()

    #Logging
    logging.basicConfig(filename="bin/mouse_controller.log", 
                        format='%(asctime)s %(message)s', 
                        filemode='w') 
    logger=logging.getLogger() 
    logger.setLevel(logging.DEBUG) 
    
    #initialize models
    face_detection_model = args.face_detection_model
    fd = face_detection.face_detection(face_detection_model,device=args.device)
    fd.load_model()
    fd.check_model()
    fd.get_input_name()

    landmark_detection_model = args.landmark_detection_model
    ld = landmark_detection.landmark_detection(landmark_detection_model,args.device)
    ld.load_model()
    ld.check_model()
    ld.get_input_name()

    head_pose_estimation_model = args.head_pose_estimation_model
    hd = head_pose_estimation.head_pose_estimation(head_pose_estimation_model,args.device)
    hd.load_model()
    hd.check_model()
    hd.get_input_name()

    gaze_estimation_model = args.gaze_estimation_model
    ge = gaze_estimation.gaze_estimation(gaze_estimation_model,args.device)
    ge.load_model()
    ge.check_model()
    ge.get_input_name()



    #initialize mouse controller
    mouse_controller = MouseController(args.mouse_precision,args.mouse_speed)
    
    if(args.inputType == 'image'):
        input_image = args.input_path
        feed = InputFeeder(input_type='image', input_file=input_image)
        feed.load_data()
        frame = feed.cap
        _,output_img = process_frame(frame,args.visualize)
        cv2.imshow("Preview",output_img)
        cv2.imwrite(args.output_path,output_img)
    
    elif(args.inputType == 'video'):
        process_video(args.input_path, args.output_path,args.visualize)
    elif(args.inputType == 'cam'):
        process_video(None, args.output_path,args.visualize)
    else:
        print("Invalid input type")
