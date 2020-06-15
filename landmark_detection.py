
import numpy as np
from openvino.inference_engine import IENetwork,IECore
import cv2
from matplotlib import pyplot as plt
from math import ceil

class landmark_detection:

    def __init__(self, model_name, device='CPU', extensions=None):       
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
         
    def load_model(self):
        # Loading network to device
        self.core = IECore()
        self.model=self.core.read_network(self.model_structure, self.model_weights)
        self.net = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)
        print("Successfully loaded Landmark detection model")
        
    def pre_process_input(self,image):
        self.image=image
        resized_img = cv2.resize(self.image, (48, 48))
        input_img = np.moveaxis(resized_img, -1, 0)
        input_img=np.expand_dims(input_img, axis=0)
        return input_img
        
    def check_model(self):
        layers_supported = self.core.query_network(network=self.model, device_name="CPU")
        layers_in_model = self.model.layers.keys()
        all_layers_supported = True
        for l in layers_in_model:
            if l  not in layers_supported:
                all_layers_supported = False
                print('Layer', l, '- Not supported')
        if all_layers_supported:
            print('All layers supported - Landmark detection model')
    
    def get_input_name(self):
        self.input_name = next(iter(self.net.inputs))
        
    def predict(self, image):
        result = self.net.infer({self.input_name:image})
        return result
    
    def preprocess_output(self, res):
        prediction = res.get('95')
        self.coord = []
        for i in prediction[0]:
            self.coord.append(i[0][0])
        tmp_image = self.image
        self.ih, self.iw = self.image.shape[:-1]
        x_coord,y_coord = [],[]
        for x,y in zip(self.coord[::2],self.coord[1::2]):
            x = np.int(self.iw * x)
            y = np.int(self.ih * y)
            x_coord.append(x)
            y_coord.append(y)
            #tmp_image = cv2.circle(tmp_image,(x, y), 1, (0,255,0), 1)
        return tmp_image,x_coord,y_coord
        
    def plot_image(self,output_image):
        img = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize = (8,5))
        plt.imshow(img,interpolation='nearest', aspect='auto')
        plt.show()
        
    def save_image(self,image_id,image):
        cv2.imwrite("bin/face_landmark-{}.jpg".format(image_id+1), image)    

    def crop_left_eye(self, crop_percent):
        left_eye = (int((self.iw*self.coord[0])),int(self.ih*self.coord[1]))
        
        diagnol = (int(self.iw*self.coord[4])-int(self.iw*self.coord[0]),
                   int(self.ih*self.coord[5])-int(self.ih*self.coord[1]))
        diagnol = [abs(ceil(x*crop_percent)) for x in diagnol]
        
        starting_point = (left_eye[0]-diagnol[0], left_eye[1]-diagnol[1])
        ending_point = (left_eye[0]+diagnol[0], left_eye[1]+diagnol[1])

        left_eye_crop = self.image[starting_point[1]:ending_point[1]+1,starting_point[0]:ending_point[0]+1]
        return left_eye_crop
    
    def crop_right_eye(self, crop_percent):
        right_eye = (int((self.iw*self.coord[2])),int(self.ih*self.coord[3]))
        
        diagnol = (-int(self.iw*self.coord[4])+int(self.iw*self.coord[2]),
                   -int(self.ih*self.coord[5])+int(self.ih*self.coord[3]))
        diagnol = [abs(ceil(x*crop_percent)) for x in diagnol]
        
        starting_point = (right_eye[0]-diagnol[0], right_eye[1]-diagnol[1])
        ending_point = (right_eye[0]+diagnol[0], right_eye[1]+diagnol[1])

        right_eye_crop = self.image[starting_point[1]:ending_point[1]+1,starting_point[0]:ending_point[0]+1]
        return right_eye_crop
