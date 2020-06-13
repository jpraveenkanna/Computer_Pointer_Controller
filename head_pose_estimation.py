
import numpy as np
from openvino.inference_engine import IENetwork,IECore
import cv2
from matplotlib import pyplot as plt
from math import cos, sin,pi

class head_pose_estimation:
    
    
    def __init__(self, model_name, device='CPU', extensions=None):
        
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
         

    def load_model(self):
        
        # Loading network to device
        self.core = IECore()
        self.model=self.core.read_network(self.model_structure, self.model_weights)
       
        self.net = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)
        
        print("Successfully loaded the network")
        
    def pre_process_input(self,image):
        self.image=image
        resized_img = cv2.resize(self.image, (60, 60))
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
                print('Layer', l, 'is not supported')
        if all_layers_supported:
            print('All layers supported')
    
    def get_input_name(self):
        self.input_name = next(iter(self.net.inputs))
        
    def predict(self, image):
        result = self.net.infer({self.input_name:image})
        return result
    
    def preprocess_output(self, result):
        pitch = result.get('angle_p_fc')[0,0]
        roll = result.get('angle_r_fc')[0,0]
        yaw = result.get('angle_y_fc')[0,0]
        pred = [yaw,pitch,roll]
        pred=np.asarray(pred)
        pred=np.expand_dims(pred, axis=0)
        return pred
        
    def plot_image(self,output_image):
        img = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize = (3,3))

        plt.imshow(img,interpolation='nearest', aspect='auto')
        plt.show()
        
    def save_image(self,image_id,image):
        cv2.imwrite("bin/head_pose-{}.jpg".format(image_id+1), image)

        
        
