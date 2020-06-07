
import numpy as np
from openvino.inference_engine import IENetwork,IECore
import cv2
from matplotlib import pyplot as plt

class face_detection:
    
    
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
        
    def pre_process_input(self,image_path):
        self.image=cv2.imread(image_path)
        #print("Input shape",image.shape) #height, width, channel 
        resized_img = cv2.resize(self.image, (672,384)) #width,height
        #resized_img.shape
        input_img = np.moveaxis(resized_img, -1, 0)
        input_img=np.expand_dims(input_img, axis=0)
        #print("shape: ",input_img.shape)
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
    
    def preprocess_output(self, res):
        prediction = res.get('detection_out')
        
        count=0
        ih, iw = self.image.shape[:-1]
        tmp_image = self.image
        self.boxes=[]

        for results in prediction[0][0]:
            if (np.int(results[1])==1 and results[2] > 0.9):
                count +=1
                xmin = np.int(iw * results[3])
                ymin = np.int(ih * results[4])
                xmax = np.int(iw * results[5])
                ymax = np.int(ih * results[6])

                self.boxes.append([xmin, ymin, xmax, ymax])
                
        #for box in self.boxes:
        #    cv2.rectangle(tmp_image, (box[0], box[1]), (box[2], box[3]), (232, 35, 244), 2)
        return tmp_image
        
    def plot_image(self,output_image):
        img = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize = (15,9))

        plt.imshow(img,interpolation='nearest', aspect='auto')
        plt.show()
        
    def save_image(self,image_id,image):
        cv2.imwrite("bin/face_detection-{}.jpg".format(image_id+1), image)
        print("saved image as: bin/face_detection-{}.jpg".format(image_id+1))
        
    def cropped_image(self):
        cropped_output = []
        for id,images in enumerate(self.boxes):
            crop_img_org = self.image[images[1]:images[3], images[0]:images[2]]
            #cv2.imwrite("bin/face-{}.jpg".format(id+1), crop_img_org)
            #print("saved cropped image as: bin/face-{}.jpg".format(id+1))
            cropped_output.append(crop_img_org)
        return cropped_output


      
   
