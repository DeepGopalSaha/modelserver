import cv2
import numpy as np
import tensorflow.lite as tflite
import json


class FinalProject:
    def __init__(self, VideoBuffer, model_path: str = None, lane_model_path: str = None):
        self.VideoBuffer = VideoBuffer
        self._load_model(model_path, lane_model_path)
        #self._scaling_factor = None
        self.limit_height = 640 * 0.75  # You can make this dynamic if needed

    def _load_model(self, modelpath, lanemodelpath):
        self.interpreter = tflite.Interpreter(model_path=modelpath)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_height = self.input_details[0]['shape'][1]
        self.input_width = self.input_details[0]['shape'][2]

    def object_detection(self, frame,scale_factor: list=[-1,-1]):
        # Resize the input frame to model's expected input size
        temp_org_h,temp_org_w,_=frame.shape
        if scale_factor[0]==-1:
            temp_scale_h=temp_org_h/640
        else:
            temp_scale_h=scale_factor[0]

        if scale_factor[1]==-1:
            temp_scale_w=temp_org_w/640
        else:
            temp_scale_w=scale_factor[1]
        

        scale_to_original_factor=[temp_scale_h,temp_scale_w] #[height or y, width or x]
        resized_frame = cv2.resize(frame, (self.input_width, self.input_height), interpolation=cv2.INTER_AREA)

        normal_mask = np.zeros_like(resized_frame, dtype="uint8")
        danger_mask = normal_mask.copy()

        def _createrectangle(pt1=(0, 0), pt2=(0, 0), type=0):
            #type== 1 --> danger   ; #type== 0 --> Normal
            
            colorRed = (0, 0, 255)
            colorGreen = (0, 255, 0)
            image = danger_mask if type else normal_mask
            color = colorRed if type else colorGreen
            thickness = -1 if type else 3
            cv2.rectangle(image, pt1, pt2, color, thickness)

        # Normalize image and prepare input
        input_image = resized_frame.astype(np.float32) / 255.0
        input_image = np.expand_dims(input_image, axis=0)
        
        temp_dict = {}

        try:
            self.interpreter.set_tensor(self.input_details[0]['index'], input_image)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])[0].T

            boxes_xywh = output[..., :4]
            scores = np.max(output[..., 4:], axis=1)
            classes = np.argmax(output[..., 4:], axis=1)
            confidence_threshold = 0.25
            iou_threshold = 0.5

            indices = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), scores.tolist(), confidence_threshold, iou_threshold)


            
            temp_index = 0
            for i in indices:
                i = i[0] if isinstance(i, (np.ndarray, list)) else i
                if scores[i] >= confidence_threshold:
                    x_center, y_center, width, height = boxes_xywh[i]
                    x1 = int((x_center - width / 2)*scale_to_original_factor[1])
                    y1 = int((y_center - height / 2)*scale_to_original_factor[0])
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)


                    box_type = 1
                    if y2 < self.limit_height:
                        box_type = 0
                    

                        #_createrectangle((x1, y1), (x2, y2), 0)
                    #else:
                        #_createrectangle((x1, y1), (x2, y2), 1)
                    
                    temp_dict[temp_index] = [x1, y1, np.float64(width*scale_to_original_factor[1]), np.float64(height*scale_to_original_factor[0]), box_type]
                    temp_index += 1
                    #print(temp_dict)

        except Exception as exception_msg:
            print(exception_msg)

        # Overlay rectangles on frame
        if(temp_dict == {}):
            temp_dict = {-1:"null"}
            
        jsonString = json.dumps(temp_dict)
        #result = cv2.addWeighted(resized_frame, 1, normal_mask, 1, 0)
        #result = cv2.addWeighted(result, 1, danger_mask, 0.5, 0)
        #print(jsonString, "#")

        return jsonString

    def process_frame(self, frame,scale_factor):
        return self.object_detection(frame,scale_factor)

processor = FinalProject(VideoBuffer=None, model_path='./best.tflite')
print(1)

def process_frame_from_api(frame, scale_factor):
    print(2)
    return processor.process_frame(frame,scale_factor)

if __name__=='__main__':
    input_video_path=r"D:\D drive\001. COLLEGE _ GCELT\0. STUDY MATERIAL\Final_year_project\assets\sample_test_video_2.mp4"
    cap = cv2.VideoCapture(input_video_path)
    while (cap.isOpened()):
        # Capture each frame
        ret, frame = cap.read()
        if ret == True:
            t=frame.copy()
            boxes=json.loads(process_frame_from_api(t))
            for key in boxes:
                i=boxes[key]
                cv2.rectangle(frame, (i[0],i[1]), (int(i[0]+i[2]),int(i[1]+i[3])), (0,255,0), 3)
            cv2.imshow('APP WINDOW', frame)
            if cv2.waitKey(10) & 0xFF in [ord('q'),ord('Q')]:
                break    
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset to the beginning
        
    cap.release()
            
        