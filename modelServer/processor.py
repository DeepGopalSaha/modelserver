import cv2
import numpy as np
import tensorflow.lite as tflite


class FinalProject:
    def __init__(self, VideoBuffer, model_path: str = None, lane_model_path: str = None):
        self.VideoBuffer = VideoBuffer
        self._load_model(model_path, lane_model_path)
        self._scaling_factor = None
        self.limit_height = 640 * 0.75  # You can make this dynamic if needed

    def _load_model(self, modelpath, lanemodelpath):
        self.interpreter = tflite.Interpreter(model_path=modelpath)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_height = self.input_details[0]['shape'][1]
        self.input_width = self.input_details[0]['shape'][2]

    def object_detection(self, frame):
        # Resize the input frame to model's expected input size
        resized_frame = cv2.resize(frame, (self.input_width, self.input_height), interpolation=cv2.INTER_AREA)

        normal_mask = np.zeros_like(resized_frame, dtype="uint8")
        danger_mask = normal_mask.copy()

        def _createrectangle(pt1=(0, 0), pt2=(0, 0), type=0):
            colorRed = (0, 0, 255)
            colorGreen = (0, 255, 0)
            image = danger_mask if type else normal_mask
            color = colorRed if type else colorGreen
            thickness = -1 if type else 3
            cv2.rectangle(image, pt1, pt2, color, thickness)

        # Normalize image and prepare input
        input_image = resized_frame.astype(np.float32) / 255.0
        input_image = np.expand_dims(input_image, axis=0)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_image)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0].T

        boxes_xywh = output[..., :4]
        scores = np.max(output[..., 4:], axis=1)
        classes = np.argmax(output[..., 4:], axis=1)
        confidence_threshold = 0.25
        iou_threshold = 0.5

        indices = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), scores.tolist(), confidence_threshold, iou_threshold)

        for i in indices:
            i = i[0] if isinstance(i, (np.ndarray, list)) else i
            if scores[i] >= confidence_threshold:
                x_center, y_center, width, height = boxes_xywh[i]
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)

                if y2 < self.limit_height:
                    _createrectangle((x1, y1), (x2, y2), 0)
                else:
                    _createrectangle((x1, y1), (x2, y2), 1)

        # Overlay rectangles on frame
        result = cv2.addWeighted(resized_frame, 1, normal_mask, 1, 0)
        result = cv2.addWeighted(result, 1, danger_mask, 0.5, 0)

        return result

    def process_frame(self, frame):
        return self.object_detection(frame)


def process_frame_from_api(frame, model_path: str):
    processor = FinalProject(VideoBuffer=None, model_path=model_path)
    return processor.process_frame(frame)
