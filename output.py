import torch
import numpy as np
import cv2
import io
from time import time
from ultralytics import YOLO

def bytes_to_image(byte_array):
    # Convert byte array to a numpy array
    nparr = np.frombuffer(byte_array, np.uint8)
    # Decode numpy array to an image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

class ObjectDetection:

    def __init__(self, capture_index=None):
        self.capture_index = capture_index
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names

    def load_model(self):
        model = YOLO("Total.pt")  # load a pretrained YOLOv8 model
        model.fuse()
        return model

    def predict(self, frame):
        results = self.model(frame)
        return results

    def plot_bboxes(self, results, frame):
        xyxys = []
        confidences = []
        class_ids = []

        # Extract detections for person class
        for result in results:
            boxes = result.boxes.cpu().numpy()
            xyxys.append(boxes.xyxy)
            confidences.append(boxes.conf)
            class_ids.append(boxes.cls)

        # Use YOLOv8's built-in plotting function to draw bounding boxes on the image
        frame_with_bboxes = results[0].plot()

        return frame_with_bboxes, xyxys, confidences, class_ids

    def process_image_bytes(self, byte_array):
        # Convert byte array to image
        frame = bytes_to_image(byte_array)
        # Make prediction
        results = self.predict(frame)
        # Plot bounding boxes
        frame_with_bboxes, xyxys, confidences, class_ids = self.plot_bboxes(results, frame)
        return frame_with_bboxes, xyxys, confidences, class_ids

    def __call__(self):
        if self.capture_index is not None:
            cap = cv2.VideoCapture(self.capture_index)
            assert cap.isOpened()

            while True:
                start_time = time()
                ret, frame = cap.read()
                assert ret
                results = self.predict(frame)
                frame_with_bboxes, xyxys, confidences, class_ids = self.plot_bboxes(results, frame)
                end_time = time()
                fps = 1 / np.round(end_time - start_time, 2)
                cv2.imshow('YOLOv8 Detection', frame_with_bboxes)

                if cv2.waitKey(5) & 0xFF == 27:
                    break

            cap.release()
            cv2.destroyAllWindows()

detector = ObjectDetection()
detector()