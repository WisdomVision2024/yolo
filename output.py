import torch
import numpy as np
import cv2
from time import time, sleep
from ultralytics import YOLO
import os
from collections import Counter


class ObjectDetection:

    def __init__(self):
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
        class_count = []

        # Extract detections for person class
        for result in results:
            boxes = result.boxes.cpu().numpy()
            xyxys.append(boxes.xyxy)
            confidences.append(boxes.conf)
            class_ids.append(boxes.cls)

        # Use YOLOv8's built-in plotting function to draw bounding boxes on the image
        frame_with_bboxes = results[0].plot()

        return frame_with_bboxes, xyxys, confidences, class_ids

    def process_image(self, image_path):
        frame = cv2.imread(image_path)

        if frame is None:
            print(f"Error reading image: {image_path}")
            return

        start_time = time()
        results = self.predict(frame)
        frame_with_bboxes, xyxys, confidences, class_ids = self.plot_bboxes(results, frame)

        # Print results
        print(f"Results for {os.path.basename(image_path)}:")
        image_result = results[0].verbose()     # 顯示圖片中辨識出的物品種類和數量
        print(image_result)

        json_result = results[0].tojson()       # 辨識結果json格式
        print(json_result)

        end_time = time()
        fps = 1 / np.round(end_time - start_time, 2)

        # Display the processed frame
        cv2.imshow(f'YOLOv8 Detection - {os.path.basename(image_path)}', frame_with_bboxes)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = ObjectDetection()

    while True:
        # Replace 'path_to_jpg_image' with the path to the incoming JPEG image file.
        image_path = 'images/8.jpg'  # 需辨識的jpg檔案
        detector.process_image(image_path)

        # Simulate waiting for the next image (e.g., 1 second)
        sleep(1)
