import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO
import os

class ObjectDetection:

    def __init__(self, image_folder_path):
        self.image_folder_path = image_folder_path
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

    def process_images(self):
        # Iterate over all files in the provided directory
        for filename in os.listdir(self.image_folder_path):
            if filename.lower().endswith('.jpg'):
                image_path = os.path.join(self.image_folder_path, filename)
                frame = cv2.imread(image_path)

                if frame is None:
                    print(f"Error reading image: {image_path}")
                    continue

                start_time = time()
                results = self.predict(frame)
                frame_with_bboxes, xyxys, confidences, class_ids = self.plot_bboxes(results, frame)
                end_time = time()
                fps = 1 / np.round(end_time - start_time, 2)

                # Uncomment to show FPS on the frame
                # cv2.putText(frame_with_bboxes, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

                # Display the processed frame
                cv2.imshow(f'YOLOv8 Detection - {filename}', frame_with_bboxes)

                # Wait for a key event to close the image display
                cv2.waitKey(3000)
                cv2.destroyAllWindows()

if __name__ == "__main__":
    image_folder_path = 'path_for_images'  # Update this to your folder path
    detector = ObjectDetection(image_folder_path=image_folder_path)
    detector.process_images()
