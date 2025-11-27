import cv2
import os
from random import randint
import json

with open('/home/orange/Production/RecommendationServices/config.json', 'r') as f:
    config = json.load(f)

class PersonDetection():
    def __init__(self):
        self.text_weights = config["models"]["yolov7Weights"]
        self.text_netcfg = config["models"]["yolov7cfg"]
        self.model = cv2.dnn_DetectionModel(self.text_weights, self.text_netcfg)
        self.model.setInputSize(416, 416)
        self.model.setInputScale(1.0 / 255)
        self.model.setInputSwapRB(True)
        self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print("Loaded Person Detection Model")

    def random_with_N_digits(self, n):
        range_start = 10**(n-1)
        range_end = (10**n)-1
        return randint(range_start, range_end)
    
    def crop_image(self, image, box):
        x, y, w, h = box
        y_start = max(0, y)
        y_end = min(image.shape[0], y+h)
        x_start = max(0, x)
        x_end = min(image.shape[1], x+w)
        return image[y_start:y_end, x_start:x_end]
    
    def person_detector(self, video_path, camera_no):
        """ Detect the Head of person in the video, Blur the video, trim images from video
        and return the video with pixelated head.
        Parameters:
        ----------
        source : string
                Input Video Path 
        output_path: string
                Output Video Path 
        camera_no: string
                Camera no that is used to trim the images to that folder 

        Returns:
        -------
        Person Count
        """
        print(f"Processing {video_path}")
        cap = cv2.VideoCapture(video_path)

        base_video_name = os.path.splitext(os.path.basename(video_path))[0]
        event_id = video_path.split("/")[-1]
        event_id = event_id.split(".")[0]
        base_path = config["productEventImgsPath"]
        folder_path = base_path + "/" + str(event_id)+"_"+str(camera_no)
        os.makedirs(folder_path, exist_ok=True)
        frame_count = 0
        personCount = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            person_counter = 0
            classes, scores, boxes = self.model.detect(frame, confThreshold=0.8, nmsThreshold=0.8)
            if classes is not None and len(classes) > 0:
                 for (class_id, confidence, box) in zip(classes.flatten(), scores.flatten(), boxes):
                    if class_id == 0:
                        person_counter += 1  
                        cropped_frame = self.crop_image(frame, box)
                        frame_filename = f"{self.random_with_N_digits(10)}.jpg"
                        save_frame_path = os.path.join(folder_path, frame_filename)
                        cv2.imwrite(save_frame_path, cropped_frame)
                        # print(f"Saved cropped person image with random filename {frame_filename} in {video_save_directory}"
            if person_counter > personCount:
                personCount = person_counter
             

        cap.release()
        return personCount


