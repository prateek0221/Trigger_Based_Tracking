import os
import time
from pathlib import Path
from memory_profiler import profile

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, \
    scale_coords, set_logging
# from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_sync
import numpy as np
import json

counter = 0
with open('/home/orange/Production/RecommendationServices/config.json', 'r') as f:
    config = json.load(f)


class FaceBlur:

    # @profile
    def __init__(self, weights=config["models"]["yoloCrowdHuman"], device=0):
        device = select_device(device)
        self.model = attempt_load(weights, map_location=device)
        self.model = self.model.half()
        self.model(torch.zeros(1, 3, 512, 640).to(
            device).type_as(next(self.model.parameters())))

    def detect(self, source, camera_no):
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
        save_img = True
        view_img = False
        imgsz = 640
        conf_thres = 0.50
        iou_thres = 0.50
        classes = None
        device = 0
        heads = False
        person = True

        """
        ROI Regions for every camera
        """
        roi_dict = {
            "D10": (190, 7, 335, 569),
            "D11": (147, 16, 353, 560),
            'D12':  (51, 4, 372, 572),
            'D14':  (216, 70, 381, 458),
            'D15': (245, 1, 281, 475),
            'D16': (171, 10, 292, 512),
            'D17':  (256, 0, 287, 554),
            'D19': (224, 41, 400, 445),
            'D20': (172, 23, 386, 465),
            'D21': (197, 3, 319, 573),
            'D22': (79, 24, 450, 501),
            'D23': (1, 59, 542, 517),
            'D24': (131, 39, 531, 511),
            'D25': (1, 93, 495, 483),
            'D26':  (91, 1, 491, 514),
            'D27': (180, 60, 540, 514),
            'D6':  (270, 214, 273, 353),
            'D7':  (224, 138, 238, 412),
            'D8':  (550, 15, 704, 343),
            "D18": (89, 33, 332, 464)

        }

        try:
            webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
                ('rtsp://', 'rtmp://', 'http://'))
        except:
            webcam = False

        # Initialize
        set_logging()
        device = select_device(device)
        half = device.type != 'cpu'  # half precision only supported on CUDA
        # print("Half: ", half)
        if half:
            self.model.half()  # to FP16
        stride = int(self.model.stride.max())
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        # Set Dataloader
        vid_path, vid_writer = None, None

        if webcam:
            view_img = False  # check_imshow()
            save_img = True
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        else:
            save_img = True
            cudnn.benchmark = True
            dataset = LoadImages(source, img_size=imgsz, stride=stride)
        names = self.model.module.names if hasattr(
            self.model, 'module') else self.model.names
        event_id = source.split("/")[-1]
        event_id = event_id.split(".")[0]
        base_path = config["productEventImgsPath"]
        folder_path = base_path + "/" + str(event_id)+"_"+str(camera_no)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        counter = 0
        t0 = time.time()
        x1, y1, x2, y2 = roi_dict[camera_no]
        start_point = (x1, y1)
        end_point = (x1+x2, y1+y2)
        color = (0, 128, 255)
        color2 = (128, 0, 255)
        thickness = 5
        personCount = 0
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            # print(img.shape)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            pred = self.model(img, augment=False)[0]
            # Apply NMS
            pred = non_max_suppression(
                pred, conf_thres, iou_thres, classes=classes, agnostic=False)
            # Process detections
            count = 0
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(
                    ), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(
                        dataset, 'frame', 0)
                blurFrame = im0.copy()
                p = Path(p)  # to Path
                s += '%gx%g ' % img.shape[2:]  # print string
                # normalization gain whwh
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                if len(det):
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        # add to string
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_img or view_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            if heads or person:
                                if "person" in label and person:
                                    person_img = im0[int(xyxy[1]):int(
                                        xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                                    filepath = folder_path + \
                                        "/"+str(counter)+".jpg"
                                    count += 1
                                    counter += 1
                                    x1, y1, x2, y2 = (int(xyxy[0]), int(
                                        xyxy[1]), int(xyxy[2]), int(xyxy[3]))
                                    start_point2 = (x1, y1)
                                    end_point2 = (x2, y2)
                                    center_point = (
                                        int((x1+x2)/2), int((y1+y2)/2))
                                    if start_point[0] <= center_point[0] <= end_point[0] and start_point[1] <= center_point[1] <= end_point[1]:
                                        cv2.imwrite(filepath, person_img)
                                       
                              
                # Stream results
                if view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(0)  # 1 millisecond
            if count > personCount:
                # print("Person Copunt Value after Incearsing: ", count)
                personCount = count
        cv2.destroyAllWindows()
        # print("Person COunt Value WHile returning: ", personCount)
        return personCount