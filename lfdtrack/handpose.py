from ultralytics import YOLO
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2


class YOLOHandPose:
    """Renders and extracts the landmarks.
    
    Parameters
    ----------
    frames: list, default ``[]``
        A list of ``numpy.ndarray`` of images.
    model_path: str, default ``'../models/freiHand0.pt'``
        File path for weights of the model.
        
    Attributes
    ----------
    results: list
        Store the results of detection.
    keypoints: list
        A list of 21 keypoints detected in image frame.
    xyn: list
        A list of normalised keypoint coordinates.
    xy: list
        A list of keypoint coordinates.
    EDGES: list
        A list of edges that connect the keypoints.
    rendered_images: list
        A list of rendered images with keypoint graph.
    model: ultralytics.yolo.engine.model.YOLO
        YOLO pose model.
        
    Methods
    -------
    process()
        Process the video or image.
    render_pose(font_color=(0, 0, 0), edge_color=(255, 255, 255), landmark_color=(255, 0, 0),font_scale=0.2)
        Renders the pose and saves the rendered images in ``rendered_images``.
    save()
        Save image/video depending on the number of images in ``frames``.
    
    """
    def __init__(self, frames=[], model_path='../models/freiHand0.pt'):
        self.frames = frames
        self.model_path = model_path
        self.results = []
        self.keypoints = []
        self.xyn = []
        self.xy = []
        self.EDGES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20]]
        self.rendered_images = []
        
        # Importing weights using YOLO
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            print(e)
                     
    def _extract_results(self, frame):
        """Extracts results"""
        # Using YOLO model to predict
        result = self.model(frame)
        # Appending results
        self.results.append(result)

        # Appending keypoints
        kpt = result[0].keypoints
        self.keypoints.append(kpt)

        # Normalised keypoints
        xyn_array = kpt.xyn.cpu().numpy()[0]
        xyn = [tuple(i) for i in xyn_array]
        self.xyn.append(xyn)

        # Keypoints
        xy_array = kpt.xy.cpu().numpy()[0]
        xy = [tuple(i) for i in xy_array]
        self.xy.append(xy)
    
    def process(self):
        """Processes the video frames.""" 
        for frame in self.frames:
            self._extract_results(frame)
            
    def render_pose(self, font_color=(0, 0, 0), edge_color=(255, 255, 255), landmark_color=(255, 0, 0),font_scale=0.2):
        """Renders the image."""
        for idx, frame in enumerate(self.frames):
            if not self.xy[idx]:
                continue
            uv = [(np.int32(i[0]), np.int32(i[1])) for i in self.xy[idx]]
            for e in self.EDGES:
                frame = cv2.line(frame, uv[e[0]], uv[e[1]], edge_color, 2)
            
            for n, landmark in enumerate(uv):
                frame = cv2.circle(frame, landmark, 2, landmark_color, -1)
                frame = cv2.putText(frame, 
                                    text=str(n), 
                                    org=landmark, 
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=font_scale,
                                    color=font_color,
                                    thickness=1
                                   )
                
            self.rendered_images.append(frame)
            
    def save(self, filename='rendered', path=".", frame_rate=30):
        """Saves the rendered images or video."""
        
        height, width, _ = self.rendered_images[0].shape
        
        file_path = os.path.join(path, filename)
        
        if len(self.frames) > 1:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Specify the codec (XVID is a common codec)
            out = cv2.VideoWriter(file_path + '.avi', fourcc, frame_rate, (width, height))

            for image in self.rendered_images:
                out.write(image)

            out.release()
            
        else:
            cv2.imwrite(file_path + '.png', self.rendered_images[0])
                