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
        self.boxes = []
        self.boxes_xywh = []
        self.boxes_xywhn = []
        self.boxes_xyxy = []
        self.boxes_xyxyn = []
        
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
        kpts = result[0].keypoints
        self.keypoints.append(kpts)
        
        # Appending boxes
        boxes = result[0].boxes
        self.boxes.append(boxes)
        
        # Appending anchor coordinates detected in the image frame to a list
        self.boxes_xywh.append(result[0].boxes.xywh.cpu().numpy())
        self.boxes_xywhn.append(result[0].boxes.xywhn.cpu().numpy())

        # Appending box vertices to the list
        self.boxes_xyxy.append(result[0].boxes.xyxy.cpu().numpy())
        self.boxes_xyxyn.append(result[0].boxes.xyxyn.cpu().numpy())

        # Normalised keypoints
        xyn_array = result[0].keypoints.xyn.cpu().numpy()
        xyn_temp = []
        for i in xyn_array:
            xyn = [tuple(j) for j in i]
            xyn_temp.append(xyn)
        self.xyn.append(xyn_temp)

        # Keypoints
        xy_array = result[0].keypoints.xy.cpu().numpy()
        xy_temp = []
        for i in xy_array:
            xy = [tuple(j) for j in i]
            xy_temp.append(xy)
        self.xy.append(xy_temp)
    
    def process(self):
        """Processes the video frames.""" 
        for frame in self.frames:
            self._extract_results(frame)
            
    def render_pose(self, 
                    font_color=(0, 0, 0), 
                    edge_color=(255, 255, 255), 
                    landmark_color=(255, 0, 0),
                    font_scale=0.2,
                    show_landmarks=True,
                    show_box=True,
                    show_confidence=True,
                    show_label=True
                   ):
        """Renders the image."""
        for idx, frame in enumerate(self.frames):
            if not self.xy[idx][0]:
                continue
                
            for xy in self.xy[idx]:
                uv = [(np.int32(i[0]), np.int32(i[1])) for i in xy]
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

def flip_pose(poses, T):
    """Flips the pose as per transformation matrix T.
    
    Parameters
    ----------
    poses: list
        A list of pose detected from the YOLOHandPose
    T: numpy.ndarray
        A transformation matrix for flipped.
    
    Returns
    -------
    list
        A list of transformed coordinates.
        
    """
    flipped_lmks = []

    for pose in poses:
        if pose:
            lmks = []
            for lmk in pose:
                x, y = lmk
                flip_lmk = np.dot([x,y,1], T)
                fx, fy, _ = flip_lmk
                lmks.append((fx, fy))
            flipped_lmks.append(lmks)
        else:
            flipped_lmks.append([])
    
    return flipped_lmks