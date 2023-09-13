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
    boxes: list
        A list of boxes from YOLO pose model.
    boxes_xywh: list
        A list of YOLO box coordinates ``[x y w h]``
    boxes_xywhn: list
        A list of normalised YOLO box coordinates ``[x y w h]``
    boxes_xyxy: list
        A list of YOLO box end vertices of box ``[x0 y0 x1 y1]``
    boxes_xyxyn: list
        A list of normalised YOLO box end vertices of box ``[x0 y0 x1 y1]``
    confidence: list
        Confidence score of detection.
    detections: list
        Class of detection
        
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
        self.confidence = []
        self.detections = []
        self.class_map = None
        
        # Importing weights using YOLO
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            print(e)
                     
    def _extract_results(self, frame):
        """Extracts results.
        
        Parameters
        ----------
        frame: numpy.ndarray
            An image matrix.
            
        """
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
        
        # Adding confidence
        self.confidence.append(boxes.conf.cpu().numpy())
        
        # Adding class
        self.detections.append(boxes.cls.cpu().numpy())
        
        # Extracting class names
        self.class_map = result[0].names
        
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
                    label_font_color=(255, 255, 255),
                    label_font_scale=0.8,
                    label_font_thickness=2,
                    edge_color=(255, 255, 255), 
                    landmark_color=(255, 0, 0),
                    font=cv2.FONT_HERSHEY_SIMPLEX,
                    font_thickness=2,
                    box_color=(255, 0, 0),
                    box_thickness=2,
                    font_scale=0.2,
                    show_landmarks=True,
                    show_box=True,
                    show_label=True
                   ):
        """Renders the image.
        
        Renders the bounding box, hand pose, class label and confidence.
        Also passes un-rendered images to ``self.rendered_images``. This helps
        in maintaining the continutity of input frames so that the rendered frames
        are same as input frames.
        
        Parameters
        ----------
        font_color: tuple, default ``(0, 0, 0)``
            Font color for landmark text.
        label_font_color: tuple, default ``(255, 255, 255)``
            Font color for label
        label_font_scale: float, default ``0.8``
            Font scale for label
        label_font_thickness: int, default ``2``
            Font thickness for label.
        edge_color: tuple, default ``(255, 255, 255)`` 
            Edge color
        landmark_color: tuple, default ``(255, 0, 0)``
            Landmark color
        font: int, default ``cv2.FONT_HERSHEY_SIMPLEX``
            Font
        font_thickness: int, default ``2``
            Font thickness for landmarks.
        box_color: tuple, default ``(255, 0, 0)``
            Color of bounding box.
        box_thickness: int, default ``2``
            Thickness of bounding box.
        font_scale: float, default ``0.2``
            Font scale for landmarks.
        show_landmarks: bool, default ``True``
            Renders landmarks
        show_box: bool, default ``True``
            Renders bounding box
        show_label: bool, default ``True``
            Shows the label with confidence and detection class.
            
        """
        for idx, frame in enumerate(self.frames):
            
            # For landmark
            if not self.xy[idx][0]:
                self.rendered_images.append(frame)
                continue
                
            for xy, xyxy, conf, det in zip(self.xy[idx], self.boxes_xyxy[idx], self.confidence[idx], self.detections[idx]):
                
                if show_landmarks:
                    uv = [(np.int32(i[0]), np.int32(i[1])) for i in xy]
                    for e in self.EDGES:
                        frame = cv2.line(frame, uv[e[0]], uv[e[1]], edge_color, 2)

                    for n, landmark in enumerate(uv):
                        frame = cv2.circle(frame, landmark, 2, landmark_color, -1)
                        frame = cv2.putText(frame, 
                                            text=str(n), 
                                            org=landmark, 
                                            fontFace=font,
                                            fontScale=font_scale,
                                            color=font_color,
                                            thickness=font_thickness
                                           )
                if show_box:
                    # Unpacking
                    x0, y0, x1, y1 = xyxy
                    
                    start_point = (int(x0), int(y0))
                    end_point = (int(x1), int(y1))

                    # Bounding box
                    frame = cv2.rectangle(frame, start_point, end_point, box_color, box_thickness)
                    
                    if show_label:
                        text = str(f"{self.class_map[det]}:{conf:.2f}")
                        text_size, _ = cv2.getTextSize(text, font, label_font_scale, font_thickness)
                        text_w, text_h = text_size
                        text_end_point = (start_point[0] + text_w, start_point[1] + text_h)
                        frame = cv2.rectangle(frame, start_point, text_end_point , box_color, -1)
                        frame = cv2.putText(frame,
                                            text=text,
                                            org=(start_point[0], start_point[1]+int(text_h)),
                                            fontFace=font,
                                            fontScale=label_font_scale,
                                            color=label_font_color,
                                            thickness=label_font_thickness
                                           )
                    

            self.rendered_images.append(frame)
            
    def save(self, filename='rendered', path=".", frame_rate=30):
        """Saves the rendered images or video.
        
        Parameters
        ----------
        filename: str, default ``'rendered'``
            Name of the file to save the video.
        path: str, default ``'.'``
            Path where the vide with ``filename`` will be stored.
        frame_rate: int, default ``30``
            Frames per second in the video.
        
        """
        
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