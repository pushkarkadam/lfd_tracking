import numpy as np 
import cv2 
import sys 
import os
import xml.etree.ElementTree as ET 
import re
import json
import random
import matplotlib.pyplot as plt


class VOC:
    def __init__(self,
                 dir_path,
                 images_path,
                 object_name="hand",
                 img_ext='.jpg'
                ):
        """Extracts the annotation of VOC dataset into YOLO format.

        Parameters
        ----------
        dir_path: str
            The path to the directory where the annotation data is stored.
            **Example**: ``dir_path='~/Documents/data/VOC2010/Annotations/'``
        images_path: str
            The path of the directory where the image data is stored
            **Example**: ``dir_path='~/Documents/data/VOC2010/JPEGImages/'``
        object_name: str, default ``'hand'``
            The object in the xml file that should be searched i.e. hand.
        img_ext: str, default ``'.jpg'``
            The extension of the image from the directory that should be read.
            The VOC dataset has all the images as ``.jpg`` format.
        
        Attributes
        ----------
        roots: list
            A list of the annotations from its root structure.
        bboxes: dict
            A dictionary that maps the image name to bounding boxes.
        objects_bboxes: dict
            A dictionary that maps the object bounding box.
        yolo_annot: dict
            A dictionary that stores yolo annotation.
            The keys are the element from ``file_names`` and the value is a list.
            The data format for pose is a list: ``[class x y w h]``
            The total number of element in yolo annotation list are 5.
        image_shape: dict
            A dictionary that maps ``file_names`` to the tuple of image shape.

        """
        
        self.dir_path = dir_path
        self.images_path = images_path
        self.object_name = object_name
        self.img_ext = img_ext
        
        self.roots = []
        
        self.bboxes = dict()
        
        self.objects_bboxes = dict()
        
        self.files = []
        
        self.yolo_annot = dict()
        
        self.object_images = dict()
        
        self.image_size = dict()
        
    
    def _get_xml_root(self, file_path):
        """Gets the root for the XML object"""
        
        """Gets the root of the file"""
        
        tree = ET.parse(file_path)
        
        root = tree.getroot()
        
        self.roots.append(root)
        
    def _get_roots(self):
        """Gets the annotation directory path and collects the root from each"""
        
        with os.scandir(self.dir_path) as entries:
            for entry in entries:
                file, extension = os.path.splitext(entry.name)
                self.files.append(file)
        
        for file in self.files:
            self._get_xml_root(os.path.join(self.dir_path, file+".xml"))
            
    def _get_bboxes(self):
        """Gets bounding boxes"""
        
        self._get_roots()
        
        for root, file in zip(self.roots, self.files):
            root_filename = file
            print(f"Now processing root: {root_filename}")
            self.bboxes[root_filename] = []
            for x in root.findall('object'):
                for y in x.findall('part'):
                    name = y.find('name').text
                    if name == self.object_name:
                        bb = y.find('bndbox')

                        xmin = int(bb.find('xmin').text)
                        ymin = int(bb.find('ymin').text)
                        xmax = int(bb.find('xmax').text)
                        ymax = int(bb.find('ymax').text)

                        self.bboxes[root_filename].append([xmin, ymin, xmax, ymax])
    
    def _remove_empty_bboxes(self):
        """Removes all the empty lists"""
        
        for k, v in self.bboxes.items():
            if v:
                self.objects_bboxes[k] = v
                
                
    def _read_object_images(self):
        """Takes all the images with hands and stores them in a list"""
        for k in self.objects_bboxes.keys():
            img_path = os.path.join(self.images_path, k + self.img_ext)
            print(f"Reading: {img_path}")
            img = cv2.imread(img_path)
            self.object_images[k] = img
    
    def _get_image_size(self):
        """Gets image size"""
        
        self._get_roots()
        
        for root, file in zip(self.roots, self.files):
            root_filename = file
            print(f"Now processing root: {root_filename}")
            self.bboxes[root_filename] = []
            for x in root.findall('size'):
                img_size = int(x.find('width').text), int(x.find('height').text)
                
                self.image_size[root_filename] = img_size
                
    
    def _get_yolo_coords(self):
        """Gets the YOLO type coordinates for every image"""
        
        for k, v in self.objects_bboxes.items():
            W, H = self.image_size[k]
            
            self.yolo_annot[k] = []
            
            # iterating over all bounding boxes
            for c in v:
                xmin, ymin, xmax, ymax = c
                
                yolo_coord = [0, 
                              (xmin + (xmax - xmin)/2)/W, 
                              (ymin + (ymax - ymin)/2)/H, 
                              (xmax - xmin)/W,
                              (ymax - ymin)/H
                             ]
                self.yolo_annot[k].append(yolo_coord)
    
    def run(self):
        """Runs the annotation tasks"""
        
        self._get_bboxes()
        
        self._remove_empty_bboxes()
        
        self._read_object_images()
        
        self._get_image_size()
        
        self._get_yolo_coords()
        
    def save(self, save_path='.', directory="train", img_ext='.jpg', label_ext='.txt'):
        """Saves the images and labels.
        
        Parameters
        ----------
        save_path: str, default ``'.'``
            The path where the new dataset must be saved.
        directory: str, default ``'train'``
            The name of the directory which will contain ``images`` and ``labels`` subdirectory.
        img_ext: str, default ``'.jpg'``
            The image format to store the image.
        label_ext: str, default ``'.txt'``
            The extension of the text file where the data for YOLO will be stored.

        """
        
        path = os.path.join(save_path, directory)
        
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"New directory {directory} created at {path}")
            
        for k, v in self.object_images.items():
            image_path = os.path.join(path, 'images')
            
            # Creates image directory if not already present
            if not os.path.exists(image_path):
                os.makedirs(image_path)
            
            # Image file path with extension
            image_file = os.path.join(image_path, k + img_ext)
            
            # Save image
            cv2.imwrite(image_file, v)
            
            # Labels
            labels_path = os.path.join(path, 'labels')
            
            # Checks if there is a label directory
            if not os.path.exists(labels_path):
                os.makedirs(labels_path)
            
            # Label file
            label_file = os.path.join(labels_path, k + label_ext)
            
            labels = self.yolo_annot[k]
            
            with open(label_file, 'w') as file:
                for label in labels:
                    for c in label:
                        file.write('%s' % c)
                        file.write(' ')
                    file.write('\n')

class Panoptic:
    """Panoptic class that takes annotation pickle file.
        
        Attributes
        ----------
        data: dict
            A dictionary of data stored as a json file.
            Ideally, to work in Jupyter notebook load json file as a dict in ipython console.
            Save it as a pickle file and then import it before adding as an attribute to the object
            of this class

        yolo_annot: dict
            A dictionary that stores yolo annotation.
            The keys are the element from ``file_names`` and the value is a list.
            The data format for pose is a list: ``[class x y w h]``
            The total number of element in yolo annotation list are 5.
        yolo_pose: dict
            A dictionary that stores the yolo pose data.
            The keys are the element from ``file_names`` and the value is a list.
            The data format for pose is a list: ``[class x y w h px0 py0 px1 py1 ... px20 py21]``
            The total number of elements in the yolo pose list is (5 + 2 * 21) = 47 elements (5 yolo and 21 (x,y) landmark)
        image_shape: dict
            A dictionary that maps ``file_names`` to the tuple of image shape.
        bounding_boxes: dict
            A dictionary that maps ``file_names`` to bounding box coordinates.
        images:
            A dictionary that maps the ``file_names`` to images.

        Methods
        -------
        get_images(images_path, image_extension='.jpg', verbose=False)
            Gets all the images from the ``images_path`` 
        run()
            Runs all the methods to populate the attributes
        save(save_path='.', directory='labels', label_extension='.txt')
            Save the labels
        save_as_pickle(file_path)
            Save the object as a pickle file.

        """
    def __init__(self, data):
        self.data = data['root']        
        self.yolo_annot = dict()
        self.yolo_pose = dict()
        self.image_shape = dict()
        self.bounding_boxes = dict()
        self.images = dict()
        
    def _get_joints(self):
        """Gets x and y coordinates"""
        
        for d in self.data:
            landmarks = d['joint_self']
            
            x = []
            y = []
            
            pose = []
            
            W = d['img_width']
            H = d['img_height']
            
            for landmark in landmarks:
                x.append(landmark[0])
                y.append(landmark[1])
                
                # Appending pose coordinates 
                pose.append(landmark[0]/W)
                pose.append(landmark[1]/H)
                
            xmin = np.min(x)
            ymin = np.min(y)
            xmax = np.max(x)
            ymax = np.max(y)
            
            yolo_coord = [0, 
                          (xmin + (xmax - xmin)/2)/W, 
                          (ymin + (ymax - ymin)/2)/H, 
                          (xmax - xmin)/W,
                          (ymax - ymin)/H
                         ]
            
            pose_coord = yolo_coord + pose
            
            filename = re.split('\W+', d['img_paths'])
            
            self.yolo_annot[filename[1]] = yolo_coord
            self.yolo_pose[filename[1]] = pose_coord
            
            # Storing image shape 
            self.image_shape[filename[1]] = (H, W)
            
            # Bounding boxes | format: [xmin, ymin, xmax, ymax]
            self.bounding_boxes[filename[1]] = [xmin, ymin, xmax, ymax]
            
            
    def get_images(self, images_path, image_extension='.jpg', verbose=False):
        """Gets the images from the directory.
        Use this function only if necessary, because if the data consists of large
        number of images, then the jupyter notebook will likely crash.
        
        Parameters
        ----------
        images_path: str
            The file path where the images are stored.
        image_extension: str, default ``.jpg``
            The extension of the file in the directory to read.
        verbose: bool, default ``False``
            Shows the message in the console about the processing steps.
        """
        filename = list(self.image_shape.keys())
        
        for f in filename:
            image_path = os.path.join(images_path, f + self.image_extension)
            if self.verbose:
                print(f"Reading: {image_path}")
                
            image = cv2.imread(image_path)
            self.images[f] = image
    
    def run(self):
        """Runs the Panoptic model"""
        
        self._get_joints()
        
    def save(self, save_path='.', directory='labels', label_extension='.txt', annotations=dict()):
        """Saves the labels.
        
        Parameters
        ----------
        save_path: str, default ``'.'``
            The path where the result directory and files are to be saved.
        directory: str, default ``'labels'``
            The directory name under which the labels text files are to be saved.
        label_extension: str, default ``'.txt'``
            The extension with which to save the yolo annotation in a text file.
        annotations: dict, default ``dict()``
            A dictionary of annotations.
        
        """
        
        labels_path = os.path.join(save_path, directory)
        
        if not os.path.exists(labels_path):
            os.makedirs(labels_path)
            print(f"New directory {directory} created at {labels_path}")
            
        for k, v in annotations.items():
            label_file = os.path.join(labels_path, k + label_extension)
            
            yolo_annotation = annotations[k]
            
            with open(label_file, 'w') as file:
                for c in yolo_annotation:
                    file.write('%s' % c)
                    file.write(' ')
                     
    def save_as_pickle(self, filepath='.', filename='pan_yolo.pickle'):
        """Save the object as a pickle file
        
        Parameters
        ----------
        filepath: str, default ``'.'``
            The filepath where to store the object
        filename: str, default ``'pan_yolo.pickle'``
            The filename to store pickle file
        """
        
        file_path = os.path.join(filepath, filename)
        
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

class PanopticManual(Panoptic):
    """Manual annotation to YOLO conversion,
    
    Parameters
    ----------
    data_path: str
        Path where the root directory for data is located.
    image_extension: str, default ``'.jpg'``
        Image extension
    
    Attributes
    ----------
    data_path: str
        The root path for the directory where the images and json files are stored.
    image_extension: str
        The extension of the image files in the data directory
    files: list
        A list of the data files that end with json extension.
    file_names: list
        A list of str file name without the file extension.
        This is helpful as the image and data files have same names but different extension.
    data: dict
        A dictionary that stores the data for all the images together.
        The key is the element from ``file_names`` and the value is a dict with data from json files.
    yolo_pose: dict
        A dictionary that stores the yolo pose data.
        The keys are the element from ``file_names`` and the value is a list.
        The data format for pose is a list: ``[class x y w h px0 py0 px1 py1 ... px20 py21]``
        The total number of elements in the yolo pose list is (5 + 2 * 21) = 47 elements (5 yolo and 21 (x,y) landmark)
    yolo_annot: dict
        A dictionary that stores yolo annotation.
        The keys are the element from ``file_names`` and the value is a list.
        The data format for pose is a list: ``[class x y w h]``
        The total number of element in yolo annotation list are 5.
    image_shape: dict
        A dictionary that maps ``file_names`` to the tuple of image shape.
    bounding_boxes: dict
        A dictionary that maps ``file_names`` to bounding box coordinates.
    EDGES: list
        A list that is contains the edges of the graph of hand landmark.
    sample_files: list
        A list of filenames with str values.
        
    Methods
    -------
    read_write_images(save_path='.', directory='images')
        Reads the image from the ``data_path`` and stores them in ``directory`` of the ``save_path``.
    bbox_from_yolo(dims, coords)
        Converts YOLO coordinates to bounding box.
    read_images(files)
        Returns a list of images from the file names mentioned in a list of ``files``.
    bounding_box_check(save=False, filename="results.png")
        Performs a sanity check by providing qualitative results to bounding box from YOLO coordinates.
        This is essential step to visually check if the yolo coordinates can properly represent the object.
    hand_landmark_check(save=False, filename='landmark_results.png')
        Performs a sanity check by providing qualitative results to hand landmark from YOLO pose annotations.
        This is essential to visually check if the yolo pose annotations are correct.
    
    """
    
    def __init__(self, data_path, image_extension='.jpg', seed=0, invert_coords=False):
        self.data_path = data_path
        
        self.image_extension = image_extension
        
        # Setting the seed for random sampling
        self.seed = seed
        
        self.invert_coords = invert_coords
        
        self.files = sorted([f for f in os.listdir(data_path) if f.endswith('.json')])
        
        self.file_names = [re.split('\W+', f)[0] for f in self.files]
        
        self.data = dict()
        
        self.yolo_pose = dict()
        
        self.yolo_annot = dict()
        
        self.image_shape = dict()
        
        self.bounding_boxes = dict()
        
        self.sample_files = []
        
        self.EDGES = [[0,1], [1,2], [2,3], [3,4], 
                      [0,5], [5,6], [6,7], [7,8], 
                      [0,9], [9,10],[10,11], [11,12],
                      [0,13],[13,14], [14,15], [15,16], 
                      [0,17],[17,18], [18,19], [19,20]]
        
        for f, fn in zip(self.files, self.file_names):
            with open(os.path.join(data_path, f), 'r') as fid:
                self.data[fn] = json.load(fid)
            
        
        for fn in self.file_names:
            image_name = fn + image_extension
            try:
                image = cv2.imread(os.path.join(data_path, image_name), 0)

                H, W = image.shape

                landmarks = self.data[fn]['hand_pts']

                x = []
                y = []

                pose = []

                for landmark in landmarks:
                    x_n = landmark[0] / W
                    y_n = landmark[1] / H

                    x.append(x_n)
                    y.append(y_n)

                    pose.append(x_n)
                    pose.append(y_n)

                # Selecting the boundary coordinates for bounding box
                xmin = np.min(x)
                ymin = np.min(y)
                xmax = np.max(x)
                ymax = np.max(y)

                yolo_coord = [0, 
                              (xmin + (xmax - xmin)/2), 
                              (ymin + (ymax - ymin)/2), 
                              (xmax - xmin),
                              (ymax - ymin)
                             ]

                pose_coord = yolo_coord + pose

                self.yolo_annot[fn] = yolo_coord
                self.yolo_pose[fn] = pose_coord

                self.image_shape[fn] = (H, W)

                self.bounding_boxes[fn] = [xmin, ymin, xmax, ymax]
            except:
                continue
                
    def read_write_images(self, save_path='.', directory='images'):
        """Reads the image from the directory and writes them.
        
        Parameters
        ----------
        save_path: str, default ``'.'``
            The path where the result directory and files are to be saved.
        directory: str, default ``'labels'``
            The directory name under which the labels text files are to be saved.
        
        """
        image_dir_path = os.path.join(save_path, directory)
        
        if not os.path.exists(image_dir_path):
            os.makedirs(image_dir_path)
            print(f"New directory {directory} created at {image_dir_path}")
        
        for fn in self.file_names:
            image_path = os.path.join(self.data_path, fn + self.image_extension)
            image_save_path = os.path.join(image_dir_path, fn + self.image_extension)
            print(image_save_path)
            
            try:
                image = cv2.imread(image_path)
                cv2.imwrite(image_save_path, image)
            except:
                continue
                   
    def bbox_from_yolo(self, dims, coords):
        """Returns the bounding box vertices.
        
        Parameters
        ----------
        dims: list
            The x, y, w, h coordinates as a list.
        coords: tuple
            The height and width of the image (row, col).

        Returns
        -------
        list:
            A list of vertices.
        """
    
        x, y, w, h = dims
        H, W = coords

        X = x * W
        Y = y * H
        Wb = w * W
        Hb = h * H

        xmin = X - Wb/2
        ymin = Y - Hb/2
        xmax = X + Wb/2
        ymax = Y + Hb/2

        verts = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]

        return verts
    
    def read_images(self, files):
        """Returns a list of image file from list of files.
        
        Parameters
        ----------
        files: list
            A list of file names
        
        """ 
        images = dict()
        
        for idx, file in enumerate(files):
            image_path = os.path.join(self.data_path, file + self.image_extension)
            img = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images[file] = img_rgb
            
        return images
    
    def bounding_box_check(self, save=False, filename="results.png", sample_size=9):
        """Checks the bounding box by showing samples of the implementation.
        
        Parameters
        ----------
        save: bool, default ``False``
            Saves the images
        filename: str, default ``'results.png'``
            Filename to be saved.
        sample_size: int, default ``9``
            Sample size to select
        
        """
        vertices = []
        sample_yolo_annot = []
        sample_image_shapes = []
        
        if not self.sample_files:
            if sample_size > 9:
                sample_size = 9
            self.sample_files = random.choices(list(self.yolo_pose.keys()), k=sample_size)
        
        for f in self.sample_files:
            sample_yolo_annot.append(self.yolo_pose[f][1:5])
            sample_image_shapes.append(self.image_shape[f])
            
        for f, a, s in zip(self.sample_files, sample_yolo_annot, sample_image_shapes):
            vertices.append(self.bbox_from_yolo(a, s))
            
            
        vertices_np = [np.array(v, dtype=np.int32).reshape((4,2)) for v in vertices]
        
        # Reads all the images from a list of files
        images = self.read_images(self.sample_files)
                
        fig = plt.figure(figsize=(10, 10))

        for i in range(len(self.sample_files)):
            ax = fig.add_subplot(3, 3, i+1)
            cv2.polylines(images[self.sample_files[i]], [vertices_np[i]], isClosed=True, color=(255,0,0), thickness=5)
            plt.imshow(images[self.sample_files[i]])
            plt.title(self.sample_files[i])
            plt.axis('off')
        
        if save:
            fig.savefig(filename)
                   
    def hand_landmark_check(self, save=False, filename='landmark_results.png', sample_size=9):
        """Checks the hand landmark by selecting samples.
        
        Parameters
        ----------
        save: bool, default ``False``
            Saves the images.
        filename: str, default ``'landmark_results.png'``
            Filename to be saved.
        sample_size: int, default ``9``
            Sample size to select 
            
        """
        if not self.sample_files:
            if sample_size > 9:
                sample_size = 9
            self.sample_files = random.sample(list(self.yolo_pose.keys()), k=sample_size)
        
        uv = dict()
        
        for f in self.sample_files:
            landmarks = self.yolo_pose[f][5:]
            uv[f] = []
            
            i = 0
            while i < int(len(landmarks)):
                H, W = self.image_shape[f]
                scaling = np.multiply(landmarks[i:i+2], list((W, H)))
                uv[f].append(np.array(scaling, np.int32))
                i+=2
            
            uv[f] = np.array(uv[f])
                
        images = self.read_images(self.sample_files)

        rendered_images = dict()
        
        for f in self.sample_files:
            rendered_images[f] = []
            for c in self.EDGES:
                rendered_image = cv2.line(images[f], uv[f][c[0]], uv[f][c[1]], (255, 0, 0), 2)
                rendered_images[f] = rendered_image
                
                
        for f in self.sample_files:
            for point in uv[f]:
                rendered_images[f] = cv2.circle(rendered_images[f], point, 2, (0, 0, 255), -1)
        
        fig = plt.figure(figsize=(10, 10))
        
        for idx, f in enumerate(self.sample_files):
            ax = fig.add_subplot(3,3, idx+1)
            plt.imshow(rendered_images[f])
            plt.title(f)
            plt.axis('off')
            
        if save:
            fig.savefig(filename)

class MergeHands(PanopticManual):
    """Merges hand data.
    
    The data is currently stored in a YOLO format as images and labels subdirectory.
    This class reads the data for right and left hand and merges them into a single
    label file and also writes the image as one image for both the hand labels.
    
    Parameters
    ----------
    data_path: str
        Data path where the ``images`` and ``labels`` is stored.
        
    Attributes
    ----------
    sub_dir: list, default ``['images', 'labels']``
        A list of sub directories.
    paths: list
        A list of all the file paths with the subdirectories.
    yolo_annot: dict
        A dictionary that stores the YOLO annotation.
    annotation_files: list
        A list of the annotation files
        
    Methods
    -------
    combine_annotations(separate_hands=True)
        Combines the annotation for the hands in the image.
    
    """
    
    def __init__(self, data_path, image_extension='.jpg', label_extension='.txt'):
        self.data_path = data_path
        
        self.image_extension = image_extension
        
        self.label_extension = label_extension
        
        self.sub_dir = ['images', 'labels']
        
        self.paths = []
        
        self.yolo_annot = dict()
        
        # Getting all the data paths
        for s in self.sub_dir:
            self.paths.append(os.path.join(data_path, s))
                  
    def combine_annotations(self, separate_hands=True):
        """Combines the annotations.

        Parameters
        ----------
        separate_hands: bool, default ``True``
            Separates the hands as left and right hand.
            ``left_hand_label = 0``
            ``right_hand_label = 1``
        
        """
        
        # Assigning the labels path
        labels_path = self.paths[1]
        
        # Reading the labels to get the root of the names
        self.annotation_files = os.listdir(labels_path)

        # Separate extension
        self.label_files = [os.path.splitext(f)[0] for f in self.annotation_files]

        # Adding empty list to yolo annotation dict
        for label in self.label_files:
            self.yolo_annot[label[:-2]] = []

        # Reading text files and assigning labels
        for file, label_file in zip(self.annotation_files, self.label_files):
            with open(os.path.join(labels_path, file), 'r') as f:
                line = f.readline().strip()
                annot = [float(num) for num in line.split()]

                # Checking if the hands are to be separated as left and right hands
                if separate_hands:
                    # Changing annotation class of right hand to 1
                    if label_file[-1:] == 'r':
                        annot[0] = 1.0

                # Extracting the filename without extension
                filename = label_file[:-2]

                # Appending the annotation of both right and left hands to a single key
                # of same filename
                self.yolo_annot[filename].append(annot)
                
    def combine_images(self, save_path='.', directory='combined_images'):
        """Combines the image to a single file"""
        
        # Assigning the file path
        images_path = self.paths[0]
        
        all_files = list(self.yolo_annot.keys())
        
        images_save_path = os.path.join(save_path, directory)
             
        for file in all_files:
            # Assigning original left hand annotation images to image_file
            image_file = os.path.join(images_path, file + '_l' + self.image_extension)

            # Checking if the image exists in the directory
            # if the image doesn't exist, we try to see if right hand image exists
            if not os.path.exists(image_file):
                image_file = os.path.join(images_path, file + '_r' + self.image_extension)
            
            try:
                image = cv2.imread(image_file)
            except:
                continue
            
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                print(f"New directory {directory} created at {labels_path}")
            
            cv2.imwrite(os.path.join(images_save_path, file + self.image_extension), image)
            
    def save_annotations(self, save_path='.', directory='combined_labels'):
        """Saves the annotations"""
        
        labels_path = os.path.join(save_path, directory)
        
        if not os.path.exists(labels_path):
            os.makedirs(labels_path)
            print(f"New directory {directory} created at {labels_path}")
            
        if self.yolo_annot:
            for k, v in self.yolo_annot.items():
                label_file = os.path.join(labels_path, k + self.label_extension)

                yolo_annotation = self.yolo_annot[k]

                with open(label_file, 'w') as file:
                    for label in yolo_annotation:
                        for c in label:
                            file.write('%s' % c)
                            file.write(' ')
                        file.write('\n')