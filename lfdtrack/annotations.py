import numpy as np 
import cv2 
import sys 
import os
import xml.etree.ElementTree as ET 
import re


class VOC:
    def __init__(self,
                 dir_path,
                 images_path,
                 object_name="hand",
                 img_ext='.jpg'
                ):
        """Extracts the annotation of VOC dataset into YOLO format.

        Attributes
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