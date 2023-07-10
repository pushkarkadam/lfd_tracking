import numpy as np 
import cv2 
import sys 
import os
import xml.etree.ElementTree as ET 


class Annotation:
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