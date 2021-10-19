# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager
from detectron2.data.datasets import register_coco_instances, register_coco_panoptic
from detectron2.data import MetadataCatalog   

from .coco import load_coco_json, load_sem_seg

__all__ = ["fire_panoptic_train", "fire_panoptic_val"]


import json
import os
import numpy as np
import cv2

# import some common libraries
import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import json
import os
import random
from pycocotools.coco import COCO
import cv2
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.model_zoo import model_zoo
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine import DefaultTrainer, DefaultPredictor, launch
from detectron2.config import get_cfg
from google.colab.patches import cv2_imshow



# def generate_segmentation_file(img_dir):
#     json_file = os.path.join(img_dir, "FireClassification.json")
#     # with open(json_file) as f:
#     #     imgs_anns = json.load(f)
#     # print(imgs_anns)
#     coco=COCO(json_file)
#     #print(coco)
#     cats = coco.loadCats(coco.getCatIds())
#     nms=[cat['name'] for cat in cats]
#     print('COCO categories: \n{}\n'.format(' '.join(nms)))
#     imgIds_1 = coco.getImgIds()
#     print(imgIds_1)
#     for i in imgIds_1:
#         imgIds = coco.getImgIds(imgIds = i) ##Image id part in the json
#         img = coco.loadImgs(imgIds)[0]
#         #print(img)
#         annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
#         anns = coco.loadAnns(annIds)
#         mask = coco.annToMask(anns[0])
#         for i in range(len(anns)):
#           mask += coco.annToMask(anns[i])
#         file_name, ext = img["file_name"].split(".")
#         print(file_name)
#         output = os.path.join('/content/data/Fire/train/', "segmentation", file_name+".png")
#         cv2.imwrite(output, mask)





def register_fire_panoptic(root):

    print("Inside Fire train------")

    """
    Register a "standard" version of COCO panoptic segmentation dataset named `name`.
    The dictionaries in this registered dataset follows detectron2's standard format.
    Hence it's called "standard".

    Args:
        name (str): the name that identifies a dataset,
            e.g. "coco_2017_train_panoptic"
        metadata (dict): extra metadata associated with this dataset.
        image_root (str): directory which contains all the images
        panoptic_root (str): directory which contains panoptic annotation images in COCO format
        panoptic_json (str): path to the json panoptic annotation file in COCO format
        sem_seg_root (none): not used, to be consistent with
            `register_coco_panoptic_separated`.
        instances_json (str): path to the json instance annotation file


    """
    meta = {}
    meta['thing_dataset_id_to_contiguous_id'] =  {1: 0, 2: 1, 3: 2, 4: 3}
    meta['stuff_dataset_id_to_contiguous_id'] =  {1: 0, 2: 1, 3: 2, 4: 3}
    image_root = "/content/detectron2/projects/Panoptic-DeepLab/datasets/train2017"
    panoptic_root = "/content/detectron2/projects/Panoptic-DeepLab/datasets/panoptic_train2017"
    panoptic_json = "/content/detectron2/projects/Panoptic-DeepLab/datasets/annotations/panoptic_train2017.json"
    instances_json = "/content/detectron2/projects/Panoptic-DeepLab/datasets/annotations/instances_train2017.json"
    register_coco_panoptic("fire_train", meta, image_root, panoptic_root, panoptic_json, instances_json)


    #register_coco_instances("fire_train", {}, '/content/drive/MyDrive/Colab_Notebooks/CMPE295B/detectron2/projects/Panoptic-DeepLab/datasets/coco/annotations/panoptic_train2017.json', "/content/drive/MyDrive/Colab_Notebooks/CMPE295B/detectron2/projects/Panoptic-DeepLab/datasets/coco/train2017")
    dataset_dicts = DatasetCatalog.get("fire_train")
    print("dataset_dict---------------", dataset_dicts)

  

    MetadataCatalog.get("fire_train").thing_classes = ['Fire']
    MetadataCatalog.get("fire_train").stuff_classes = ['NoFire', 'Smoke', 'BurntArea']



