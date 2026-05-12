from torch.utils.data import Dataset
import random
from copy import deepcopy
from torchvision.transforms import Resize
from glob import glob
import torch
from PIL import Image
from torch.utils.data import Dataset
import random
from copy import deepcopy
from torchvision.transforms import Resize
from datasets import load_dataset
from src.utils.funcs import convert_to_np, load_jsonl
import numpy as np
import os
from src.utils.funcs import *

class Dataset_edit_coco_edit(Dataset):
    def __init__(
        self,
        args=None,
        is_test=False,
    ):
        self.args = args
        self.is_test = is_test
        # self.datas = load_json("project/janus/dataset/edit/edit.json")

    def __len__(self):
        return 200

    def __getitem__(self, i):
        base_caption = ''
        # base_caption = "Please generate a scene of a home washing machine in use. The image should depict a modern washing machine placed in a bright and tidy laundry room. On the walls of the laundry room, there are a few freshly washed clothes hanging. On the floor, there is a basket filled with dirty laundry. Next to the washing machine, there is a small table with detergent and fabric softener on it. The sunlight coming through the window makes the whole scene cozy and comfortable. Ensure that every detail in the image reflects the practicality and convenience of the washing machine in daily life."
        # base_caption = 'In a cozy and comfortable living room, sunlight filters through the curtains, casting a warm glow on the plush sofa. A few art pieces adorn the walls, adding a touch of homely warmth. In the corner of the room stands a stylish water dispenser, seamlessly blending with the overall decor of the living room. A soft rug lies on the floor, and a small vase with flowers sits on the coffee table alongside a few magazines, creating an atmosphere of relaxation and ease. The water dispenser is not just a practical appliance but an integral part of the living room, offering convenient hydration for family and guests.'
        # base_caption = 'A {} of cosmetics oil is placed on a stone, with virtual flowers and leaves in the background, golden light, close-up, and natural scenery.'
        # path = self.args.coco_200_path
        path = '/home/project/PlanGen/datasets/coco_data'

        # image_path = f'{path}/image/{i}.png'
        # mask_path = f'{path}/mask/{i}.png'
        # box_path = f'{path}/box/{i}.json'
        # box_new_path = f'{path}/box_new/{i}.json'

        image_path = '/home/project/PlanGen/0000.png'
        mask_path = f'{path}/mask/{i}.png'
        box_path = '/home/project/PlanGen/00.json'
        box_new_path = '/home/project/PlanGen/00.json'
        print(box_new_path)
        assert 0
        
        image = load2ts(image_path)
        data1 = load_json(box_path)
        data2 = load_json(box_new_path)
        obj_bbox_1, obj_class_1 = data1['obj_bbox'], data1['obj_class']
        obj_bbox_2, obj_class_2 = data2['obj_bbox'], data2['obj_class']

        obj_bbox_1 = torch.tensor(obj_bbox_1).reshape(1,4)
        obj_bbox_2 = torch.tensor(obj_bbox_2).reshape(1,4)

        obj_bbox_edit = torch.cat([obj_bbox_1, obj_bbox_2], dim=0)
        obj_bbox = obj_bbox_2
        obj_class = [obj_class_2]

        obj_bbox_neg = torch.zeros((0,4))
        obj_class_neg = []

        ret = dict(
            base_caption=base_caption,
            image=image,
            image_path=image_path,
            obj_class=obj_class,
            obj_bbox=obj_bbox,
            obj_bbox_edit=obj_bbox_edit,
            obj_class_neg=obj_class_neg,
            obj_bbox_neg=obj_bbox_neg,
        )
        return ret
    