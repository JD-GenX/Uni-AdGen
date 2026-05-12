#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from collections import defaultdict
import random
import PIL
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

from .dataset.util import image_normalize
from .dataset.augmentations import RandomMirror, RandomSampleCrop, CenterSampleCrop





import pdb


Image.MAX_IMAGE_PIXELS = None

class GritSceneGraphDataset(Dataset):
    def __init__(self, list_tokenizers, grit_json, 
                 image_dir, instances_json, stuff_json=None,
                 stuff_only=True, image_size=(64, 64), mask_size=16,
                 max_num_samples=None,proportion_empty_prompts=0.05,
                 include_relationships=True, min_object_size=0.02,
                 min_objects_per_image=3, max_objects_per_image=8, left_right_flip=False,
                 include_other=False, instance_whitelist=None, stuff_whitelist=None, mode='train',
                 use_deprecated_stuff2017=False, deprecated_coco_stuff_ids_txt='', filter_mode='LostGAN',
                 use_MinIoURandomCrop=False,
                 return_origin_image=False, specific_image_ids=None,
                 args=None
                 ):
        """
        A PyTorch Dataset for loading Coco and Coco-Stuff annotations and converting
        them to scene graphs on the fly.

        Inputs:
        - image_dir: Path to a directory where images are held
        - instances_json: Path to a JSON file giving COCO annotations
        - stuff_json: (optional) Path to a JSON file giving COCO-Stuff annotations
        - stuff_only: (optional, default True) If True then only iterate over
          images which appear in stuff_json; if False then iterate over all images
          in instances_json.
        - image_size: Size (H, W) at which to load images. Default (64, 64).
        - mask_size: Size M for object segmentation masks; default 16.
        - max_num_samples: If None use all images. Other wise only use images in the
          range [0, max_num_samples). Default None.
        - include_relationships: If True then include spatial relationships; if
          False then only include the trivial __in_image__ relationship.
        - min_object_size: Ignore objects whose bounding box takes up less than
          this fraction of the image.
        - min_objects_per_image: Ignore images which have fewer than this many
          object annotations.
        - max_objects_per_image: Ignore images which have more than this many
          object annotations.
        - include_other: If True, include COCO-Stuff annotations which have category
          "other". Default is False, because I found that these were really noisy
          and pretty much impossible for the system to model.
        - instance_whitelist: None means use all instance categories. Otherwise a
          list giving a whitelist of instance category names to use.
        - stuff_whitelist: None means use all stuff categories. Otherwise a list
          giving a whitelist of stuff category names to use.
        """
        super(Dataset, self).__init__()

        self.args = args

        self.return_origin_image = return_origin_image
        if self.return_origin_image:
            self.origin_transform = T.Compose([
                T.ToTensor(),
                image_normalize()
            ])

        if stuff_only and stuff_json is None:
            print('WARNING: Got stuff_only=True but stuff_json=None.')
            print('Falling back to stuff_only=False.')

        self.proportion_empty_prompts = proportion_empty_prompts
        self.use_deprecated_stuff2017 = use_deprecated_stuff2017
        self.deprecated_coco_stuff_ids_txt = deprecated_coco_stuff_ids_txt
        self.mode = mode
        self.max_objects_per_image = max_objects_per_image
        self.image_dir = image_dir
        self.mask_size = mask_size
        self.max_num_samples = max_num_samples
        self.include_relationships = include_relationships
        self.filter_mode = filter_mode
        self.image_size = image_size
        self.min_image_size = min(self.image_size)
        self.min_object_size = min_object_size
        self.left_right_flip = left_right_flip
        if left_right_flip:
            self.random_flip = RandomMirror()

        self.layout_length = self.max_objects_per_image + 2

        self.no_text = args.no_text

        self.use_MinIoURandomCrop = use_MinIoURandomCrop
        #self.use_MinIoURandomCrop = False
        if use_MinIoURandomCrop:
            self.MinIoURandomCrop = RandomSampleCrop()
            self.MinIoUCenterCrop = CenterSampleCrop()

        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize(size=image_size, antialias=True),
            image_normalize()
        ])

        self.transform_history = T.Compose([
            T.ToTensor(),
        ])

        self.transform_cond = T.Compose([
            T.ToTensor(),
            #T.Resize(size=image_size, antialias=True),
            #image_normalize()
        ])

        self.total_num_bbox = 0
        self.total_num_invalid_bbox = 0

        #self.tokenizers = tokenizers
        self.tokenizers_one, self.tokenizers_two = list_tokenizers

        if args.grit_json is not None:
            grit_json = args.grit_json
        else:
            grit_json='/home/dataset_process/personalized_dataset/test_dataset.json'
        
        with open(grit_json, 'r') as f:
            grit_data = json.load(f)

        self.image_ids = []
        self.image_id_to_objects = {}
        for idx, obj_data in enumerate(grit_data):
            f_img_path = obj_data["f_path"]
            if obj_data['f_path_white'] != '':
                f_trans_path = obj_data['f_path_white']
                f_white_path = obj_data['f_path_white']
            else:
                f_trans_path = obj_data['f_path_trans']
                f_white_path = obj_data['f_path_trans']                
            list_exps = obj_data["ref_exps"]
            image_w = obj_data["width"]
            image_h = obj_data["height"]
            caption = obj_data["caption"]
            url = obj_data["url"]
            sku_title = obj_data['sku_title_cn']
            
            # ocr_path
            f_ocr_path = obj_data['f_ocr_path']
            max_ratio = obj_data['max_ratio']

            # gt_title, word_list
            gt_title = obj_data["gt_title_cn"]
            word_list = obj_data["word_list"]

            # history data
            f_history_images_path = obj_data["history_images"]
            history_titles = obj_data["history_titles"]
            text_similarities = obj_data["text_similarity"]

            obj_nums = len(list_exps)
            # get sub-caption
            list_bbox_info = []
            for box_info in list_exps:
                x1, y1, x2, y2 = box_info
                phrase = obj_data['cid3_english_name']

                x1, y1 = min(x1, image_w), min(y1, image_h)
                x2, y2 = min(x2, image_w), min(y2, image_h)
                if int(x2 - x1) < 0.05 * image_w or int(y2 - y1) < 0.05 * image_h:
                    continue
                
                list_bbox_info.append([phrase, [x1, y1, int(x2 - x1), int(y2 - y1)]])
                if len(list_bbox_info) >= self.max_objects_per_image:
                    break

            list_bbox_info = [['', [3,3,100,100]]]
            self.image_ids.append([idx, f_img_path, obj_nums, f_trans_path, f_white_path, f_ocr_path, f_history_images_path])
            self.image_id_to_objects.setdefault(idx, [sku_title, caption, image_w, image_h, list_bbox_info, url, max_ratio, gt_title, word_list, history_titles, text_similarities])

        print ("data nums : %s." % len(self.image_id_to_objects))

    def filter_invalid_bbox(self, H, W, bbox, is_valid_bbox, verbose=False):
        for idx, obj_bbox in enumerate(bbox):
            if not is_valid_bbox[idx]:
                continue
            self.total_num_bbox += 1

            x, y, w, h = obj_bbox

            if (x >= W) or (y >= H):
                is_valid_bbox[idx] = False
                self.total_num_invalid_bbox += 1
                if verbose:
                    print(
                        'total_num = {}, invalid_num = {}, x = {}, y={}, w={}, h={}, W={}, H={}'.format(
                            self.total_num_bbox, self.total_num_invalid_bbox, x, y, w, h, W, H,
                        )
                    )
                continue

            x0, y0, x1, y1 = x, y, x + w, y + h
            x1 = np.clip(x + w, 1, W)
            y1 = np.clip(y + h, 1, H)

            if (y1 - y0 < self.min_object_size * H) or (x1 - x0 < self.min_object_size * W):
                is_valid_bbox[idx] = False
                self.total_num_invalid_bbox += 1
                if verbose:
                    print(
                        'total_num = {}, invalid_num = {}, x = {}, y={}, w={}, h={}, W={}, H={}'.format(
                            self.total_num_bbox, self.total_num_invalid_bbox, x, y, w, h, W, H,
                        )
                    )
                continue
            bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3] = x0, y0, x1, y1

        return bbox, is_valid_bbox

    def total_objects(self):
        total_objs = 0
        for i, image_info in enumerate(self.image_ids):
            total_objs += image_info[2]
        return total_objs

    def get_init_meta_data(self, image_id, caption):
        layout_length = self.layout_length
        list_clip_text_ids = self.tokenize_caption("")
        meta_data = {
            'obj_bbox': torch.zeros([layout_length, 4]),
            'obj_class': [""] * layout_length,
            'is_valid_obj': torch.zeros([layout_length]),
            'upd_is_valid_obj': torch.zeros([layout_length]),
            'obj_class_text_ids': [list_clip_text_ids] * layout_length,
        }

        meta_data['obj_bbox'][0] = torch.FloatTensor([0, 0, 1, 1])
        meta_data['obj_class'][0] = caption
        meta_data['is_valid_obj'][0] = 1.0
        meta_data['upd_is_valid_obj'][0] = 1.0

        list_clip_text_ids = self.tokenize_caption(caption)
        meta_data['obj_class_text_ids'][0] = list_clip_text_ids

        return meta_data

    def load_image(self, image_path, alpha=False):
        with open(image_path, 'rb') as f:
            with PIL.Image.open(f) as image:
                image = image.resize((512,512))
                if not alpha:
                    image = image.convert('RGB')
        return image

    def adjust_online_trans(self, trans_img, max_ratio):
        w, h = trans_img.size
        alpha = trans_img.split()[3]
        y_coord, x_coord = np.where(np.array(alpha) > 50)
        
        min_x = np.min(x_coord)
        max_x = np.max(x_coord)
        min_y = np.min(y_coord)
        max_y = np.max(y_coord)
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        crop_img = trans_img.crop((min_x, min_y, max_x, max_y))
        crop_x = max_x - min_x
        crop_y = max_y - min_y
    
        if (crop_x)/w> (crop_y)/h:
            if (crop_x)/w > max_ratio:
                crop_img = crop_img.resize((int(max_ratio*w), int(crop_y*max_ratio*w/crop_x)))
        else:
            if crop_y/h>max_ratio:
                crop_img = crop_img.resize((int(max_ratio*h*crop_x/crop_y), int(max_ratio*h)))
       
     
        background = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        background.paste(crop_img, (int(center_x - crop_img.size[0] / 2), int(center_y - crop_img.size[1] / 2)))
        return background
    
    def load_trans_image(self, image_path, alpha=False, no_text=True, max_ratio=1.0):
        with open(image_path, 'rb') as f:
            with PIL.Image.open(f) as image:
                image = image.convert('RGBA')
                if no_text:
                    image = self.adjust_online_trans(image, max_ratio) 
                if not alpha:
                    image = image.convert('RGB')
                image = image.resize((512,512))
        return image
    
    def __len__(self):
        return len(self.image_ids)

    def tokenize_caption(self, caption):
        captions = []
        if random.random() < self.proportion_empty_prompts:
            captions.append("")
        else:
            captions.append(caption)
        clip_inputs_one = self.tokenizers_one(
            captions, max_length = self.tokenizers_one.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        clip_inputs_two = self.tokenizers_two(
            captions, max_length = self.tokenizers_two.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return [clip_inputs_one.input_ids, clip_inputs_two.input_ids]

    def resize_img(self, image, obj_bbox, obj_class):
        ori_width, ori_height = image.size
        res_min_size = self.min_image_size
        if ori_height < ori_width:
            resize_height = res_min_size
            aspect_r = ori_width / ori_height
            resize_width = int(resize_height * aspect_r)
            im_resized = image.resize((resize_width, resize_height))

            rescale = resize_height / ori_height
            re_obj_bbox = obj_bbox * rescale
        else:
            resize_width = res_min_size
            aspect_r = ori_height / ori_width
            resize_height = int(resize_width * aspect_r)
            im_resized = image.resize((resize_width, resize_height))

            rescale = resize_height / ori_height
            re_obj_bbox = obj_bbox * rescale

        return im_resized, re_obj_bbox, obj_class

    def draw_image(self, image, obj_bbox, obj_class, img_save):
        dw_img = PIL.Image.fromarray(np.uint8(image * 255))
        draw = PIL.ImageDraw.Draw(dw_img)
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        for iix in range(len(obj_bbox)):
            rec = obj_bbox[iix]
            d_rec = [int(xx) for xx in rec]
            draw.rectangle(d_rec, outline = color, width = 3)

            text = obj_class[iix]
            font = ImageFont.truetype("/home/jovyan/boomcheng-data/tools/font/msyh.ttf", size=10)
            draw.text((d_rec[0], d_rec[1]), text, font = font, fill="red", align="left")
        dw_img.save(img_save)

    def draw_image_xywh(self, image, obj_bbox, obj_class, img_save):
        dw_img = PIL.Image.fromarray(np.uint8(image * 255))
        draw = PIL.ImageDraw.Draw(dw_img)
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        for iix in range(len(obj_bbox)):
            rec = obj_bbox[iix]
            d_rec = [int(xx) for xx in rec]
            d_rec[2] += d_rec[0]
            d_rec[3] += d_rec[1]
            draw.rectangle(d_rec, outline = color, width = 3)

            text = obj_class[iix]
            font = ImageFont.truetype("/home/jovyan/boomcheng-data/tools/font/msyh.ttf", size=10)
            draw.text((d_rec[0], d_rec[1]), text, font = font, fill="red", align="left")
        dw_img.save(img_save)

    def transparent_to_white(self, input_path, no_text=True, max_ratio=1.0): 
        img = Image.open(input_path).convert("RGBA")
        if no_text:
            img = self.adjust_online_trans(img, max_ratio)
        white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        merged = Image.alpha_composite(white_bg, img)
        merged = merged.convert("RGB")
        merged = merged.resize((512,512))
        return merged

    def weighted_random_sampling(self, history_images_paths, history_titles, text_similarities, k = 10, random_seed = None, gt_title=None):
        """
        基于text_similarity进行加权随机采样
        
        Args:
            history_images_paths: List[str], 历史图像路径列表
            history_titles: List[str], 历史标题列表
            text_similarities: List[float], 文本相似度列表
            k: int, 采样数量
            random_seed: int, 随机种子，用于确保可重现性
            
        Returns:
            tuple: (采样后的图像路径列表, 采样后的标题列表, 采样后的相似度列表)
        """
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # 确保所有列表长度一致
        n_items = len(history_titles)
        if len(text_similarities) != n_items or len(history_images_paths) != n_items:
            # 如果长度不一致，截取到最短长度
            min_len = min(n_items, len(text_similarities), len(history_images_paths))
            history_titles = history_titles[:min_len]
            text_similarities = text_similarities[:min_len]
            history_images_paths = history_images_paths[:min_len]
            n_items = min_len
            print(f"警告：数据长度不一致，已截取到最短长度 {min_len}")

        
        # 将相似度转换为权重（确保所有权重为正数）
        weights = np.array(text_similarities)
        
        # 处理负数或零值，加上小的正数偏移
        min_weight = weights.min()
        if min_weight <= 0:
            weights = weights + abs(min_weight) + 1e-6
            print(f"检测到负数或零权重，已添加偏移量: {abs(min_weight) + 1e-6}")
        
        # 归一化权重
        weights = weights / weights.sum()
        
        # 进行加权随机采样（不放回）
        if n_items < k:
            selected_indices = np.random.choice(
                n_items, 
                size=k-n_items, 
                replace=False, 
                p=weights
            )
            selected_indices = np.array(list(range(n_items)) + list(selected_indices))
        else:
            selected_indices = np.random.choice(
                n_items, 
                size=k, 
                replace=False, 
                p=weights
            )
        
        if gt_title is not None:
            duplicate_positions = [i for i, idx in enumerate(selected_indices) if history_titles[idx] == gt_title]
            
            if duplicate_positions:
                if n_items == k:
                    # 历史标题数量等于采样数量，复制现有不重复的标题
                    valid_positions = [i for i, idx in enumerate(selected_indices) if history_titles[idx] != gt_title]
                    if valid_positions:
                        for pos in duplicate_positions:
                            replace_pos = random.choice(valid_positions)
                            selected_indices[pos] = selected_indices[replace_pos]
                else:
                    # 历史标题数量大于采样数量，从未选中的标题中随机选择
                    available_indices = [i for i in range(n_items) if history_titles[i] != gt_title and i not in selected_indices]
                    for pos in duplicate_positions:
                        if available_indices:
                            new_idx = random.choice(available_indices)
                            selected_indices[pos] = new_idx
                            available_indices.remove(new_idx)

        # 按索引提取采样结果
        sampled_images = [history_images_paths[i] for i in selected_indices]
        sampled_titles = [history_titles[i] for i in selected_indices]
        sampled_similarities = [text_similarities[i] for i in selected_indices]
        
        return sampled_images, sampled_titles, sampled_similarities

    def __getitem__(self, index):
        """
        Get the pixels of an image, and a random synthetic scene graph for that
        image constructed on-the-fly from its COCO object annotations. We assume
        that the image will have height H, width W, C channels; there will be O
        object annotations, each of which will have both a bounding box and a
        segmentation mask of shape (M, M). There will be T triples in the scene
        graph.

        Returns a tuple of:
        - image: FloatTensor of shape (C, H, W)
        - objs: LongTensor of shape (O,)
        - boxes: FloatTensor of shape (O, 4) giving boxes for objects in
          (x0, y0, x1, y1) format, in a [0, 1] coordinate system
        - masks: LongTensor of shape (O, M, M) giving segmentation masks for
          objects, where 0 is background and 0 is object.

        """
        f_idx, f_image_path, f_obj_nums, f_trans_path, f_white_path, f_ocr_path, f_history_images_path = self.image_ids[index] 

        image = self.load_image(f_image_path)

        W, H = image.size
        sku_title, caption, image_w, image_h, list_bbox_info, url, max_ratio, gt_title, word_list, history_titles, text_similarities = self.image_id_to_objects[f_idx]

        if W != image_w or H != image_h:
            index = 0
            f_idx, f_image_path, f_obj_nums, f_trans_path, f_white_path, f_ocr_path, f_history_images_path = self.image_ids[index]
            image = self.load_image(f_image_path)
            sku_title, caption, image_w, image_h, list_bbox_info, url, max_ratio, gt_title, word_list, history_titles, text_similarities = self.image_id_to_objects[f_idx]

        trans_image = self.load_trans_image(f_trans_path, no_text=self.no_text, max_ratio=max_ratio)
        trans_image_alpha = self.load_trans_image(f_trans_path, alpha=True, no_text=self.no_text, max_ratio=max_ratio)
        white_image = self.transparent_to_white(f_white_path, no_text=self.no_text, max_ratio=max_ratio)

        f_img_nm = f_image_path.split("/")[-1]

        num_obj = len(list_bbox_info)
        obj_bbox = [obj[1] for obj in list_bbox_info]   # [x, y, w, h]
        obj_bbox = np.array(obj_bbox)
        obj_class = [obj[0] for obj in list_bbox_info]
        is_valid_obj = [True for _ in range(num_obj)]

        if True:
            W, H = image.size
            obj_bbox, is_valid_obj = self.filter_invalid_bbox(H=H, W=W, bbox=obj_bbox, is_valid_bbox=is_valid_obj)

        if True:
            image, obj_bbox, obj_class = self.resize_img(image, obj_bbox, obj_class)

        if self.return_origin_image:
            origin_image = np.array(image, dtype=np.float32) / 255.0
        image = np.array(image, dtype=np.float32) / 255.0
        trans_image = np.array(trans_image, dtype=np.float32) / 255.0
        white_image = np.array(white_image, dtype=np.float32) / 255.0
        trans_image_alpha = np.array(trans_image_alpha, dtype=np.float32) / 255.0


        H, W, _ = image.shape

        # get meta data
        meta_data = self.get_init_meta_data(f_idx, caption)
        meta_data['width'], meta_data['height'] = image_w, image_h
        meta_data['original_sizes_hw'] = (image_h, image_w)
        meta_data['num_obj_ori'] = num_obj

        for iid in range(len(is_valid_obj)):
            meta_data['is_valid_obj'][1+iid] = is_valid_obj[iid]

        # flip
        if self.left_right_flip and random.random() < 0.5:
            image, obj_bbox, obj_class = self.random_flip(image, obj_bbox, obj_class)
        
        base_class = obj_class
        base_bbox = obj_bbox
        base_image = PIL.Image.fromarray(np.uint8(image * 255))

        # random crop image and its bbox
        crop_top_left = (0,0)
        if self.use_MinIoURandomCrop:#true
            r_obj_bbox = obj_bbox[is_valid_obj]
            r_obj_class = [obj_class[ii] for ii in range(len(is_valid_obj)) if is_valid_obj[ii]]

            if True:
                crop_top_left, image, upd_obj_bbox, upd_obj_class, upd_is_valid_obj = self.MinIoUCenterCrop(image, r_obj_bbox, r_obj_class)
                

            meta_data['new_height'] = image.shape[0]
            meta_data['new_width'] = image.shape[1]
            H, W, _ = image.shape
        else:
            #### add
            upd_is_valid_obj = is_valid_obj
            upd_obj_bbox = obj_bbox
            upd_obj_class = obj_class

        meta_data["crop_top_lefts"] = crop_top_left     # (x, y)
        for iid in range(len(upd_is_valid_obj)):
            meta_data['upd_is_valid_obj'][1+iid] = int(upd_is_valid_obj[iid])

        obj_bbox, obj_class = upd_obj_bbox, upd_obj_class

        H, W, C = image.shape
        ############### condition_image #############
        list_cond_image = []
        cond_image = np.zeros_like(image, dtype=np.uint8)
        list_cond_image.append(cond_image)
        for iit in range(len(obj_bbox)):
            dot_bbox = obj_bbox[iit]
            dx1, dy1, dx2, dy2 = [int(xx) for xx in dot_bbox]
            cond_image = np.zeros_like(image, dtype=np.uint8)
            cond_image[dy1:dy2, dx1:dx2] = 1
            list_cond_image.append(cond_image)

        obj_bbox = torch.FloatTensor(obj_bbox)

        obj_bbox[:, 0::2] = obj_bbox[:, 0::2] / W
        obj_bbox[:, 1::2] = obj_bbox[:, 1::2] / H

        num_selected = min(obj_bbox.shape[0], self.max_objects_per_image)
        selected_obj_idxs = random.sample(range(obj_bbox.shape[0]), num_selected)#[2, 0, 1]

        meta_data['obj_bbox'][1:1 + num_selected] = obj_bbox[selected_obj_idxs]
        list_text_select = [obj_class[iv] for iv in selected_obj_idxs]
        meta_data['obj_class'][1:1 + num_selected] = list_text_select #['Pink Vans, with pink roses on the outer side', 'Pink Vans, with pink roses on the outer side', 'pink roses on the outer side', 'pink roses on the outer side', '', '', '', '', '', '']

        obj_cond_image = np.stack(list_cond_image, axis=0)
        meta_data['cond_image'] = np.zeros([self.layout_length, H, W, C])
        meta_data['cond_image'][0:len(list_cond_image)] = obj_cond_image
        meta_data['cond_image'][1:1 + num_selected] = obj_cond_image[1:][selected_obj_idxs]
        meta_data['cond_image'] = torch.from_numpy(meta_data['cond_image'].transpose(0,3,1,2))
            
        list_clip_text_ids = self.tokenize_caption(caption)
        meta_data['base_caption'] = caption
        meta_data['base_class_text_ids'] = list_clip_text_ids

        meta_data['num_selected'] =  1 + num_selected
        meta_data['url'] = url
        meta_data['sku_title'] = sku_title
        meta_data['gt_title'] = gt_title
        meta_data['word_list'] = word_list

        # tokenizer
        for iit in range(len(list_text_select)):
            text = list_text_select[iit]
            list_clip_text_ids = self.tokenize_caption(text)
            meta_data['obj_class_text_ids'][1+iit] = list_clip_text_ids

        if self.return_origin_image:
            meta_data['origin_image'] = self.origin_transform(origin_image)

        # history images & history titles
        k = self.args.history_length
        if self.args.sampling_mode == 'random':
            random.seed(self.args.seed)
            sampled_indices = random.sample(range(len(f_history_images_path)), k)
            sampled_images_path = [f_history_images_path[i] for i in sampled_indices]
            sampled_titles = [history_titles[i] for i in sampled_indices]            
        else:
            sampled_images_path, sampled_titles, sampled_similarities = self.weighted_random_sampling(
                f_history_images_path, history_titles, text_similarities, k=k, random_seed=42, gt_title=gt_title
            )

        history_images = []
        history_titles_full = []

        for idx, f_history_image_path in enumerate(sampled_images_path):
            try:
                history_image = self.load_image(f_history_image_path)
                history_titles_full.append(sampled_titles[idx])
                history_image = np.array(history_image, dtype=np.float32)
                history_images.append(self.transform_history(history_image))
            except:
                if len(history_images) > 0:
                    history_images.append(history_images[-1])
                    history_titles_full.append(history_titles_full[-1])
                else:
                    history_images.append(np.zeros((512,512,3), dtype=np.float32))
                    history_titles_full.append('')
                print(f"Error loading image {f_history_image_path}")

        history_images = torch.stack(history_images, 0)   
        meta_data["history_images"] = history_images
        meta_data["history_titles"] = history_titles_full
        meta_data["pixel_values"] = self.transform(image)
        meta_data["trans_pixel_values"] = self.transform(trans_image)
        meta_data["white_pixel_values"] = self.transform(white_image)
        meta_data["trans_pixel_values_alpha"] = self.transform_cond(trans_image_alpha)

        meta_data["image_path"] = f_image_path
        meta_data["trans_image_path"] = f_trans_path
        meta_data["white_image_path"] = f_white_path

        meta_data["obj_bbox_neg"] = torch.zeros((0,4))
        meta_data["obj_class_neg"] = []

        if True:  ## canny
            from src.condition.canny import CannyDetector
            get_control = CannyDetector()

        # 使用白底图/原图
        white_image = self.transparent_to_white(f_white_path, no_text=self.no_text, max_ratio=max_ratio)
        white_image = np.array(white_image, dtype=np.float32) / 255.0
        meta_data["control"] = self.transform(white_image)
        
        return meta_data

def resize_image_to_16_multiple(image_path, condition_type='seg'):
    image = Image.open(image_path)
    width, height = image.size
    
    if condition_type == 'depth':  # The depth model requires a side length that is a multiple of 32
        new_width = (width + 31) // 32 * 32
        new_height = (height + 31) // 32 * 32
    else:
        new_width = (width + 15) // 16 * 16
        new_height = (height + 15) // 16 * 16

    resized_image = image.resize((new_width, new_height))
    return resized_image

def shrink_and_position_transparent_image(input_path, target_size, position):
    img = Image.open(input_path).convert("RGBA")
    img = img.resize((512,512))

    bbox = img.getbbox()
    
    if not bbox:
        raise ValueError("图像没有非零区域(完全透明)")

    cropped = img.crop(bbox)

    original_width, original_height = cropped.size

    target_width, target_height = target_size
    width_ratio = target_width / original_width
    height_ratio = target_height / original_height
    ratio = min(width_ratio, height_ratio)

    new_size = (int(original_width * ratio), int(original_height * ratio))
    resized = cropped.resize(new_size, Image.Resampling.LANCZOS)

    canvas = Image.new("RGBA", (img.width, img.height), (0, 0, 0, 0))

    canvas.paste(resized, position, resized)

    return canvas.convert("RGB")


def grit_collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()   # [bs, 3, 512, 512]

    layo_cond_image = [example["cond_image"] for example in examples]   # bs, [10, 3, 512, 512]
    layo_cond_image = torch.stack(layo_cond_image)  # [bs, 10, 3, 512, 512]
    layo_cond_image = layo_cond_image.to(memory_format=torch.contiguous_format)

    original_sizes = [example["original_sizes_hw"] for example in examples] # bs
    crop_top_lefts = [example["crop_top_lefts"] for example in examples]    # bs
    base_caption = [example["base_caption"] for example in examples]        # bs
    num_selected = [example["num_selected"] for example in examples]

    base_input_ids_one = torch.concat([example["base_class_text_ids"][0] for example in examples])  # bs, 77
    base_input_ids_two = torch.concat([example["base_class_text_ids"][1] for example in examples])  # bs, 77

    list_input_ids_one = []
    list_input_ids_two = []
    for example in examples:
        list_input_text_ids = example['obj_class_text_ids']
        clip_input_ids_one = torch.concat([x[0] for x in list_input_text_ids])  # [10, 77]
        clip_input_ids_two = torch.concat([x[1] for x in list_input_text_ids])  # [10, 77]
        list_input_ids_one.append(clip_input_ids_one)
        list_input_ids_two.append(clip_input_ids_two)

    layo_input_ids_one = torch.stack(list_input_ids_one)    # bs, 10, 77
    layo_input_ids_two = torch.stack(list_input_ids_two)    # bs, 10, 77

    out_data = {
        "pixel_values": pixel_values,
        "cond_image": layo_cond_image,
        "original_sizes_hw": original_sizes,
        "crop_top_lefts": crop_top_lefts,
        "num_selected": num_selected,
        #"base_caption": base_caption,
        "base_input_ids_one": base_input_ids_one,
        "base_input_ids_two": base_input_ids_two,
        "layo_input_ids_one": layo_input_ids_one,
        "layo_input_ids_two": layo_input_ids_two,
    }
    return out_data

def grit_collate_fn_for_layout(batch):
    all_meta_data = defaultdict(list)
    all_imgs = []

    #pdb.set_trace()
    for i, (img, meta_data) in enumerate(batch):
        all_imgs.append(img[None])
        for key, value in meta_data.items():
            all_meta_data[key].append(value)

    all_imgs = torch.cat(all_imgs)
    for key, value in all_meta_data.items():
        if key in ['obj_bbox'] or key.startswith('labels_from_layout_to_image_at_resolution'):
            all_meta_data[key] = torch.stack(value)

    return all_imgs, all_meta_data


def build_grit_dsets(cfg, list_tokenizer, mode='train', args=None):
    assert mode in ['train', 'val', 'test']
    params = cfg.data.parameters
    dataset = GritSceneGraphDataset(
        list_tokenizers=list_tokenizer,
        grit_json=params.grit_json,
        mode=mode,
        filter_mode=params.filter_mode,
        stuff_only=params.stuff_only,
        proportion_empty_prompts=params.proportion_empty_prompts,
        image_size=(params.image_size, params.image_size),
        mask_size=params.mask_size_for_layout_object,
        min_object_size=params.min_object_size,
        min_objects_per_image=params.min_objects_per_image,
        max_objects_per_image=params.max_objects_per_image,
        instance_whitelist=params.instance_whitelist,
        stuff_whitelist=params.stuff_whitelist,
        include_other=params.include_other,
        include_relationships=params.include_relationships,
        use_deprecated_stuff2017=params.use_deprecated_stuff2017,
        deprecated_coco_stuff_ids_txt=os.path.join(params.root_dir, params[mode].deprecated_stuff_ids_txt),
        image_dir=os.path.join(params.root_dir, params[mode].image_dir),
        instances_json=os.path.join(params.root_dir, params[mode].instances_json),
        stuff_json=os.path.join(params.root_dir, params[mode].stuff_json),
        max_num_samples=params[mode].max_num_samples,
        left_right_flip=params[mode].left_right_flip,
        use_MinIoURandomCrop=params[mode].use_MinIoURandomCrop,
        return_origin_image=params.return_origin_image,
        specific_image_ids=params[mode].specific_image_ids,
        args=args
    )

    num_objs = dataset.total_objects()
    num_imgs = len(dataset)
    print('%s dataset has %d images and %d objects' % (mode, num_imgs, num_objs))
    print('(%.2f objects per image)' % (float(num_objs) / num_imgs))

    return dataset

if __name__ == '__main__':

    from omegaconf import OmegaConf
    cfg_data = OmegaConf.load("/home/jovyan/boomcheng-data/aigc/LayoutProj/diffusers_0263/examples/controlnet/latent_LayoutDiffusion_large.yaml")
    from transformers import AutoTokenizer
    pretrained_model = "/home/jovyan/boomcheng-data-shcdt/herunze/models/stable-diffusion-xl-base-1.0"
    tokenizer_one = AutoTokenizer.from_pretrained(
                pretrained_model,
                subfolder="tokenizer",
                revision=None,
                use_fast=False,
            )
    tokenizer_two = AutoTokenizer.from_pretrained(
                pretrained_model,
                subfolder="tokenizer_2",
                revision=None,
                use_fast=False,
            )

    dataset = build_grit_dsets(cfg_data, [tokenizer_one, tokenizer_two], mode='train')

    if True:
        for ii in range(852, 860):
            meta_data = dataset[ii]
            print (ii,meta_data["pixel_values"].shape)
            pdb.set_trace()
            pass



