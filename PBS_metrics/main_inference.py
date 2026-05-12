# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse

import torch, base64
from torch import nn
from torch.utils.data import Dataset
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
# from torchvision import models as torchvision_models

from util import utils
# from model import vits
import model.vision_transformer as vits
from util.text_datasets import ImageFolder
import numpy as np


def extract_feature_pipeline(args):
    # ============ preparing data ... ============
    transform = pth_transforms.Compose([
        pth_transforms.Resize(256),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_train = ReturnIndexDataset(args.data_path, part_index=args.part_index, transform=transform)
    # dataset_val = ReturnIndexDataset(os.path.join(args.data_path, "val"), transform=transform)
    sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Data loaded with {len(dataset_train)} inference imgs.")

    # ============ building network ... ============
    if args.arch in vits.__dict__.keys():
        model = vits.__dict__[args.arch](patch_size=args.patch_size, head_dim=args.moco_dim)
    else:
        print("Wrong arch! only support vit!")
        sys.exit(1)
        
    assert os.path.isfile(args.pretrained_weights), "=> no checkpoint found at '{}'".format(args.pretrained_weights)
    print("=> loading checkpoint '{}'".format(args.pretrained_weights))
    
    checkpoint = torch.load(args.pretrained_weights, map_location="cpu")
    # rename moco pre-trained keys
    state_dict = checkpoint['model']
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('module.base_encoder'):
            # remove prefix
            state_dict[k[len("module.base_encoder."):]] = state_dict[k]
            print(k)
        # delete renamed or unused k
        del state_dict[k]
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Model {args.arch} built with msg: {msg}")
    
    model.cuda()
    model.eval()

    # ============ extract features ... ============
    print("Extracting features for train set...")
    train_features = extract_features(model, data_loader_train, args.use_cuda)
    train_features = train_features.cpu()
    train_labels = np.array([s[0] for s in dataset_train.samples])
    # print(train_labels[:10])
    
    # ============ del dataset ... ============
    del data_loader_train
    del dataset_train
    
    # save features and labels
    if args.dump_features and dist.get_rank() == 0:
        torch.save(train_features, os.path.join(args.dump_features, args.part_index+"_train_feat.pth"))
        # torch.save(train_labels.cpu(), os.path.join(args.dump_features, args.part_index+"_train_labels.pth"))
        np.save(os.path.join(args.dump_features, args.part_index+"_train_label.npy"), train_labels)
    return train_features, train_labels

@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True, multiscale=False):
    metric_logger = utils.MetricLogger(delimiter="  ")
    features = None
    for samples, index in metric_logger.log_every(data_loader, 100):
        samples = samples.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        
        if multiscale:
            feats = utils.multi_scale(samples, model)
        else:
            feats = model(samples).clone()

        # init storage feature matrix
        if dist.get_rank() == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            if use_cuda:
                features = features.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")

        # get indexes from all processes
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        # share features between processes
        feats_all = torch.empty(
            dist.get_world_size(),
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )
        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()

        # update storage feature matrix
        if dist.get_rank() == 0:
            if use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l))
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
    return features

def deal_dig(dig):
    dig = round(dig,9)
    return str(dig)

def merge_featAndLabel(features, labels):
    features = features.numpy().tolist()
    with open(os.path.join(args.dump_features, args.part_index+".txt"), "w") as wf:
        for label, feat in zip(labels, features):
            feat_str = " ".join(map(deal_dig, feat))
            wf.write("%s %s\n"%(str(label), feat_str))
    print("write dict success!!!!!!")

class ReturnIndexDataset(ImageFolder):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with weighted k-NN on ImageNet')
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument('--nb_knn', default=[10, 20, 100, 200], nargs='+', type=int,
        help='Number of NN to use. 20 is usually working the best.')
    parser.add_argument('--temperature', default=0.07, type=float,
        help='Temperature used in the voting coefficient')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--part_index', default='part-00000', type=str, help="Part index to inference.")
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag,
        help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--dump_features', default=None,
        help='Path where to save computed features, empty for no saving')
    parser.add_argument('--load_features', default=None, help="""If the features have
        already been computed, where to find them.""")
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local-rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
    parser.add_argument('--moco_dim', default=256, type=int, help='feature dimension (default: 256)')
    args = parser.parse_args()
    
    if args.dump_features:
        assert os.path.isdir(args.dump_features), "dump_featuresl路径不存在!!!!"
        
    utils.init_distributed_mode(args)
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    train_features, train_labels = extract_feature_pipeline(args)

    print("*"*10)
    print(train_features.shape)
    print(train_labels.shape)
    print("*"*10)
    print("Features are ready!")
    
    merge_featAndLabel(train_features, train_labels)
    
    dist.barrier()
