from torchvision.datasets.vision import VisionDataset

from PIL import Image
import base64
from io import BytesIO
import os
import os.path
import random
import json
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

class DatasetFolder(VisionDataset):
    def __init__(
            self,
            root: str,
            part_index: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None
            
    ) -> None:
        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        
        # ******* from text ******
        
        self.input_path = root + '/' + part_index
        print(f"input file path:{self.input_path}")
        
        samples = []  # 空列表留着存放数据
        with open(self.input_path, 'r') as f:   # 按行读取csv
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                hashid, img_str = line.split(' ', maxsplit=1)
                samples.append((hashid, img_str))
        self.samples = samples

        print("Find %d samples in file!" % (len(samples)))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        hashid, img_str = self.samples[index]
        
        # img_byte = base64.urlsafe_b64decode(img_str)
        # code_base64=base64.urlsafe_b64encode(img_str)
        image_io = BytesIO(base64.urlsafe_b64decode(img_str))
        image=Image.open(image_io).convert("RGB")
        image_io.close()
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, hashid

    def __len__(self) -> int:
        return len(self.samples)

# ******   
class ImageFolder(DatasetFolder):
    
    def __init__(
            self,
            root: str,
            part_index: str = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None
    ):
        super(ImageFolder, self).__init__(root, part_index,
                                          transform=transform,
                                          target_transform=target_transform)
        self.imgs = self.samples
