from torchvision.datasets.vision import VisionDataset

from PIL import Image

import os
import os.path
import torch
import random
import json
from typing import Any, Callable, cast, Dict, List, Optional, Tuple


def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(
    directory: str,
    class_to_idx: Dict[str, int],
    extensions: Optional[Tuple[str, ...]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))
    is_valid_file = cast(Callable[[str], bool], is_valid_file)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
    return instances


class DatasetFolder(VisionDataset):
    def __init__(
        self,
        root: str,
        download_root: str,
        loader: Callable[[str], Any],
        pair_num: int,
        for_eval: bool,
        extensions: Optional[Tuple[str, ...]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        index_file: Optional[str] = None, 
    ) -> None:
        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        
        with open(index_file, mode="r", encoding="utf-8") as reader:
            samples = []
            for line in reader:
                hashid_pair = line.strip().split("\t")[1:]
                assert len(hashid_pair) == pair_num
                samples.append(hashid_pair)
        
        self.root = root
        self.download_root = download_root
        self.loader = loader
        self.extensions = extensions
        self.samples = samples
        self.for_eval = for_eval

        print("Find %d samples in root!" % (len(samples)))

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
    
    def _get_transformed_img(self, class_name, img_file):
        if self.download_root:
                os.makedirs(os.path.join(self.download_root, class_name), exist_ok=True)
                if os.path.exists(os.path.join(self.download_root, class_name, img_file)):
                    sample = self.loader(os.path.join(self.download_root, class_name, img_file))
                else:
                    sample = self.loader(os.path.join(self.root, class_name, img_file))
                    sample.save(os.path.join(self.download_root, class_name, img_file))
        else:
            # root/class_name/each_file download地址
            sample = self.loader(os.path.join(self.root, class_name, img_file))
            
        if self.transform is not None:
            sample = self.transform(sample)
            
        return sample

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        
        hashids = self.samples[index]
        # print(hashids)
        img_files = map(lambda hashid: hashid + ".JPEG", hashids)
        # print(list(img_files))
        class_names = map(lambda hashid: hashid[-4:], hashids)
        # print(list(class_names))
        
        view1_list = []
        view2_list = []
        eval_view_list = []
        
        for img_file, class_name in zip(img_files, class_names):
            sample = self._get_transformed_img(class_name, img_file)
            
            if not self.for_eval:
                view1_list.append(sample[0])
                view2_list.append(sample[1])
            else:
                eval_view_list.append(sample)
            
        if not self.for_eval:
            # [3,dim], [B,3,dim]
            return torch.stack(view1_list, dim=0), torch.stack(view2_list, dim=0)
        else:
            return torch.stack(eval_view_list, dim=0)

    def __len__(self) -> int:
        return len(self.samples)

    def filenames(self, indices=[], basename=False):
        if indices:
            if basename:
                return [os.path.basename(self.samples[i][0]) for i in indices]
            else:
                return [self.samples[i][0] for i in indices]
        else:
            if basename:
                return [os.path.basename(x[0]) for x in self.samples]
            else:
                return [x[0] for x in self.samples]


# ******
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)
   
        
def default_loader(path: str) -> Any:
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
            
            
class ImageFolder(DatasetFolder):
    
    def __init__(
            self,
            root: str,
            download_root: str = None,
            pair_num: int = 3,
            for_eval: bool = False,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            index_file: Optional[str] = None, 
    ):
        super(ImageFolder, self).__init__(
            root, download_root, loader, pair_num, for_eval,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file, index_file=index_file)
        
        self.imgs = self.samples