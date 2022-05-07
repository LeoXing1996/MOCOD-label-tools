import os
import os.path as osp
import random
import re
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .annotation import output_xml, read_xml
"""
.. module:: streamlit_img_label
   :synopsis: manage.
.. moduleauthor:: Tianning Li <ltianningli@gmail.com>
"""


class ImageManager:
    """ImageManager
    Manage the image object.

    Args:
        filename(str): the image file.
    """

    def __init__(self, filename):
        """initiate module"""
        self._filename = filename
        self._img = Image.open(filename)
        self._rects = []
        self._load_rects()
        self._resized_ratio_w = 1
        self._resized_ratio_h = 1

    def _load_rects(self):
        rects_xml = read_xml(self._filename)
        if rects_xml:
            self._rects = rects_xml

    def get_img(self):
        """get the image object

        Returns:
            img(PIL.Image): the image object.
        """
        return self._img

    def get_rects(self):
        """get the rects

        Returns:
            rects(list): the bounding boxes of the image.
        """
        return self._rects

    def resizing_img(self, max_height=700, max_width=700):
        """resizing the image by max_height and max_width.

        Args:
            max_height(int): the max_height of the frame.
            max_width(int): the max_width of the frame.
        Returns:
            resized_img(PIL.Image): the resized image.
        """
        resized_img = self._img.copy()
        if resized_img.height > max_height:
            ratio = max_height / resized_img.height
            resized_img = resized_img.resize((int(resized_img.width * ratio),
                                              int(resized_img.height * ratio)))
        if resized_img.width > max_width:
            ratio = max_width / resized_img.width
            resized_img = resized_img.resize((int(resized_img.width * ratio),
                                              int(resized_img.height * ratio)))

        self._resized_ratio_w = self._img.width / resized_img.width
        self._resized_ratio_h = self._img.height / resized_img.height

        # print(f'Orig Size: {self._img.size}, '
        #       f'Resize Size: {resized_img.size}')
        return resized_img

    def _resize_rect(self, rect):
        resized_rect = {}
        resized_rect["left"] = rect["left"] / self._resized_ratio_w
        resized_rect["width"] = rect["width"] / self._resized_ratio_w
        resized_rect["top"] = rect["top"] / self._resized_ratio_h
        resized_rect["height"] = rect["height"] / self._resized_ratio_h
        if "label" in rect:
            resized_rect["label"] = rect["label"]
        return resized_rect

    def get_resized_rects(self):
        """get resized the rects according to the resized image.

        Returns:
            resized_rects(list): the resized bounding boxes of the image.
        """
        return [self._resize_rect(rect) for rect in self._rects]

    def _chop_box_img(self, rect):
        rect["left"] = int(rect["left"] * self._resized_ratio_w)
        rect["width"] = int(rect["width"] * self._resized_ratio_w)
        rect["top"] = int(rect["top"] * self._resized_ratio_h)
        rect["height"] = int(rect["height"] * self._resized_ratio_h)
        left, top, width, height = (
            rect["left"],
            rect["top"],
            rect["width"],
            rect["height"],
        )

        raw_image = np.asarray(self._img).astype("uint8")
        prev_img = np.zeros(raw_image.shape, dtype="uint8")
        prev_img[top:top + height,
                 left:left + width] = raw_image[top:top + height,
                                                left:left + width]
        prev_img = prev_img[top:top + height, left:left + width]
        label = ""
        if "label" in rect:
            label = rect["label"]
        return (Image.fromarray(prev_img), label)

    def init_annotation(self, rects):
        """init annotation for current rects.

        Args:
            rects(list): the bounding boxes of the image.
        Returns:
            prev_img(list): list of preview images with default label.
        """
        self._current_rects = rects
        return [self._chop_box_img(rect) for rect in self._current_rects]

    def set_annotation(self, index, label):
        """set the label of the image.

        Args:
            index(int): the index of the list of bounding boxes of the image.
            label(str): the label of the bounding box
        """
        self._current_rects[index]["label"] = label

    def save_annotation(self):
        """output the xml annotation file."""
        output_xml(self._filename, self._img, self._current_rects)


class MOCODImageManager(ImageManager):

    def __init__(self, filename):
        self._filename = filename
        self._img = Image.open(filename)
        self._rects = []
        self._resized_ratio_w = 1
        self._resized_ratio_h = 1

    def init_rects_from_obj(self, obj_im=None):
        if obj_im is None:
            return
        bbox = obj_im.get_bbox()
        rects = obj_im.bbox_to_rects(bbox)
        self._rects = rects

    def set_annotation(self, index, label):
        raise NotImplementedError(
            'Do not support \'set_annotation\' in \'MOCODImageManager\'.')


class MOCODPairImageManager(ImageManager):

    def __init__(self, fg_name, mask_name):
        self._fg_name, self._mask_name = fg_name, mask_name
        self._fg_img = Image.open(self._fg_name).convert('RGB')
        self._mask_img = Image.open(self._mask_name).convert('RGB')
        self._binary_mask = (np.array(self._mask_img) == 255).astype(np.int64)

        self._fg_crop_img, self._bg_crop_img = self.crop_image()

    def get_fg_img(self):
        return self._fg_img

    def get_mask_img(self):
        return self._mask_img

    def get_binary_mask(self):
        return self._binary_mask

    def get_fg_crop_img(self):
        return self._fg_crop_img

    def get_mask_crop_img(self):
        return self._bg_crop_img

    def get_img(self):
        raise NotImplementedError(
            'Do not support \'get_img\' in \'MOCODPairImageManager\'.')

    @staticmethod
    def bbox_to_rects(bbox):
        """This function provide a mapping from bbox to rectangle.
        """
        h_min, w_min, h_max, w_max = bbox
        rects = [
            dict(top=h_min,
                 left=w_min,
                 height=h_max - h_min,
                 width=w_max - w_min,
                 label='')
        ]
        return rects

    def get_bbox(self):
        h, w, _ = np.nonzero(self.get_binary_mask() == 1)
        h_min, h_max = h.min(), h.max()
        w_min, w_max = w.min(), w.max()

        return [int(coord) for coord in [h_min, w_min, h_max, w_max]]

    def crop_image(self):
        image_tensor = np.array(self.get_fg_img())
        # binary_mask = deepcopy(self.get_binary_mask())
        mask_tensor = np.array(self.get_mask_img())

        h_min, w_min, h_max, w_max = self.get_bbox()

        image_crop = image_tensor[h_min:h_max, w_min:w_max, ...]
        mask_crop = mask_tensor[h_min:h_max, w_min:w_max, ...]

        image_crop_pil = Image.fromarray(image_crop)
        mask_crop_pil = Image.fromarray(mask_crop)
        return image_crop_pil, mask_crop_pil

    @torch.no_grad()
    def forward_model(self, fg, mask, bg, model):

        def pil_to_tensor(inp):
            tensor = torch.from_numpy(inp)
            tensor = tensor[None, ...].permute(0, 3, 1, 2)
            return tensor.type(torch.float32)

        def normalize(ten):
            ten = ((ten / 255.) - 0.5) / 0.5
            return ten

        # 1. to tensor
        fg_tensor = pil_to_tensor(fg).cuda()
        bg_tensor = pil_to_tensor(bg).cuda()
        mask_tensor = pil_to_tensor(mask).cuda()

        # 2. transform
        fg_tensor = normalize(fg_tensor)
        bg_tensor = normalize(bg_tensor)
        # mask_tensor[mask_tensor != 0] = 1  # convert to [0, 1]
        mask_tensor = mask_tensor[:, 0][:, None, ...]

        # 3. resize
        old_size = fg_tensor.shape[2:]
        fg_tensor = F.interpolate(fg_tensor, size=(256, 256), mode='bicubic')
        bg_tensor = F.interpolate(bg_tensor, size=(256, 256), mode='bicubic')
        mask_tensor = F.interpolate(mask_tensor,
                                    size=(256, 256),
                                    mode='bicubic')

        # 4. forward network
        pred = model.processImage(fg_tensor, mask_tensor, bg_tensor)

        pred_rgb = pred[0:1]
        pred_rgb_resize = F.interpolate(pred_rgb,
                                        size=[*old_size],
                                        mode='bicubic')[0]

        # 5. post process
        pred_rgb_resize_np = pred_rgb_resize.permute(1, 2, 0).cpu().numpy()
        pred_rgb_resize_np = (pred_rgb_resize_np + 1) / 2 * 255.

        return pred_rgb_resize_np

    def paste_on_bg(self, rects, bg_im, model=None):
        """
        Args:
            rects (list): the position of the bounding box
        """
        left = rects[0]['left']
        top = rects[0]['top']
        width = rects[0]['width']
        height = rects[0]['height']
        bg_img_np = np.array(bg_im.get_img().convert('RGB'))

        bg_crop = bg_img_np[top:top + height, left:left + width, :]
        bg_crop_size = [width, height]

        obj_img_resize = (deepcopy(self.get_fg_crop_img()).resize(
            bg_crop_size, resample=Image.LANCZOS))
        mask_img_resize = deepcopy(self.get_mask_crop_img()).resize(
            bg_crop_size, resample=Image.LANCZOS)

        obj_resize = np.array(obj_img_resize)
        mask_resize = np.array(mask_img_resize) / 255.  # norm to [0, 1]

        bg_comb = mask_resize * obj_resize + \
            (1 - mask_resize) * bg_crop
        bg_img_np[top:top + height, left:left + width, :] = bg_comb

        bg_img = Image.fromarray(bg_img_np)
        result_dict = dict(fg_resize=obj_img_resize,
                           mask_resize=mask_img_resize,
                           bg_paste=bg_img)

        if model is not None:
            bg_comb_refine = self.forward_model(obj_resize, mask_resize,
                                                bg_crop, model)
            bg_img_np_refine = deepcopy(bg_img_np)
            bg_img_np_refine[top:top + height,
                             left:left + width, :] = bg_comb_refine
            bg_img_refine = Image.fromarray(bg_img_np_refine.astype(np.uint8))
            result_dict['bg_refine'] = bg_img_refine

        return result_dict


class ImageDirManager:

    def __init__(self, dir_name):
        self._dir_name = dir_name
        self._files = []
        self._annotations_files = []

    def get_all_files(self, allow_types=["png", "jpg", "jpeg"]):
        allow_types += [i.upper() for i in allow_types]
        mask = ".*\.[" + "|".join(allow_types) + "]"
        self._files = [
            file for file in os.listdir(self._dir_name)
            if re.match(mask, file)
        ]
        return self._files

    def get_exist_annotation_files(self):
        self._annotations_files = [
            file for file in os.listdir(self._dir_name)
            if re.match(".*.xml", file)
        ]
        return self._annotations_files

    def set_all_files(self, files):
        self._files = files

    def set_annotation_files(self, files):
        self._annotations_files = files

    def get_image(self, index):
        return self._files[index]

    def _get_next_image_helper(self, index):
        while index < len(self._files) - 1:
            index += 1
            image_file = self._files[index]
            image_file_name = image_file.split(".")[0]
            if f"{image_file_name}.xml" not in self._annotations_files:
                return index
        return None

    def get_next_annotation_image(self, index):
        image_index = self._get_next_image_helper(index)
        if image_index:
            return image_index
        if not image_index and len(self._files) != len(
                self._annotations_files):
            return self._get_next_image_helper(0)


class MOCODManager:
    """We assume foreground objects are formed as
    fg_dir
      + A
      | +--- 1.png
      | +--- 2.png
      + B
        + ....
    """

    LABELS = ['Person', 'Carrier', 'Tank', 'Armored', 'Car']
    # LABELS = ['Person']

    LABEL_TO_OBJ = {
        'Person': ['girl', 'woman', 'man', 'walkingman', 'men', 'Ped3'],
        'Carrier': [
            'GAZ_Tiger',
            'TIGER',
            'MAZ537',
            'BTR152',
            'DJS-2022',
        ],
        'Tank': ['T-90', 'T-90A'],
        'Armored': ['LAV', 'VH_BTR70', 'DV_LAV-Jackal', 'DV-LAV-Jackal'],
        'Car': ['HatchBack', 'Sedan', 'sedan2Door', 'Hybrid']
    }

    def __init__(self, fg_dir, bg_dir):
        self._fg_dir = fg_dir
        self._bg_dir = bg_dir
        self.fg_objs = os.listdir(self._fg_dir)

        # load images
        self.fg_files = self.get_fg_files()
        self.bg_files = self.get_bg_files()

    def folder_to_label(self, image_info):
        fg_name = image_info['fg'].split('/')[-2]
        for label, objects in self.LABEL_TO_OBJ.items():
            if any([obj.upper() in fg_name.upper() for obj in objects]):
                return label
        raise ValueError(f'Folder \'{fg_name}\' not belong to any label.')

    def get_fg_files(self):

        obj_fnames = defaultdict(list)
        for obj in self.fg_objs:
            obj_root = osp.join(self._fg_dir, obj)
            for file in os.listdir(obj_root):
                if 'fg' in file and 'png' in file:
                    prefix = file.split('_')[1]
                    mask_file = f'mask_{prefix}'
                    info = dict(fg=osp.join(obj_root, file),
                                mask=osp.join(obj_root, mask_file))
                    obj_fnames[obj].append(info)

        return obj_fnames

    def get_bg_files(self):
        bg_fnames = []
        for root_, dir_, files in os.walk(self._bg_dir):
            for file in files:
                if file and (file.endswith('jpg') or file.endswith('png')):
                    bg_fnames.append(os.path.join(root_, file))
        return bg_fnames

    def get_random_pair(self, label=None):
        pair_info = self.get_random_obj(label)
        pair_info.update(self.get_random_bg())
        return pair_info

    def get_random_bg(self):
        return dict(bg=random.choice(self.bg_files))

    def get_random_obj(self, label=None):
        if label is None:
            label = random.choice(self.LABELS)
        tar_objs = self.LABEL_TO_OBJ[label]
        all_obj_files = []
        for obj, files in self.fg_files.items():
            # if obj.upper in [o.upper() for o in tar_objs]:
            if any([o.upper() in obj.upper() for o in tar_objs]):
                all_obj_files += files

        obj_info = deepcopy(random.choice(all_obj_files))
        return obj_info
