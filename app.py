import json
import os
import os.path as osp
from collections import defaultdict
from copy import deepcopy
from uuid import uuid1

import streamlit as st
import torch
import torch.nn.functional as F

from model_util.networks import RainNet
from model_util.normalize import RAIN
from model_util.util import copy_state_dict
from streamlit_img_label import st_img_label
from streamlit_img_label.manage import (MOCODImageManager, MOCODManager,
                                        MOCODPairImageManager)

OBJ_DIR = './dataset/MOCOD_objs'
BG_DIR = './dataset/MOCOD_bgs'

RES_DIR = './results'
HIST_DIR = './results/hist'
HIST_SEPARTE = '@'
EACH_NEED = 100

DEFAULT_NAME = 'You Name Please'


@st.cache
def load_network():
    from model_util.test_config import cfg
    net = RainNet(input_nc=cfg.input_nc,
                  output_nc=cfg.output_nc,
                  ngf=cfg.ngf,
                  norm_layer=RAIN,
                  use_dropout=not cfg.no_dropout)

    load_path = os.path.join(cfg.checkpoints_dir, cfg.name, 'net_G.pth')
    if not os.path.exists(load_path):
        raise FileExistsError(
            print('%s not exists. Please check the file' % (load_path)))
    print(f'loading the model from {load_path}')
    state_dict = torch.load(load_path)
    copy_state_dict(net.state_dict(), state_dict)
    return net.cuda()


def run():
    st.set_option("deprecation.showfileUploaderEncoding", False)
    user_name = st.sidebar.text_input('User Name', 'You Name Please')

    # load user save hist
    os.makedirs(HIST_DIR, exist_ok=True)
    hist_path = osp.join(HIST_DIR, f'{user_name}.txt')

    model = load_network()

    if 'save_hist' not in st.session_state:
        st.session_state['save_hist'] = defaultdict(list)
    if osp.exists(hist_path):
        with open(hist_path, 'r') as file:
            hist_infos = file.readlines()
            hist_infos = [
                info.strip().split(HIST_SEPARTE) for info in hist_infos
            ]
            st.session_state['save_hist'][user_name] = hist_infos

    # init saved state
    if 'saved' not in st.session_state:
        st.session_state['saved'] = False

    idm = MOCODManager(OBJ_DIR, BG_DIR)

    # init image_info
    if 'image_info' not in st.session_state:
        st.session_state['image_info'] = idm.get_random_pair()

    def user_name_lock(fn):

        def warp_func(*args, **kwargs):
            if user_name == DEFAULT_NAME:
                st.warning('Please Input The User Name')
            else:
                fn(*args, **kwargs)

        return warp_func

    @user_name_lock
    def update_image_info(image_info):
        st.session_state['image_info'].update(image_info)

    @user_name_lock
    def random_bg():
        image_info = idm.get_random_bg()
        update_image_info(image_info)
        st.session_state['saved'] = False

    @user_name_lock
    def random_obj():
        image_info = idm.get_random_obj(st.session_state['label'])
        st.session_state['label'] = idm.folder_to_label(image_info)
        update_image_info(image_info)
        st.session_state['saved'] = False

    @user_name_lock
    def save_results(bbox, img_paste, img_refine):
        if st.session_state['saved']:
            st.warning('Current Results has been saved. '
                       'Please load new image pair.')
            return

        # NOTE: just for debug
        saved = st.session_state['saved']
        print(f'saved: {saved}')
        # << just for debug

        label = st.session_state['label']
        meta = deepcopy(st.session_state['image_info'])
        uuid_str = str(uuid1())
        meta['bbox'] = bbox
        meta['uuid'] = str(uuid1())
        meta['user'] = user_name

        # save
        image_path = osp.join(RES_DIR, label, f'{uuid_str}_paste.png')
        image_refine_path = osp.join(RES_DIR, label, f'{uuid_str}.png')
        meta_path = osp.join(RES_DIR, label, f'{uuid_str}.json')

        img_paste.save(image_path)
        img_refine.save(image_refine_path)

        with open(meta_path, 'w') as file:
            json.dump(meta, file)

        # save to image list
        if 'save_hist' not in st.session_state:
            st.session_state['save_hist'] = defaultdict(list)
            st.session_state['save_hist'][user_name] = [
                (osp.join(RES_DIR, label), uuid_str)
            ]
        else:
            st.session_state['save_hist'][user_name].append(
                (osp.join(RES_DIR, label), uuid_str))

        # save hist to file
        with open(osp.join(HIST_DIR, f'{user_name}.txt'), 'a+') as file:
            file.write(f'{osp.join(RES_DIR, label)}{HIST_SEPARTE}{uuid_str}\n')

        info_str = (f'User \'{user_name}\' save image to \'{image_path}\' '
                    f'(label is \'{label}\').')

        st.info(info_str)
        print(info_str)
        st.session_state['saved'] = True

    @user_name_lock
    def undo():
        if ('save_hist' not in st.session_state) or (
                user_name not in st.session_state['save_hist']) or (len(
                    st.session_state['save_hist'][user_name]) == 0):
            st.info(f'No Save History for user \'{user_name}\'.')
        else:
            save_info = st.session_state['save_hist'][user_name].pop()
            save_dir, save_uuid = save_info

            img_path = osp.join(save_dir, f'{save_uuid}_paste.png')
            image_refine_path = osp.join(save_dir, f'{save_uuid}.png')

            meta_path = osp.join(save_dir, f'{save_uuid}.json')

            # recover image info
            with open(meta_path, 'r') as file:
                image_info = json.load(file)

            st.session_state['image_info'] = dict(fg=image_info['fg'],
                                                  mask=image_info['mask'],
                                                  bg=image_info['bg'])

            os.remove(img_path)
            os.remove(image_refine_path)
            os.remove(meta_path)

            info_str = f'User \'{user_name}\' remove image \'{img_path}\'.'
            st.info(info_str)
            print(info_str)

            # remove the last line
            with open(osp.join(HIST_DIR, f'{user_name}.txt'),
                      mode='r') as file:
                all_lines = file.readlines()
            with open(osp.join(HIST_DIR, f'{user_name}.txt'),
                      mode='w') as file:
                file.writelines(all_lines[:-1])
            st.session_state['saved'] = False

    def show_results():
        # 1. init results dir if not inited
        for lab in idm.LABELS:
            os.makedirs(osp.join(RES_DIR, lab), exist_ok=True)

        # static results
        result_info = {
            lab: len([
                f for f in os.listdir(osp.join(RES_DIR, lab))
                if f.endswith('json')
            ])
            for lab in idm.LABELS
        }
        st.sidebar.write(
            f"Total files: {sum(result_info.values())} / {EACH_NEED * 5}")
        for lab in idm.LABELS:
            st.sidebar.write(
                f'LABEL  \'{lab}\': {result_info[lab]} / {EACH_NEED}')

    show_results()

    st.session_state['label'] = st.sidebar.selectbox("Label", idm.LABELS)

    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.button(label="Random OBJ", on_click=random_obj)
    with col2:
        st.button(label="Random BG", on_click=random_bg)

    # Main content: annotate images

    obj_name = st.session_state['image_info']['fg']
    mask_name = st.session_state['image_info']['mask']
    bg_name = st.session_state['image_info']['bg']

    fg_pair_im = MOCODPairImageManager(obj_name, mask_name)
    obj_crop = fg_pair_im.get_fg_crop_img()
    mask_crop = fg_pair_im.get_mask_crop_img()

    bg_im = MOCODImageManager(bg_name)
    bg_im.init_rects_from_obj(fg_pair_im)

    bg_img = bg_im.get_img()
    resized_bg_img = bg_im.resizing_img()
    resized_bg_rects = bg_im.get_resized_rects()
    rects_bg = st_img_label(resized_bg_img,
                            box_color="red",
                            rects=resized_bg_rects)

    if rects_bg:
        col_img1, col_img2, col_img3, col_img4 = st.columns(4)
        preview_bgs = bg_im.init_annotation(rects_bg)

        # draw image
        paste_res_dict = fg_pair_im.paste_on_bg(rects_bg, bg_im, model)
        with col_img1:
            st.write('Object')
            st.image(obj_crop)
        with col_img2:
            st.write('Mask')
            st.image(mask_crop)
        with col_img3:
            st.write('Object Resize')
            st.image(paste_res_dict['fg_resize'])
        with col_img4:
            st.write('Mask Resize')
            st.image(paste_res_dict['mask_resize'])
        st.image(paste_res_dict['bg_paste'])
        st.image(paste_res_dict['bg_refine'])

    else:
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.write('Object')
            st.image(obj_crop)
        with col_img2:
            st.write('Mask')
            st.image(mask_crop)

    col_save, col_undo = st.sidebar.columns(2)
    with col_save:
        st.sidebar.button(label='save',
                          on_click=save_results,
                          kwargs=dict(bbox=fg_pair_im.get_bbox(),
                                      img_paste=paste_res_dict['bg_paste'],
                                      img_refine=paste_res_dict['bg_refine']))
    with col_undo:
        st.sidebar.button(label='undo', on_click=undo)


if __name__ == "__main__":
    run()
