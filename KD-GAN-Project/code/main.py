import os
import errno
import numpy as np
from torch.nn import init
import torch
import torch.nn as nn
import torchvision.utils as vutils
from PIL import Image, ImageDraw, ImageFont
from copy import deepcopy
import skimage.transform
import ntpath
import matplotlib.pyplot as plt
from miscc.config import cfg

# For visualization ################################################
COLOR_DIC = {0:[128,64,128],  1:[244, 35,232],
             2:[70, 70, 70],  3:[102,102,156],
             4:[190,153,153], 5:[153,153,153],
             6:[250,170, 30], 7:[220, 220, 0],
             8:[107,142, 35], 9:[152,251,152],
             10:[70,130,180], 11:[220,20, 60],
             12:[255, 0, 0],  13:[0, 0, 142],
             14:[119,11, 32], 15:[0, 60,100],
             16:[0, 80, 100], 17:[0, 0, 230],
             18:[0,  0, 70],  19:[0, 0,  0]}
FONT_MAX = 50

def drawCaption(convas, captions, ixtoword, vis_size, off1=2, off2=2):
    num = captions.size(0)
    img_txt = Image.fromarray(convas)
    # get a font
    # fnt = None  # ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 50)
    fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 50)
    # get a drawing context
    d = ImageDraw.Draw(img_txt)
    sentence_list = []
    for i in range(num):
        cap = captions[i].data.cpu().numpy()
        sentence = []
        for j in range(len(cap)):
            if cap[j] == 0:
                break
            word = ixtoword[cap[j]].encode('ascii', 'ignore').decode('ascii')
            d.text(((j + off1) * (vis_size + off2), i * FONT_MAX), '%d:%s' % (j, word[:6]),
                   font=fnt, fill=(255, 255, 255, 255))
            sentence.append(word)
        sentence_list.append(sentence)
    return img_txt, sentence_list

def build_super_images(real_imgs, captions, ixtoword,
                       attn_maps, att_sze, lr_imgs=None,
                       batch_size=cfg.TRAIN.BATCH_SIZE,
                       max_word_num=cfg.TEXT.WORDS_NUM):
    nvis = 8
    real_imgs = real_imgs[:nvis]
    if lr_imgs is not None:
        lr_imgs = lr_imgs[:nvis]
    if att_sze == 17:
        vis_size = att_sze * 16
    else:
        vis_size = real_imgs.size(2)

    text_convas = \
        np.ones([batch_size * FONT_MAX,
                 (max_word_num + 2) * (vis_size + 2), 3],
                dtype=np.uint8)

    for i in range(max_word_num):
        istart = (i + 2) * (vis_size + 2)
        iend = (i + 3) * (vis_size + 2)
        text_convas[:, istart:iend, :] = COLOR_DIC[i]

    real_imgs = \
        nn.Upsample(size=(vis_size, vis_size), mode='bilinear')(real_imgs)
    # [-1, 1] --> [0, 1]
    real_imgs.add_(1).div_(2).mul_(255)
    real_imgs = real_imgs.data.numpy()
    # b x c x h x w --> b x h x w x c
    real_imgs = np.transpose(real_imgs, (0, 2, 3, 1))
    pad_sze = real_imgs.shape[0] * vis_size
    middle_pad = np.zeros([pad_sze, 2, 3])
    post_pad = np.zeros([pad_sze, vis_size, 3])
    if lr_imgs is not None:
        lr_imgs = \
            nn.Upsample(size=(vis_size, vis_size), mode='bilinear')(lr_imgs)
        # [-1, 1] --> [0, 1]
        lr_imgs.add_(1).div_(2).mul_(255)
        lr_imgs = lr_imgs.data.numpy()
        # b x c x h x w --> b x h x w x c
        lr_imgs = np.transpose(lr_imgs, (0, 2, 3, 1))

    # batch x seq_len x 17 x 17 --> batch x 1 x 17 x 17
    seq_len = max_word_num
    img_set = []
    num = nvis  # len(attn_maps)

    text_map, sentences = \
        drawCaption(text_convas, captions, ixtoword, vis_size)
    text_map = np.asarray(text_map).astype(np.uint8)

    bUpdate = 1
    for i in range(num):
        attn = attn_maps[i].cpu().view(1, -1, att_sze, att_sze)
        # --> 1 x 1 x 17 x 17
        attn_max = attn.max(dim=1, keepdim=True)
        attn = torch.cat([attn_max[0], attn], 1)
        #
        attn = attn.view(-1, 1, att_sze, att_sze)
        attn = attn.repeat(1, 3, 1, 1).data.numpy()
        # n x c x h x w --> n x h x w x c
        attn = np.transpose(attn, (0, 2, 3, 1))
        num_attn = attn.shape[0]
        #
        img = real_imgs[i]
        if lr_imgs is None:
            lrI = img
        else:
            lrI = lr_imgs[i]
        row = [lrI, middle_pad]
        row_merge = np.concatenate(row, 1)
        row = [text_map, middle_pad]
        row_txt = np.concatenate(row, 1)
        if num_attn == 1:
            row = [attn[0], middle_pad, img]
        else:
            row = [attn[0], middle_pad, img]
            for j in range(num_attn-1):
                row.append(post_pad)
                row.append(attn[j+1])
        row = np.concatenate(row, 1)
        txt = np.concatenate([row_txt, row_merge], 0)
        img_set.append(np.concatenate([txt, row], 0))
    img_set = np.concatenate(img_set, 0)
    img_set = img_set.astype(np.uint8)
    return img_set, sentences

def build_super_images2(real_imgs, captions, cap_lens, ixtoword,
                        attn_maps, att_sze, vis_size=256, topK=5):
    batch_size = real_imgs.size(0)
    max_word_num = np.max(cap_lens)
    text_convas = np.ones([batch_size * FONT_MAX,
                           max_word_num * (vis_size + 2), 3],
                           dtype=np.uint8)
    
    real_imgs = \
        nn.Upsample(size=(vis_size, vis_size), mode='bilinear')(real_imgs)
    # [-1, 1] --> [0, 1]
    real_imgs.add_(1).div_(2).mul_(255)
    real_imgs = real_imgs.data.numpy()
    # b x c x h x w --> b x h x w x c
    real_imgs = np.transpose(real_imgs, (0, 2, 3, 1))

    # Draw text on canvas
    for i in range(batch_size):
        cap_len = cap_lens[i]
        cap = captions[i].data.cpu().numpy()
        kept_cap = cap[:cap_len]
        if len(kept_cap) > topK:
            kept_cap = kept_cap[:topK]
        sent = []
        for j in range(len(kept_cap)):
            word = ixtoword[kept_cap[j]]
            sent.append(word)
            width = (j + 1) * (vis_size + 2)
            for k in range(FONT_MAX):
                text_convas[i * FONT_MAX + k, width - vis_size:width, :] = COLOR_DIC[j]
        
        # Draw attention maps
        attn = attn_maps[i].cpu().view(1, -1, att_sze, att_sze)
        # --> 1 x topK x att_sze x att_sze
        attn = attn[:, :cap_len, :, :]
        if len(attn[0]) > topK:
            attn = attn[:, :topK, :, :]
        
        # Resize attention maps to the same size as the image
        attn_img = []
        for j in range(len(kept_cap)):
            one_map = attn[0, j].view(att_sze, att_sze)
            one_map = one_map.data.numpy()
            one_map = one_map / np.max(one_map)  # normalize
            one_map = np.repeat(one_map[:, :, np.newaxis], 3, axis=2)
            one_map = skimage.transform.resize(one_map, (vis_size, vis_size))
            attn_img.append(one_map)
        
        # Combine attention maps with real image
        img_set = []
        for j in range(len(attn_img)):
            one_img = real_imgs[i]
            # Overlay attention map on real image
            att = attn_img[j]
            att = att * 0.8 + 0.2  # make the attention map more visible
            one_img = one_img * att
            img_set.append(one_img)
        
        # Combine all images into one
        if len(img_set) > 0:
            img_set = np.concatenate(img_set, 1)
            img_set = np.concatenate([img_set, text_convas[i * FONT_MAX:(i + 1) * FONT_MAX]], 0)
        else:
            img_set = text_convas[i * FONT_MAX:(i + 1) * FONT_MAX]
        
        img_set = img_set.astype(np.uint8)
        # Save or return the image
        return img_set, sent

def save_img_results(imgs_tcpu, fake_imgs, att_maps, epoch,
                     count, image_dir, summary_writer):
    num = cfg.TRAIN.VIS_COUNT

    # Save images
    fig = plt.figure(figsize=(num, 4))
    real_img = imgs_tcpu[-1][0:num]
    grid_real = vutils.make_grid(real_img, normalize=True)
    grid_fake = []
    for i in range(len(fake_imgs)):
        fake_img = fake_imgs[i][0:num]
        grid_fake.append(vutils.make_grid(fake_img, normalize=True))
        
    fig.add_subplot(1, 2, 1)
    plt.imshow(grid_real.permute(1, 2, 0))
    plt.axis('off')
    for i in range(len(grid_fake)):
        fig.add_subplot(1, 2, i+2)
        plt.imshow(grid_fake[i].permute(1, 2, 0))
        plt.axis('off')
    
    # Save attention maps
    if att_maps is not None:
        fig_attn = plt.figure(figsize=(6, 4))
        num_maps = len(att_maps)
        img_set = real_img[0:num]
        for i in range(num_maps):
            att_map = att_maps[i][0:num]
            for j in range(num):
                fig_attn.add_subplot(num, num_maps+1, i*num+j+1)
                plt.imshow(att_map[j].detach().cpu(), cmap='viridis')
                plt.axis('off')
                plt.title(f'Attn {i+1}')
                
                if i == 0:
                    fig_attn.add_subplot(num, num_maps+1, (num_maps)*num+j+1)
                    plt.imshow(img_set[j].permute(1, 2, 0))
                    plt.axis('off')
                    plt.title('Image')
        
        plt.tight_layout()
        plt.savefig('%s/attention_maps_%d_%d.png' % (image_dir, epoch, count))
        plt.close(fig_attn)
    
    plt.tight_layout()
    plt.savefig('%s/epoch_%d_step_%d.png' % (image_dir, epoch, count))
    plt.close(fig)