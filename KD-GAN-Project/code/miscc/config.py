from __future__ import division
from __future__ import print_function

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
from easydict import EasyDict as edict

import torch
import torchvision.transforms as transforms

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a KD-GAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/bird_KDGAN.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='0')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args

def parse_yaml(yaml_file):
    import yaml
    with open(yaml_file, 'r') as f:
        cfg = edict(yaml.load(f))
    return cfg

def merge_cfg_from_file(cfg, cfg_file):
    cfg_from_file = parse_yaml(cfg_file)
    for k, v in cfg_from_file.items():
        if k in cfg:
            if isinstance(v, dict):
                merge_cfg_from_file(cfg[k], v)
            else:
                cfg[k] = v
        else:
            cfg[k] = v

def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    return yaml_cfg

def cfg_from_list(cfg, args_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(args_list) % 2 == 0
    for k, v in zip(args_list[0::2], args_list[1::2]):
        key_list = k.split('.')
        d = cfg
        for subkey in key_list[:-1]:
            assert subkey in d
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
                type(value), type(d[subkey]))
        d[subkey] = value

def get_output_dir(args):
    path = os.path.join('output', args.cfg_file.split('/')[-1].split('.')[0])
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def prepare_data(data):
    imgs, captions, captions_lens, class_ids, keys = data

    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = \
        torch.sort(captions_lens, 0, True)

    real_imgs = []
    for i in range(len(imgs)):
        imgs[i] = imgs[i][sorted_cap_indices]
        if cfg.CUDA:
            real_imgs.append(Variable(imgs[i]).cuda())
        else:
            real_imgs.append(Variable(imgs[i]))

    captions = captions[sorted_cap_indices].squeeze()
    class_ids = class_ids[sorted_cap_indices].numpy()
    # sent_indices = sent_indices[sorted_cap_indices]
    keys = [keys[i] for i in sorted_cap_indices.numpy()]
    # print('keys', type(keys), keys[-1])  # list
    if cfg.CUDA:
        captions = Variable(captions).cuda()
        sorted_cap_lens = Variable(sorted_cap_lens).cuda()
    else:
        captions = Variable(captions)
        sorted_cap_lens = Variable(sorted_cap_lens)

    return [real_imgs, captions, sorted_cap_lens,
            class_ids, keys]

def get_imgs(img_path, imsize, bbox=None,
             transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)

    ret = []
    if cfg.GAN.B_DCGAN:
        ret = [normalize(img)]
    else:
        for i in range(cfg.TREE.BRANCH_NUM):
            # print(imsize[i])
            if i < (cfg.TREE.BRANCH_NUM - 1):
                re_img = transforms.Resize(imsize[i])(img)
            else:
                re_img = img
            ret.append(normalize(re_img))

    return ret

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)

def copy_G_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

# -------------------------------------------------------------------------
def define_optimizers(netG, netD):
    optimizersD = []
    num_Ds = len(netD)
    for i in range(num_Ds):
        opt = optim.Adam(netD[i].parameters(),
                         lr=cfg.TRAIN.DISCRIMINATOR_LR,
                         betas=(0.5, 0.999))
        optimizersD.append(opt)

    optimizerG = optim.Adam(netG.parameters(),
                            lr=cfg.TRAIN.GENERATOR_LR,
                            betas=(0.5, 0.999))

    return optimizerG, optimizersD

def save_model(netG, netD, epoch, model_dir):
    torch.save(
        netG.state_dict(),
        '%s/netG_%d.pth' % (model_dir, epoch))
    for i in range(len(netD)):
        torch.save(
            netD[i].state_dict(),
            '%s/netD%d_%d.pth' % (model_dir, i, epoch))
    print('Save G/D models.')

def save_img_results(imgs_tcpu, fake_imgs, att_maps, epoch,
                     count, image_dir, summary_writer):
    num = cfg.TRAIN.VIS_COUNT

    # Save images
    fig = plt.figure(figsize=(num, 4))
    real_img = imgs_tcpu[-1][0:num]
    if img_merge:
        merge_half = int(num / 2)
        img_merge(real_img[0:merge_half], fake_imgs[0:merge_half], fig, epoch, count, 0)
        img_merge(real_img[merge_half:], fake_imgs[merge_half:], fig, epoch, count, 1)
    else:
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

# -------------------------------------------------------------------------
def train(dataloader, netG, netD, text_encoder, optimizerG, optimizersD, state_epoch, batch_size, device):
    for epoch in range(state_epoch+1, cfg.TRAIN.MAX_EPOCH+1):
        start_t = time.time()
        
        for step, data in enumerate(dataloader, 0):
            # Prepare data
            imgs, captions, cap_lens, class_ids, keys = prepare_data(data)
            
            # Extract text embeddings
            hidden = text_encoder.init_hidden(batch_size)
            words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
            words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
            
            # Generate fake images
            noise = torch.randn(batch_size, cfg.GAN.Z_DIM).to(device)
            fake_imgs, _, mu, logvar = netG(noise, sent_emb, words_embs)
            
            # Update D network
            errD_total = 0
            for i in range(len(optimizersD)):
                optimizersD[i].zero_grad()
                errD = discriminator_loss(netD[i], imgs[i], fake_imgs[i], sent_emb)
                errD.backward()
                optimizersD[i].step()
                errD_total += errD
            
            # Update G network
            optimizerG.zero_grad()
            errG = generator_loss(netD, fake_imgs, sent_emb, mu, logvar)
            errG.backward()
            optimizerG.step()
            
            # Print log info
            if step % cfg.TRAIN.LOG_STEP == 0:
                elapsed = time.time() - start_t
                print(f'[{epoch}/{cfg.TRAIN.MAX_EPOCH}][{step}/{len(dataloader)}] '
                      f'Loss_D: {errD_total.item():.4f} Loss_G: {errG.item():.4f} '
                      f'Time: {elapsed:.4f}')
                start_t = time.time()
            
            # Save images
            if step % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:
                save_img_results(imgs, fake_imgs, None, epoch, step, cfg.TRAIN.IMAGE_DIR, None)
        
        # Save model
        if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:
            save_model(netG, netD, epoch, cfg.TRAIN.MODEL_DIR)
    
    # Save final model
    save_model(netG, netD, cfg.TRAIN.MAX_EPOCH, cfg.TRAIN.MODEL_DIR)

def discriminator_loss(netD, real_imgs, fake_imgs, sent_emb):
    batch_size = real_imgs.size(0)
    cond_real_logits = netD(real_imgs, sent_emb)
    cond_fake_logits = netD(fake_imgs.detach(), sent_emb)
    
    real_labels = torch.ones(batch_size, 1).to(real_imgs.device)
    fake_labels = torch.zeros(batch_size, 1).to(real_imgs.device)
    
    # Real/Fake loss
    real_loss = F.binary_cross_entropy_with_logits(cond_real_logits, real_labels)
    fake_loss = F.binary_cross_entropy_with_logits(cond_fake_logits, fake_labels)
    
    # Wrong caption loss
    wrong_idx = torch.randperm(batch_size).to(real_imgs.device)
    wrong_sent_emb = sent_emb[wrong_idx]
    wrong_logits = netD(real_imgs, wrong_sent_emb)
    wrong_loss = F.binary_cross_entropy_with_logits(wrong_logits, fake_labels)
    
    loss = real_loss + fake_loss + wrong_loss
    return loss

def generator_loss(netD, fake_imgs, sent_emb, mu, logvar):
    batch_size = fake_imgs[0].size(0)
    
    # Adversarial loss
    loss = 0
    for i in range(len(netD)):
        fake_logits = netD[i](fake_imgs[i], sent_emb)
        real_labels = torch.ones(batch_size, 1).to(fake_imgs[i].device)
        loss += F.binary_cross_entropy_with_logits(fake_logits, real_labels)
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss * cfg.TRAIN.SMOOTH.LAMBDA
    
    return loss + kl_loss

# -------------------------------------------------------------------------
def compute_inception_score(imgs, model, batch_size=32, splits=10):
    """Calculate the inception score of generated images"""
    N = len(imgs)
    
    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)
    
    # Get predictions
    preds = []
    for batch in dataloader:
        batch = batch.to(next(model.parameters()).device)
        with torch.no_grad():
            pred = F.softmax(model(batch), dim=1)
        preds.append(pred.cpu().numpy())
    
    preds = np.concatenate(preds, axis=0)
    
    # Now compute the mean kl-div
    split_scores = []
    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))
    
    return np.mean(split_scores), np.std(split_scores)

def entropy(p, q):
    """Compute KL divergence between p and q"""
    return np.sum(p * np.log(p / q))

# -------------------------------------------------------------------------
def build_models():
    # build model ############################################################
    text_encoder = BertEncoder(cfg, device)
    
    netG = G_NET()
    netG.apply(weights_init)
    netG = netG.to(device)
    
    netsD = []
    if cfg.TREE.BRANCH_NUM > 0:
        netsD.append(D_NET())
    if cfg.TREE.BRANCH_NUM > 1:
        netsD.append(D_NET())
    if cfg.TREE.BRANCH_NUM > 2:
        netsD.append(D_NET())
    
    for i in range(len(netsD)):
        netsD[i].apply(weights_init)
        netsD[i] = netsD[i].to(device)
    
    return [text_encoder, netG, netsD]

def define_optimizers(netG, netsD):
    optimizersD = []
    for i in range(len(netsD)):
        opt = optim.Adam(netsD[i].parameters(),
                         lr=cfg.TRAIN.DISCRIMINATOR_LR,
                         betas=(0.5, 0.999))
        optimizersD.append(opt)
    
    optimizerG = optim.Adam(netG.parameters(),
                            lr=cfg.TRAIN.GENERATOR_LR,
                            betas=(0.5, 0.999))
    
    return optimizerG, optimizersD

def load_network_stageI(netG, netD, text_encoder):
    if cfg.TRAIN.NET_G != '':
        state_dict = torch.load(cfg.TRAIN.NET_G)
        netG.load_state_dict(state_dict)
        print('Load G from: ', cfg.TRAIN.NET_G)
    
    if cfg.TRAIN.NET_D != '':
        for i in range(len(netD)):
            state_dict = torch.load(cfg.TRAIN.NET_D % (i))
            netD[i].load_state_dict(state_dict)
            print('Load D%d from: ' % (i), cfg.TRAIN.NET_D % (i))
    
    if cfg.TRAIN.NET_E != '':
        state_dict = torch.load(cfg.TRAIN.NET_E)
        text_encoder.load_state_dict(state_dict)
        print('Load E from: ', cfg.TRAIN.NET_E)
    
    return netG, netD, text_encoder

# -------------------------------------------------------------------------
cfg = edict()
cfg.DATASET_NAME = 'birds'
cfg.CONFIG_NAME = ''
cfg.DATA_DIR = ''
cfg.GPU_ID = '0'
cfg.CUDA = True
cfg.WORKERS = 4

cfg.TREE = edict()
cfg.TREE.BRANCH_NUM = 3

cfg.TRAIN = edict()
cfg.TRAIN.FLAG = True
cfg.TRAIN.BATCH_SIZE = 64
cfg.TRAIN.MAX_EPOCH = 600
cfg.TRAIN.SNAPSHOT_INTERVAL = 50
cfg.TRAIN.DISCRIMINATOR_LR = 2e-4
cfg.TRAIN.GENERATOR_LR = 2e-4
cfg.TRAIN.NET_G = ''
cfg.TRAIN.NET_D = ''
cfg.TRAIN.NET_E = ''
cfg.TRAIN.SMOOTH = edict()
cfg.TRAIN.SMOOTH.GAMMA1 = 4.0
cfg.TRAIN.SMOOTH.GAMMA2 = 5.0
cfg.TRAIN.SMOOTH.GAMMA3 = 10.0
cfg.TRAIN.SMOOTH.LAMBDA = 5.0

cfg.GAN = edict()
cfg.GAN.DF_DIM = 64
cfg.GAN.GF_DIM = 128
cfg.GAN.Z_DIM = 100
cfg.GAN.R_NUM = 2
cfg.GAN.B_DCGAN = False

cfg.TEXT = edict()
cfg.TEXT.EMBEDDING_DIM = 256
cfg.TEXT.CAPTIONS_PER_IMAGE = 10
cfg.TEXT.WORDS_NUM = 25