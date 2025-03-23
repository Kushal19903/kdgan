import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Upsample
import numpy as np

from miscc.config import cfg
from miscc.bert_encoder import BertEncoder

# Generator network
class CA_NET(nn.Module):
    # some code is modified from vae examples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)
    def __init__(self):
        super(CA_NET, self).__init__()
        self.t_dim = cfg.TEXT.EMBEDDING_DIM
        self.c_dim = cfg.GAN.Z_DIM
        self.fc = nn.Linear(self.t_dim, self.c_dim * 2, bias=True)
        self.relu = nn.ReLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if cfg.CUDA:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar


class INIT_STAGE_G(nn.Module):
    def __init__(self, ngf):
        super(INIT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.in_dim = cfg.GAN.Z_DIM + cfg.TEXT.EMBEDDING_DIM
        self.define_module()

    def define_module(self):
        in_dim = self.in_dim
        ngf = self.gf_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, ngf * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4 * 2),
            nn.ReLU(True))

        self.upsample1 = Upsample(scale_factor=2, mode='nearest')
        self.upsample2 = Upsample(scale_factor=2, mode='nearest')

        self.conv1 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

    def forward(self, z_code, c_code):
        c_z_code = torch.cat((c_code, z_code), 1)
        # state size ngf x 4 x 4
        out_code = self.fc(c_z_code)
        out_code = out_code.view(-1, self.gf_dim * 2, 4, 4)
        # state size ngf/2 x 8 x 8
        out_code = self.upsample1(out_code)
        out_code = self.conv1(out_code)
        # state size ngf/4 x 16 x 16
        out_code = self.upsample2(out_code)
        out_code = self.conv2(out_code)
        return out_code


class NEXT_STAGE_G(nn.Module):
    def __init__(self, ngf, num_residual=cfg.GAN.R_NUM):
        super(NEXT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.ef_dim = cfg.TEXT.EMBEDDING_DIM
        self.num_residual = num_residual
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(self.num_residual):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = self.gf_dim
        efg = self.ef_dim

        self.att = ATT_NET(ngf, efg)
        self.residual = self._make_layer(ResBlock, ngf * 2)
        self.upsample = Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

    def forward(self, h_code, c_code, word_embs, mask):
        self.att.applyMask(mask)
        c_code, att = self.att(h_code, word_embs)
        h_c_code = torch.cat((h_code, c_code), 1)
        out_code = self.residual(h_c_code)
        out_code = self.upsample(out_code)
        out_code = self.conv(out_code)
        return out_code, att


class GET_IMAGE_G(nn.Module):
    def __init__(self, ngf):
        super(GET_IMAGE_G, self).__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(
            nn.Conv2d(ngf, 3, 3, 1, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img


class G_NET(nn.Module):
    def __init__(self):
        super(G_NET, self).__init__()
        self.gf_dim = cfg.GAN.GF_DIM
        self.define_module()

    def define_module(self):
        self.ca_net = CA_NET()

        if cfg.TREE.BRANCH_NUM > 0:
            self.h_net1 = INIT_STAGE_G(self.gf_dim * 16)
            self.img_net1 = GET_IMAGE_G(self.gf_dim)
        if cfg.TREE.BRANCH_NUM > 1:
            self.h_net2 = NEXT_STAGE_G(self.gf_dim)
            self.img_net2 = GET_IMAGE_G(self.gf_dim)
        if cfg.TREE.BRANCH_NUM > 2:
            self.h_net3 = NEXT_STAGE_G(self.gf_dim)
            self.img_net3 = GET_IMAGE_G(self.gf_dim)

    def forward(self, z_code, text_embedding, word_embeddings, mask):
        c_code, mu, logvar = self.ca_net(text_embedding)
        fake_imgs = []
        att_maps = []
        if cfg.TREE.BRANCH_NUM > 0:
            h_code1 = self.h_net1(z_code, c_code)
            fake_img1 = self.img_net1(h_code1)
            fake_imgs.append(fake_img1)
        if cfg.TREE.BRANCH_NUM > 1:
            h_code2, att1 = self.h_net2(h_code1, c_code, word_embeddings, mask)
            fake_img2 = self.img_net2(h_code2)
            fake_imgs.append(fake_img2)
            att_maps.append(att1)
        if cfg.TREE.BRANCH_NUM > 2:
            h_code3, att2 = self.h_net3(h_code2, c_code, word_embeddings, mask)
            fake_img3 = self.img_net3(h_code3)
            fake_imgs.append(fake_img3)
            att_maps.append(att2)

        return fake_imgs, att_maps, mu, logvar


class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channel_num, channel_num, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channel_num),
            nn.ReLU(True),
            nn.Conv2d(channel_num, channel_num, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channel_num)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out


class ATT_NET(nn.Module):
    def __init__(self, idf, cdf):
        super(ATT_NET, self).__init__()
        self.conv_context = nn.Conv2d(cdf, idf, kernel_size=1, stride=1, padding=0)
        self.conv_sentence = nn.Conv2d(idf, 1, kernel_size=1, stride=1, padding=0)
        self.conv_word = nn.Conv2d(idf, 1, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Conv2d(idf, idf, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask  # batch x seq_len

    def forward(self, h_code, word_embs):
        """
            h_code: batch x idf x ih x iw (queryL=ihxiw)
            word_embs: batch x cdf x sourceL (sourceL=seq_len)
            mask: batch x sourceL
            output: batch x idf x queryL
        """
        batch_size = word_embs.size(0)
        sourceL = word_embs.size(2)
        queryL = h_code.size(2) * h_code.size(3)

        # Sentence-level attention
        context = self.conv_context(word_embs)  # batch x idf x sourceL
        context = context.view(batch_size, -1, sourceL)
        attn = torch.bmm(context.transpose(1, 2), h_code.view(batch_size, -1, queryL))  # batch x sourceL x queryL
        attn = attn.view(batch_size, -1, h_code.size(2), h_code.size(3))
        attn = self.conv_sentence(attn)  # batch x 1 x ih x iw
        attn = attn.view(batch_size, -1, queryL)
        attn = self.softmax(attn)  # batch x 1 x queryL
        
        # Word-level attention
        word_attn = self.conv_word(context.view(batch_size, -1, 1, sourceL))  # batch x 1 x 1 x sourceL
        word_attn = word_attn.view(batch_size, -1, sourceL)  # batch x 1 x sourceL
        if self.mask is not None:
            word_attn.data.masked_fill_(self.mask.unsqueeze(1).bool(), -float('inf'))
        word_attn = self.softmax(word_attn)  # batch x 1 x sourceL
        
        # Apply attention
        weightedContext = torch.bmm(word_embs.view(batch_size, -1, sourceL), word_attn.transpose(1, 2))  # batch x cdf x 1
        weightedContext = weightedContext.view(batch_size, -1, 1, 1)
        weightedContext = weightedContext.repeat(1, 1, h_code.size(2), h_code.size(3))
        
        h_codeView = h_code.view(batch_size, -1, queryL)
        attnView = attn.view(batch_size, -1, queryL)
        weightedContext = torch.bmm(h_codeView, attnView.transpose(1, 2))  # batch x idf x 1
        weightedContext = weightedContext.view(batch_size, -1, 1, 1)
        
        h_plus_c = h_code + weightedContext
        h_plus_c = self.conv(h_plus_c)
        h_plus_c = self.sigmoid(h_plus_c)
        
        return h_plus_c, attn.view(batch_size, -1, h_code.size(2), h_code.size(3))


# Discriminator network
class D_NET(nn.Module):
    def __init__(self):
        super(D_NET, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.TEXT.EMBEDDING_DIM
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        
        self.img_code_s16 = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*2) x 16 x 16
            nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*4) x 8 x 8
            nn.Conv2d(ndf*4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
            # state size (ndf * 8) x 4 x 4
        )
        
        self.img_code_s32 = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*2) x 32 x 32
            nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*4) x 16 x 16
            nn.Conv2d(ndf*4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*8) x 8 x 8
            nn.Conv2d(ndf*8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True)
            # state size (ndf*16) x 4 x 4
        )
        
        self.img_code_s64 = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 128 x 128
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*2) x 64 x 64
            nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*4) x 32 x 32
            nn.Conv2d(ndf*4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*8) x 16 x 16
            nn.Conv2d(ndf*8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*16) x 8 x 8
            nn.Conv2d(ndf*16, ndf * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 32),
            nn.LeakyReLU(0.2, inplace=True)
            # state size (ndf*32) x 4 x 4
        )
        
        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1),
            nn.Sigmoid()
        )
        
        self.logits32 = nn.Sequential(
            nn.Conv2d(ndf * 16, 1, kernel_size=4, stride=1),
            nn.Sigmoid()
        )
        
        self.logits64 = nn.Sequential(
            nn.Conv2d(ndf * 32, 1, kernel_size=4, stride=1),
            nn.Sigmoid()
        )
        
        self.jointConv = nn.Sequential(
            nn.Conv2d(ndf * 8 + efg, ndf * 8, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.jointConv32 = nn.Sequential(
            nn.Conv2d(ndf * 16 + efg, ndf * 16, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.jointConv64 = nn.Sequential(
            nn.Conv2d(ndf * 32 + efg, ndf * 32, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x_var, sent_emb):
        x_code = self.img_code_s16(x_var)  # batch x ndf*8 x 4 x 4
        sent_emb = sent_emb.view(-1, self.ef_dim, 1, 1)
        sent_emb = sent_emb.repeat(1, 1, 4, 4)
        x_c_code = torch.cat((x_code, sent_emb), 1)
        x_c_code = self.jointConv(x_c_code)
        output = self.logits(x_c_code)
        
        if x_var.size(2) >= 64:  # 64x64
            x_code32 = self.img_code_s32(x_var)
            sent_emb32 = sent_emb.repeat(1, 1, 4, 4)
            x_c_code32 = torch.cat((x_code32, sent_emb32), 1)
            x_c_code32 = self.jointConv32(x_c_code32)
            output32 = self.logits32(x_c_code32)
        else:
            output32 = None
            
        if x_var.size(2) >= 128:  # 128x128
            x_code64 = self.img_code_s64(x_var)
            sent_emb64 = sent_emb.repeat(1, 1, 4, 4)
            x_c_code64 = torch.cat((x_code64, sent_emb64), 1)
            x_c_code64 = self.jointConv64(x_c_code64)
            output64 = self.logits64(x_c_code64)
        else:
            output64 = None
            
        return [output, output32, output64]