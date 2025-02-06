import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
from PIL import Image

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets_224 import *

import torch.nn as nn
import torch.nn.functional as F
#from cbdnet import UNet,Network
from models.network_scunet import SCUNet
import pytorch_ssim

from skimage.metrics import structural_similarity as ssim
from torchvision.models import vgg19

import torch
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"

# # 定义感知损失
# class PerceptualLoss(nn.Module):
#     def __init__(self, device):
#         super(PerceptualLoss, self).__init__()
#         self.vgg = vgg19(pretrained=True).features.to(device).eval()
#         self.criterion = nn.MSELoss()
#         self.device = device
#
#     def forward(self, input, target):
#         input_vgg = self.vgg(input)
#         target_vgg = self.vgg(target)
#         loss = 0.0
#         for i in range(len(input_vgg)):
#             loss += self.criterion(input_vgg[i], target_vgg[i].detach())
#         return loss

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid() 
        )

    def forward(self, img):
        return self.model(img)

    def forward(self, img):
        return self.model(img)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=80, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="Stitching_data_6", help="name of the dataset")
    parser.add_argument("--select_net", type=str, default="UNet", help="select notwork 'ResNet','UNet'")   
    parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--n_residual_blocks", type=int, default=15, help="number of residual blocks in generator")
    parser.add_argument("--img_height", type=int, default=512, help="size of image height")
    parser.add_argument("--img_width", type=int, default=512, help="size of image width")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument(
        "--sample_interval", type=int, default=1000, help="interval between sampling of images from generators"
    )
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between model checkpoints")
    opt = parser.parse_args()
    print(opt)


    input_shape = (opt.channels, opt.img_height, opt.img_width)

    # Loss functions
    criterion_GAN = torch.nn.MSELoss()
    criterion_pixelwise = torch.nn.L1Loss()
    ssim_loss = pytorch_ssim.SSIM()

    #-------------------新增尝试
    # 在模型训练之前定义损失函数和优化�?
    adversarial_loss = torch.nn.BCELoss()

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # perceptual_loss = PerceptualLoss(device)
    #判别�?
    discriminator = Discriminator()

    generator = SCUNet()

    # 为生成器和判别器定义优化�?
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    
    generator = nn.DataParallel(generator)
    os.makedirs("saved_models_SCU_24_3_11/%s/%s" % (opt.dataset_name, opt.select_net), exist_ok=True)
    os.makedirs("images_SCU_24_3_11/%s/%s" % (opt.dataset_name, opt.select_net), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#�ж��Ƿ�GPU��Ч������ʹ��CPU

    
    generator = generator.cuda()
    discriminator = discriminator.cuda()


    # Configure dataloaders
    transforms_ = [
        #transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
        transforms.ToTensor(),
        #transforms.Normalize([0.57516927, 0.34860185, 0.23887351],  [0.21575285, 0.19056642, 0.17590816]),
        transforms.Normalize([0.5], [0.5]),
    ]

    dataloader = DataLoader(
        ImageDataset("/home/gpudata/.jiangjiewei/xinyu/data/create_lowquality_data", transforms_=transforms_, mode="train"),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )

    val_dataloader = DataLoader(
        ImageDataset("/home/gpudata/.jiangjiewei/xinyu/data/create_lowquality_data", transforms_=transforms_, mode="val"),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=0,
    )

    # Tensor type
    Tensor = torch.cuda.FloatTensor if device else torch.FloatTensor
    #Tensor = torch.FloatTensor


    def sample_images(batches_done):
        """Saves a generated sample from the validation set"""
        imgs = next(iter(val_dataloader))
        real_A = Variable(imgs["A"].type(Tensor))
        real_B = Variable(imgs["B"].type(Tensor))
        fake_B = generator(real_B)
        img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
        save_image(img_sample, "images_SCU_24_3_11/%s/%s/%s.png" % (opt.dataset_name, opt.select_net, batches_done),
                   nrow=2, normalize=True)


    # ----------
    #  Training
    # ----------

    prev_time = time.time()

    for epoch in range(opt.epoch, opt.n_epochs):
        ss=0
        #for i, (batch,_) in enumerate(dataloader):
        for i, batch in enumerate(dataloader):
            real_A = Variable(batch["A"].type(Tensor))  # 清晰
            real_B = Variable(batch["B"].type(Tensor))  # 模糊

            optimizer_G.zero_grad()
            # print(real_B.shape)
            # GAN loss
            fake_B = generator(real_B)  # 生成
            # 对真实清晰图像进行插值，使其尺寸变为 [2, 3, 29, 29]
            # real_A_resized = F.interpolate(real_A, size=(29, 29), mode='bilinear', align_corners=False)
            # # 将张量的形状更改为 [2, 1, 29, 29]
            # # real_A_resized = tf.reshape(real_A_resized, [2, 1, 29, 29])
            # real_A_resized_channel_1 = real_A_resized[:, :1, :29, :29]
            # print(real_A_resized_channel_1.shape)

            # 计算 SSIM 损失
            ssim_loss_1 = 1 - ssim_loss(fake_B, real_A)

            # # 计算对抗性损失
            # pred_fake = discriminator(fake_B)
            # valid = Variable(Tensor(fake_B.size(0), 1).fill_(1.0), requires_grad=False)
            # valid = valid.unsqueeze(2).unsqueeze(3).expand_as(pred_fake)
            # adversarial_loss_G = adversarial_loss(pred_fake, valid)
            # 计算对抗性损失
            pred_fake = discriminator(fake_B)
            # print(pred_fake.shape)
            valid = Variable(Tensor(fake_B.size(0), 1).fill_(1.0), requires_grad=False)
            valid = valid.unsqueeze(2).unsqueeze(3).expand_as(pred_fake)
            adversarial_loss_G = adversarial_loss(pred_fake, valid)

            # Pixel-wise loss
            loss_pixel = F.mse_loss(fake_B, real_A)

            # Total loss for generator
            lambda_pixel = 100  # 设置像素级别损失的权重
            total_loss_G = adversarial_loss_G + lambda_pixel * loss_pixel + 10 * ssim_loss_1
            # print('adversarial_loss_G:%f , adversarial_loss_G:%f ', adversarial_loss_G, ssim_loss_1)

            total_loss_G.backward(retain_graph=True)
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()

            # Real loss
            pred_real = discriminator(real_A)
            loss_real = adversarial_loss(pred_real, valid)

            # Fake loss
            pred_fake = discriminator(real_B.detach())
            fake = torch.zeros(pred_fake.size(), device=device)
            loss_fake = adversarial_loss(pred_fake, fake)

            # Total loss for discriminator
            total_loss_D = (loss_real + loss_fake) / 2

            total_loss_D.backward()  # 设置 retain_graph=True
            optimizer_D.step()

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            loss_value_G = total_loss_G.mean().item()
            loss_value_D = total_loss_D.mean().item()
            # loss_value = abs(loss_value)  # ����ʹ�� torch.abs()

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d]  [G loss: %f,] [D loss: %f,]  ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    #loss_D.mean().item(),
                    # loss.mean().item(),
                    loss_value_G,
                    loss_value_D,
                    #loss_pixel.mean().item(),
                    #loss_GAN.mean().item(),
                    time_left,
                )
            )

            # If at sample interval save image
            if batches_done % opt.sample_interval == 0:
               sample_images(batches_done)

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), "saved_models_SCU_24_3_11/%s/%s/generator_%d.pth" % (opt.dataset_name, opt.select_net, epoch))
            #torch.save(discriminator.state_dict(), "saved_models_50/%s/%s/discriminator_%d.pth" % (opt.dataset_name, opt.select_net, epoch))
