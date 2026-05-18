import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable 

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from PIL import Image
import cv2
from models.cbdnet import UNet
from models.network_scunet import SCUNet
from models.network_unet import UNetRes as DRUNet

os.environ["CUDA_VISIBLE_DEVICES"]="2"

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=79, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="Stitching_data_6", help="name of the dataset")
    parser.add_argument("--select_data", type=str, default="test", help="select data train and test")
    parser.add_argument("--data_save", type=str, default="Stitching_data_6/UNet", help="data save")
    parser.add_argument("--select_net", type=str, default="UNet", help="select notwork 'ResNet','UNet'")
    parser.add_argument("--model_name", type=str, default="Stitching_data_6", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--n_residual_blocks", type=int, default=15, help="number of residual blocks in generator")
    parser.add_argument("--img_height", type=int, default=512, help="size of image height")
    parser.add_argument("--img_width", type=int, default=512, help="size of image width")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument(
        "--sample_interval", type=int, default=1, help="interval between sampling of images from generators"
    )
    parser.add_argument("--checkpoint_interval", type=int, default=0, help="interval between model checkpoints")
    opt = parser.parse_args()
    print(opt)



    #os.makedirs("images/test/%s" % opt.dataset_name, exist_ok=True)
    os.makedirs("/home/gpudata/.jiangjiewei/xinyu/data/test_data_SCUNet_all_images_6/%s/%s" % (opt.data_save, opt.select_data+"_tmp",),exist_ok=True)



    input_shape = (opt.channels, opt.img_height, opt.img_width)

    # Calculate output of image discriminator (PatchGAN)
    patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)


    if opt.select_net == 'ResNet':
        generator = GeneratorResNet(input_shape, opt.n_residual_blocks)
    elif opt.select_net == 'UNet':
        #generator = DRUNet(in_nc=3,out_nc=3)
        generator = SCUNet()
        
        #generator = nn.DataParallel(generator)
        #generator =GeneratorUNet()
    else:
        print("Please select the training network")
        raise Exception ("please select the training network") 

    


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#判断是否GPU有效，否则使用CPU


    # if torch.cuda.device_count() >= 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")  # 还有问题
    #
    #     #generator = nn.DataParallel(generator)
    #
    #
    #
    generator = generator.cuda()





    if opt.epoch != 0:
        # Load pretrained models
        #generator.load_state_dict({k.replace('module.',''):v for k,v in torch.load("saved_models_SCU_24_3_7/%s/%s/generator_%d.pth" % (opt.model_name, opt.select_net, opt.epoch)).items()})
        generator.load_state_dict({k.replace('module.',''):v for k,v in torch.load("saved_models_SCU_24_3_7/%s/%s/generator_%d.pth" % (opt.model_name, opt.select_net, opt.epoch)).items()})




    # Configure dataloaders
    # transforms_ = [
    #     #transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
    #     #transforms.Resize((2048, 2048), Image.BICUBIC),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.57516927, 0.34860185, 0.23887351],  [0.21575285, 0.19056642, 0.17590816]),
    # ]
    # mean = torch.tensor([0.57516927, 0.34860185, 0.23887351])
    # std = torch.tensor([0.21575285, 0.19056642, 0.17590816])

    #transforms_ = [
        # transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
        # transforms.Resize((2048, 2048), Image.BICUBIC),
     #   transforms.ToTensor(),
        #transforms.Normalize([0.57516927, 0.34860185, 0.23887351], [0.21575285, 0.19056642, 0.17590816]),
      #  transforms.Normalize([0.5], [0.5]),
    #]
    
    mean = torch.tensor([0.5])
    std = torch.tensor([0.5])

    class CustomImageDataset(Dataset):
      def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_names = os.listdir(root_dir)

      def __len__(self):
        return len(self.file_names)

      def __getitem__(self, idx):
        img_name = self.file_names[idx]  # 只获取文件名，不包括绝对路径
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, img_name  # 返回图像和文件名作为字符串

    # 定义数据转换（transforms）和数据加载器
    data_transforms = transforms.Compose([
      transforms.Resize((512, 512)),
      transforms.ToTensor(),
      transforms.Normalize([0.5], [0.5]),
    ])

    # keratitis
    # normal
    # other

    test_dataset = CustomImageDataset(root_dir="/home/gpudata/.jiangjiewei/xinyu/data/lowquality_data_ori/other", transform=data_transforms)

    val_dataloader = DataLoader(
      test_dataset,
      batch_size=1,
      shuffle=False,
      num_workers=1,
    )
   
    # Tensor type
    Tensor = torch.cuda.FloatTensor if device else torch.FloatTensor


    def sample_images(name,imgs):
        """Saves a generated sample from the validation set"""
        # imgs = next(it)
        # real_A = Variable(imgs["A"].type(Tensor))
        # real_B = Variable(imgs["B"].type(Tensor))
        imgs = imgs.to('cuda')  # 如果模型在 GPU 上
        real_A = imgs
        fake_B = generator(real_A)  #real_A是模糊的 fake_B是经过GAN生成的
        #imgA = np.array(real_B.data)
        #imgB  = np.array(fake_B.data)
        # 如果有均值方差 就用下面的
        # real_A = real_A.detach().cpu().permute((0, 2, 3, 1)) * std + mean
        # real_A = real_A.permute((0, 3, 1, 2))
        # real_B = real_B.detach().cpu().permute((0, 2, 3, 1)) * std + mean
        # real_B = real_B.permute((0, 3, 1, 2))
        # fake_B = fake_B.detach().cpu().permute((0, 2, 3, 1)) * std + mean
        # fake_B = fake_B.permute((0, 3, 1, 2))
        """
        print((real_A.numpy().squeeze()*255))
        print((real_A.numpy().squeeze()*255).astype(np.int))
        """

        """
        PSNR = peak_signal_noise_ratio((real_A.numpy().squeeze()*255).astype(np.int).transpose((1,2,0)), (real_B.numpy().squeeze()*255).astype(np.int).transpose((1,2,0)))
        PSNR2 = peak_signal_noise_ratio((fake_B.numpy().squeeze()*255).astype(np.int).transpose((1,2,0)), (real_B.numpy().squeeze()*255).astype(np.int).transpose((1,2,0)))
        SSIM = structural_similarity((real_A.numpy().squeeze()*255).astype(np.int).transpose((1,2,0)), (real_B.numpy().squeeze()*255).astype(np.int).transpose((1,2,0)), multichannel=True)
        SSIM2 = structural_similarity((fake_B.numpy().squeeze()*255).astype(np.int).transpose((1,2,0)), (real_B.numpy().squeeze()*255).astype(np.int).transpose((1,2,0)), multichannel=True)
        print("before: PSNR:",PSNR,"SSIM:",SSIM)
        print("after: PSNR:",PSNR2,"SSIM:",SSIM2)
        """

        # print(name[0],'PSNR',PSNR,'SSIM',SSIM)
        #img_sample = torch.cat((real_B.data, fake_B.data), -1)
        # img_sample = fake_B
        # img_sample = (img_sample.cpu().permute((0,2,3,1))*std+mean)
        # img_sample = (img_sample.cpu().permute((0,3,1,2)))
        #img_sample = (img_sample.permute((0, 2, 3, 1)) * std + mean)
        #img_sample = (img_sample.permute((0, 3, 1, 2)))

        # print(img_sample.size())
        # h = img_sample.size()[2]
        # w = img_sample.size()[3]
        # img_true = img_sample.crop((0, 0, w / 2, h))
        # img_fake = img_sample.crop((w / 2, 0, w, h))
        # PSNR = peak_signal_noise_ratio(img_true, img_fake)
        # SSIM = structural_similarity(img_true, img_fake, multichannel=True)
        # print('PSNR',PSNR,'SSIM',SSIM)

        img_sample = torch.cat((real_A.data, fake_B.data), -2)
        #os.makedirs("images_DRUNet/%s/%s" % (opt.dataset_name, opt.select_net), exist_ok=True)
        #save_image(img_sample, "images_DRUNet/%s/%s/%s" % (opt.dataset_name, opt.select_net, name[0]),
        #           nrow=1, normalize=True)
        #save_image(img_sample, "test_data/%s/%s/%s.png" % (opt.data_save, opt.select_data, batches_done), nrow=1, normalize=True)
        # save_image(real_A, "test_data_DRUNet_1/%s/%s/%s_realA" % (opt.data_save, opt.select_data+"_tmp", name),nrow=1,normalize=True)
        # save_image(real_B, "test_data_DRUNet/%s/%s/%s_realB.png" % (opt.data_save, opt.select_data+"_tmp", name[0]),nrow=1,normalize=True)
        save_image(fake_B, "/home/gpudata/.jiangjiewei/xinyu/data/test_data_SCUNet_all_images_6/%s/%s/%s" % (opt.data_save, opt.select_data+"_tmp", name[0]),nrow=1,normalize=True)


        #imgA = Image.open("/home/gpudata/.jiangjiewei/xinyu/data/lowquality_data/other/%s" % (name[0]))
        #imgb = Image.open("test_data_DRUNet_1/%s/%s/%s" % (opt.data_save, opt.select_data+"_tmp", name[0]))
        #imgA = np.array(imgA)
        #imgB = np.array(imgB)
        #imgb = np.array(imgb)
        #desired_size = (512, 512)
        #imgA = imgA.resize(desired_size, Image.BILINEAR)
        #imgb = imgb.resize(desired_size, Image.BILINEAR)
        #PSNR = peak_signal_noise_ratio(imgA, imgb)
        #SSIM = structural_similarity(imgA, imgb, multichannel = True)
        #PSNR2 = peak_signal_noise_ratio(imgb, imgB)
        #SSIM2 = structural_similarity(imgb, imgB, multichannel = True)
        #print("PSNR:",PSNR,"SSIM:",SSIM)
        #print("after: PSNR:",PSNR2,"SSIM:",SSIM2)
        # w, h = img.size
        # img_true = np.array(img.crop((0, 0, w / 2, h)))  # 左上角到中间下面是A
        # img_fake = np.array(img.crop((w / 2, 0, w, h)))  # 中间上面到右下角是B
        # #print(img_true.shape)
        # PSNR = peak_signal_noise_ratio(img_true, img_fake)
        # SSIM = structural_similarity(img_true, img_fake, multichannel = True)
        #print(name[0],'PSNR',PSNR,'SSIM',SSIM)
        #return PSNR,PSNR2,SSIM,SSIM2
        # return PSNR,SSIM
    # # ----------
    #  Training
    # ----------


    # for epoch in range(opt.epoch-180, opt.n_epochs):
    # it=iter(val_dataloader)
    # for epoch in range(10):
    #     sample_images(epoch)
    #psnr = []
    #psnr2 = []
    #ssim = []
    #ssim2 = []
    #ss=0
    for i, (batch,name) in enumerate(val_dataloader):
        # if i==10:
        #     break
        # if batch['A'].size(2)*batch['A'].size(3)>2584*2000:
        #     ss += 1
        #     print("跳过: " + str(ss))
        #     continue
        print('name:%s', name)
        with torch.no_grad():
            #PSNR,PSNR2,SSIM,SSIM2 = sample_images(name,batch)
            sample_images(name,batch)
        # psnr2.append(PSNR2)
        #psnr.append(PSNR)
        #ssim.append(SSIM)
        # ssim2.append(SSIM2)
    #Psnr = sum(psnr)/len(psnr)
    #Ssim = sum(ssim)/len(ssim)
    # Psnr2 = sum(psnr2)/len(psnr2)
    # Ssim2 = sum(ssim2)/len(ssim2)
    #print(len(ssim),len(psnr))
    # print('psnr为',Psnr,'ssim为',Ssim,'psnr2为',Psnr2,'ssim2为',Ssim2)
    #print('psnr为',psnr,'ssim为',ssim)

