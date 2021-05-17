# import numpy as np
import torch
# import torch.nn as nn
from torch.utils.data import *
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as tf
# import os
# from torch.optim import RMSprop
from torchvision.utils import make_grid
# from pylab import plt
# import Mod.res_block as resblock
# import Liblinks.utis as f_s
# from Liblinks.sn_lib import *
from Liblinks.resblock import *
from torch.nn.utils import spectral_norm

# 基本配置
path = os.path.abspath('../../dataset/GetChu_aligned2/')
# num_chann=np.array([64*8,64*4,64*2,64,64])
# dnum_chann=np.array([64,64*2,64*4,64*8])
img_size = 128
noise_size = 128
# im_chann=3
bat_size = 8
tr_epoch = 2500
worker = 2
# clamp_num=0.01
learningr = 0.0002
times_batch = 8
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
gpu = True
back_version = ' v1016CP '  # 用官方SN  生成器用SN  判别器官方SN attention放在64*64层  bilinear上采样方式
version_p = ' v1000b  '  #  bilinear上采样方式  生成器有权重衰减




transform_todata=transforms.Compose([transforms.RandomCrop(img_size),transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)])




class Gan_data(Dataset):
    def __init__(self, root, transform, augment=None,img_size=256):
        self.image_files = np.array([x.path for x in os.scandir(root) if x.name.endswith(".jpg")])
        self.transform = transform
        self.augment = augment
        self.img_size=img_size

    def __getitem__(self, index):
        img_data = Image.open(self.image_files[index])
        dimw, dimh = img_data.size
        div_num = min(dimw, dimh)  # 最小维度
        ratio = div_num / self.img_size
        scale_dimw = int(dimw / ratio)
        scale_dimh = int(dimh / ratio)
        te = transforms.functional.resize(img_data, [scale_dimh, scale_dimw])
        pic_data = self.transform(te)
        return pic_data


    def __len__(self):
        return len(self.image_files)


#    def __len__(self):
#        return len(self.image_files)


# 生成器定义

class Generat(nn.Module):
    def __init__(self, channel=64, dim_z=128, bottom_width=4, n_cls=0):
        self.bottom_width = bottom_width
        self.dim_z = dim_z
        self.num_classes = n_cls
        super(Generat, self).__init__()
        self.l1 = nn.Linear(dim_z, (bottom_width ** 2) * channel * 16)
        # print(self.l1)
        self.b1 = SNGBlock(channel * 16, channel * 16, upsample=True, n_cls=n_cls)  #8*8
        self.b2 = SNGBlock(channel * 16, channel * 8, upsample=True, n_cls=n_cls)   #16*16
        #self.b3 = GBlock2(channel * 8, channel * 8, upsample=True, n_cls=n_cls)  #32*32
        self.b4 = SNGBlock(channel * 8, channel * 4, upsample=True, n_cls=n_cls)   #64*64

        self.b5 = SNGBlock(channel * 4, channel * 2, upsample=True, n_cls=n_cls)   #128*128
        self.atten = SNAttentionBlock(channel * 2)
        self.b6 = SNGBlock(channel * 2, channel, upsample=True, n_cls=n_cls)
        self.bn1 = nn.BatchNorm2d(channel)
        self.con1 = spectral_norm(nn.Conv2d(channel, 3, kernel_size=3, stride=1, padding=1))
        self.actf = nn.ReLU(inplace=True)
        self.tanhf = nn.Tanh()

    def forward(self, x):
        # if z is None:
        #    z = torch.randn(bat_size, self.dim_z).cuda()
        # if y is None:
        #    np_y = np.random.randint(0, self.num_classes, size=bat_size)
        #    y = torch.from_numpy(np_y).type(torch.LongTensor).cuda()
        # x = self.l1(x)
        # x = x.view(x.size(0), -1, self.bottom_width, self.bottom_width)
        # print(x.size())
        # x = self.b1(x)
        # x = self.b2(x)
        # x = self.b3(x)
        # x = self.b4(x)
        # x = self.b5(x)
        # x = self.b6(x)
        # x = self.bn1(x)
        # x = self.actf(x)
        # x = self.con1(x)
        # x = self.tanhf(x)
        x = self.l1(x)
        x = x.view(x.size(0), -1, self.bottom_width, self.bottom_width)
        x = self.b1(x)
        x = self.b2(x)
        #x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.atten(x)
        x = self.b6(x)
        x = self.bn1(x)
        x = self.actf(x)
        x = self.con1(x)
        x = self.tanhf(x)
        return x


class Discr_net(nn.Module):  # 用这个
    def __init__(self, channel):
        super(Discr_net, self).__init__()
        self.CNN = nn.Sequential(SNDOptimizedBlock(3, channel),  #64*64
                                 SNAttentionBlock(channel),
                                 SNDBlock(channel, channel * 2, downsample=True),  #32*32
                                 SNDBlock(channel * 2, channel * 4, downsample=True),
                                 SNDBlock(channel * 4, channel * 8, downsample=True),
                                 SNDBlock(channel * 8, channel * 16, downsample=True),
                                 SNDBlock(channel * 16, channel * 16))
        #self.FC = nn.Sequential(SNLinear(channel * 16, 1024),
        #                        nn.ReLU(inplace=True)
        #                        )
        self.FD = nn.Sequential(nn.ReLU(inplace=True), SNLinear(channel * 16, 1))
        #SNLinear(channel * 16, 1)) spectral_norm(nn.Linear(channel * 16, 1))

    def forward(self, x):
        x = self.CNN(x)
        x = torch.sum(x, dim=(2, 3))
        x = x.view(x.size(0), -1)
        output = self.FD(x)

        return output


# 定义对象
discriminator = Discr_net(64)
generator = Generat()
loss_hinge = HingeLoss()

optimizerD = torch.optim.Adam(discriminator.parameters(), lr=learningr)
optimizerG = torch.optim.Adam(generator.parameters(), lr=learningr,weight_decay=1e-9)
# optimizerD = RMSprop(discriminator.parameters(),lr=learningr)
# optimizerG = RMSprop(generator.parameters(),lr=learningr)

discriminator.load_state_dict(torch.load('./models/ v1016CP 39epoch v1000a  SNGANanimeDis128.pkl'))
generator.load_state_dict(torch.load('./models/ v1016CP 39epoch v1000a  SNGANanimeGen128.pkl'))

# 训练准备
g_loss = []
d_loss = []
fix_noise = torch.randn(bat_size, noise_size)
dataset = Gan_data(path, transform_todata,img_size=img_size)
data_trainer = DataLoader(dataset, bat_size, shuffle=True, num_workers=worker)
if gpu:
    fix_noise = fix_noise.cuda()
    generator.cuda()
    discriminator.cuda()

one = torch.ones([1], dtype=torch.float)
mione = one * -1

# 训练
mean_d_loss = 0
current_iter = 0
for epoch in range(tr_epoch):
    for i, data in enumerate(data_trainer, 0):
        if (i + 1) % times_batch == 0:
            current_iter = current_iter + 1
            print(current_iter)
        if (i + 1) % 200 == 0:
            print((i + 1) / times_batch)
        input = data
        noise = torch.randn(input.size(0), noise_size)

        if gpu:
            one = one.cuda()
            mione = mione.cuda()
            input = input.cuda()
            noise = noise.cuda()

        # for para in discriminator.parameters():
        #    para.data.clamp_(-clamp_num,clamp_num)

        discriminator.zero_grad()
        ## train netd with real img
        dis_loss = discriminator(input)  # .mean().view(1)
        # if epoch>(tr_epoch-3):
        #    ioOut=np.append(ioOut,output.cpu().detach().numpy())
        # output.backward()
        ## train netd with fake img
        fake_pic = generator(noise).detach()
        gen_loss = discriminator(fake_pic)  # .mean().view(1)
        # outputloss = (output2 - output).mean().view(1)  #
        outputloss = loss_hinge(dis_loss, gen_loss)
        mean_d_loss = mean_d_loss + outputloss

        # if (i+1)%50 ==0:
        #    print(outputloss)
        # if (i + 1) < 200:
        #    if (i + 1) % 2 == 0:
        #        print(outputloss)
        # else:
        #    if (i + 1) % 50 == 0:
        #        print(outputloss)

        if i == 0:
            discriminator.zero_grad()

        outputloss.backward()
        if (i + 1) % times_batch == 0 or i == (len(data_trainer) - 1):
            optimizerD.step()
            optimizerD.zero_grad()
            d_loss.append(mean_d_loss.cpu().detach() / times_batch)
            mean_d_loss = 0
        # optimizerD.step()

        if (i + 1) % (5 * times_batch) == 0:
            mean_g_loss = 0
            for b_i_gan in range(times_batch):
                if b_i_gan == 0:
                    generator.zero_grad()
                noise.data.normal_(0, 1)
                fake_pic = generator(noise)
                output_g = -1 * discriminator(fake_pic).mean().view(1)
                mean_g_loss = mean_g_loss + output_g
                # output_g.backward(mione)
                output_g.backward()
            g_loss.append(mean_g_loss.cpu().detach() / times_batch)

            # generator.zero_grad()
            # noise.data.normal_(0, 1)
            # fake_pic = generator(noise)
            # output = discriminator(fake_pic).mean().view(1)
            # output.backward(mione)  #
            optimizerG.step()
            if i % 100 == 0: pass
        if (current_iter) % 2000 == 0:
            np.save('./resultslog/'+'animeGAN'+str(current_iter) + back_version + 'disloss' + version_p + '.npy', d_loss)
            np.save('./resultslog/'+'animeGAN'+str(current_iter) + back_version + 'genloss' + version_p + '.npy', g_loss)

    if (epoch + 1) % 1 == 0:
        fake_u = generator(fix_noise)
        imgs = make_grid(fake_u.data * 0.5 + 0.5).cpu()  # CHW
        # imgs = make_grid(fake_u).cpu() # CHW
        # plt.imshow(imgs.permute(1,2,0).numpy()) # HWC
        outprint = tf.to_pil_image(imgs)
        outprint.save('./results/' + back_version + str(epoch + 1) + 'th epoch ' + version_p + ' SNGANanime'+str(img_size)+'.png')

    # if epoch>(tr_epoch-3):
    # minmax=np.append(minmax,max(ioOut))
    # minmax=np.append(minmax,min(ioOut))
    # np.savetxt('out.txt',ioOut)
    # np.savetxt('minmax.txt',minmax)

    if (epoch + 1) % 4 == 0:
        torch.save(generator.state_dict(),
                   './models/' + back_version + str(epoch + 1) + 'epoch' + version_p + 'SNGANanimeGen' + str(
                       img_size) + '.pkl')
        torch.save(discriminator.state_dict(),
                   './models/' + back_version + str(epoch + 1) + 'epoch' + version_p + 'SNGANanimeDis' + str(
                       img_size) + '.pkl')

np.save('./resultslog/'+back_version +version_p +'disloss.npy', d_loss)
np.save('./resultslog/'+back_version +version_p +'genloss.npy', g_loss)
