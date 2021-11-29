import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision as tv
import numpy as np
import os
import random

ANIME_DATA = './dataset/anime_data/faces/'
EXTRA_DATA = './dataset/extra_data/images/'
max_iteration = 50000
d_update = 3
g_update = 3
batch_size = 64
noise_dim = 16
d_lr = 0.0001
g_lr = 0.0001

# clip weight of D
# use RMSProp instead of Adam
# train more iteration of D

# only use faces/   images/

class Dataset0(torch.utils.data.dataset.Dataset):
    def __init__(self, data):
        self.trainData = data

    def __len__(self):
        return len(self.trainData)

    def __getitem__(self, idx):
        return self.trainData[idx]

    def sample_batch_data(self, batch_size):
        inds = random.sample(range(self.__len__()), batch_size)
        return self.trainData[inds]

class D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=8,kernel_size=3,padding=1), #8, 96, 96
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=8,out_channels=16,kernel_size=5, stride=2, padding=2), #16, 48, 48
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=5, stride=2,padding=2), #16, 24, 24
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=5, stride=2,padding=2), #16, 12, 12
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16,out_channels=8,kernel_size=5, stride=2,padding=2), #8, 6, 6
            nn.LeakyReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=6*6*8, out_features=2),
            nn.Softmax(dim = 1)
        )

    def forward(self, x):
        out = self.conv(x)
        out = torch.reshape(out, (out.shape[0], -1))
        return self.fc(out)

class G(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=noise_dim, out_features=6*6*8)
        )
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8,out_channels=16,kernel_size=5, stride=2, padding=2, output_padding=1),
        )
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8,out_channels=16,kernel_size=5, stride=2, padding=2, output_padding=1), #8, 96, 96
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=16,out_channels=16,kernel_size=5, stride=2,padding=2, output_padding=1), #16, 48, 48
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=16,out_channels=16,kernel_size=5, stride=2,padding=2, output_padding=1), #16, 24, 24
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=16,out_channels=8,kernel_size=5, stride=2,padding=2, output_padding=1), #16, 12, 12
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=8,out_channels=3,kernel_size=3,padding=1), #8, 6, 6
            #nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.fc(x)
        x = torch.reshape(x, (x.shape[0], 8, 6, 6))
        # x=self.conv1(x)
        return self.conv(x)

def loss_d_fn(real_predicts, real_labels, fake_predicts, fake_labels):
    # real_predicts # batch_size, 1
    # fake_predicts # batch_size, 1
    # loss = torch.sum(torch.log(1 - real_predicts) + torch.log(fake_predicts)) / batch_size
    loss = torch.sum(1- real_predicts + fake_predicts) / batch_size
    return loss

def loss_g_fn(fake_predicts, real_labels):
    loss = torch.sum(1-fake_predicts) / batch_size
    return loss


def update_discriminator(discriminator, generator, dataset, d_optimizer):
    real_images = dataset.sample_batch_data(batch_size) # batch_size, c, w, h     64, 3, 96, 96
    noise = sample_batch_noise(batch_size, noise_dim)
    fake_images = generator(noise)
    real_predicts = discriminator(real_images)
    real_predicts = real_predicts[:, 1] #check
    fake_predicts = discriminator(fake_images)
    fake_predicts = fake_predicts[:, 1]
    real_labels = 0 # no use
    fake_labels = 0 # no use
    d_loss = loss_d_fn(real_predicts, real_labels, fake_predicts, fake_labels)
    # freeze generator
    for param in generator.parameters():
        param.requires_grad = False
    d_optimizer.zero_grad()
    d_loss.backward()
    clip_grad = 0.1
    nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=clip_grad)
    d_optimizer.step()
    # unfreeze generator
    for param in generator.parameters():
        param.requires_grad = True
    print('d_loss: ', d_loss)

def update_generator(discriminator, generator, dataset, g_optimizer):
    noise = sample_batch_noise(batch_size, noise_dim)
    fake_images = generator(noise)
    fake_predicts = discriminator(fake_images)
    real_labels = 0 # no use
    g_loss = loss_g_fn(fake_predicts, real_labels)
    # freeze discriminator
    for param in discriminator.parameters():
        param.requires_grad = False
    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()
    # unfreeze discriminator
    for param in discriminator.parameters():
        param.requires_grad = True
    
    print('g_loss: ', g_loss)

def train(discriminator, generator, dataset):
    discriminator.train()
    generator.train()
    d_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr = d_lr)
    g_optimizer = torch.optim.RMSprop(generator.parameters(), lr = g_lr)
    for i in range(max_iteration):
        for j in range(d_update):
            update_discriminator(discriminator, generator, dataset, d_optimizer)
        for j in range(g_update):
            update_generator(discriminator, generator, dataset, g_optimizer)

def sample_batch_noise(batch_size = batch_size, noise_dim = noise_dim):
    return torch.randn(size = (batch_size, noise_dim))

def test():
    discriminator.eval()
    generator.eval()
    noise = sample_batch_noise(batch_size, noise_dim)
    output_images = generator(noise)
    # TODO print images

def loadData(datapath = ANIME_DATA):
    print('Now loading data')
    cache_dir = datapath + '/data_cache.npy'
    print(cache_dir)
    if os.path.exists(cache_dir):
        print('use cache')
        return torch.from_numpy(np.load(cache_dir).astype(np.float32))
    print('reading...')
    images_list = os.listdir(ANIME_DATA)
    images = []
    for f in images_list:
        images.append(torch.unsqueeze(tv.io.read_image(datapath + f), dim = 0))
    data = torch.cat(images, dim = 0)
    print(data.shape)
    np.save(cache_dir, data.numpy())
    print('cache saved')
    return data

if __name__ == '__main__':
    discriminator = D()
    generator = G()
    # transform
    training_data = Dataset0(data = loadData())
    train(discriminator, generator, training_data)
    print('done')
    exit()