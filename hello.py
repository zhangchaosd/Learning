import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision as tv
import numpy as np
import os
import random
from torch.utils.tensorboard import SummaryWriter



ANIME_DATA = './dataset/anime_data/faces/'
EXTRA_DATA = './dataset/extra_data/images/'
max_iteration = 50000
d_update = 1
g_update = 5
batch_size = 256
noise_dim = 100
d_lr = 0.0002
g_lr = 0.0002
save_every = 20

# clip weight of D
# use RMSProp instead of Adam
# train more iteration of D

# only use faces/   images/

class Dataset0(torch.utils.data.dataset.Dataset):
    def __init__(self, data, transform = None):
        self.trainData = data
        self.transform = transform

    def __len__(self):
        return len(self.trainData)

    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.trainData[idx])
        return self.trainData[idx]

class D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5, stride=2, padding=2), #16, 48, 48
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=5, stride=2,padding=2), #16, 24, 24
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=5, stride=2,padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features = 8*8*256, out_features=2),
            # nn.Sigmoid()
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
            nn.Linear(in_features=noise_dim, out_features=128*16*16)
        )
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 128, out_channels = 128, kernel_size = 5, stride=2, padding=2,output_padding=1),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=5,padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels = 128, out_channels = 128, kernel_size = 5, stride=2, padding=2,output_padding=1),
            nn.Conv2d(in_channels=128,out_channels=64,kernel_size=5,padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64,out_channels=3,kernel_size=5,padding=2),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = torch.reshape(x, (x.shape[0], 128, 16, 16))
        return self.conv(x)

def train(discriminator, generator, dataloader, device, writer):
    discriminator.train()
    generator.train()
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr = d_lr)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr = g_lr)
    criterions = nn.BCELoss()
    true_labels = torch.ones(batch_size).to(device)
    fake_labels = torch.zeros(batch_size).to(device)
    mix_labels = torch.cat([true_labels, fake_labels], dim = 0).to(device)
    train_d = 0
    for i in range(max_iteration):
        for iter, (img) in enumerate(dataloader):
            real_img = img.to(device)
            if iter % d_update == 0:
            # if train_d < 1:
                reals = discriminator(real_img)
                reals = reals[:,1].view(-1)
                noises = torch.randn(size = (batch_size, noise_dim)).to(device)
                fake_image = generator(noises)
                fakes = discriminator(fake_image)
                fakes = fakes[:,1].view(-1)
                d_loss = criterions(torch.cat([reals, fakes], dim = 0), mix_labels)
                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()
                loss = d_loss.to('cpu').detach().numpy()
                print('d_loss: ', d_loss.to('cpu').detach().numpy())
                if loss < 0.001:
                    train_d = 5
            if iter % g_update == 0:
            # else:
                noises = torch.randn(size = (batch_size, noise_dim)).to(device)
                fake_image = generator(noises)
                output = discriminator(fake_image)
                output = output[:,1].view(-1)
                g_loss = criterions(output, true_labels)
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()
                loss = g_loss.to('cpu').detach().numpy()
                print('g_loss: ', g_loss.to('cpu').detach().numpy())
                train_d -= 5
        print('out ', i)
        noise = torch.randn(size = (batch_size, noise_dim)).to(device)
        fake_images = generator(noise)
        fake_images += 1
        fake_images /=2
        fake_images = fake_images.to('cpu')
        writer.add_images(tag='title01', img_tensor= fake_images, global_step = i, dataformats='NCHW')
        writer.flush()
        # 保存模型
        if (i + 1) % save_every == 0:
            torch.save(discriminator.state_dict(),  './' + 'd_{0}.pth'.format(i))
            torch.save(generator.state_dict(),  './' + 'g_{0}.pth'.format(i))


def test():
    discriminator.eval()
    generator.eval()
    noise = torch.randn(size = (batch_size, noise_dim))
    output_images = generator(noise)
    # TODO print images

def loadData(datapath = ANIME_DATA):
    print('Now loading data')
    cache_dir = './data_cache.npy'
    print(cache_dir)
    if os.path.exists(cache_dir):
        print('use cache')
        return torch.from_numpy(np.load(cache_dir).astype(np.float32))
    print('reading...')
    images_list = os.listdir(ANIME_DATA)
    images = []
    for f in images_list:
        images.append(torch.unsqueeze(tv.io.read_image(datapath + f, mode = tv.io.ImageReadMode.RGB), dim = 0))
    data = torch.cat(images, dim = 0) / 255
    print(data.shape)
    np.save(cache_dir, data.numpy())
    print('cache saved')
    return data

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    writer = SummaryWriter(log_dir = 'runs/fashion_mnist_experiment_1')
    discriminator = D().to(device)
    generator = G().to(device)
    # transform
    transforms = tv.transforms.Compose([
        tv.transforms.Resize([64,64]),
        tv.transforms.CenterCrop([64,64]),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    training_data = Dataset0(data = loadData(), transform = transforms)
    dataloader = DataLoader(training_data, batch_size = batch_size, drop_last=True)
    train(discriminator, generator, dataloader, device, writer)
    print('done')
    exit()