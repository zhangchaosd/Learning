import cv2
import torch
import torch.nn as nn

max_iteration = 50000
d_update = 100
g_update = 100
batch_size = 64
noise_dim = 16

# clip weight of D
# use RMSProp instead of Adam
# train more iteration of D

# only use faces/   images/

class D(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x

class G(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x

discriminator = D()
generator = G()

def update_discriminator():
    real_images = sample_batch_data(training_data, batch_size)
    noise = sample_batch_noise(batch_size, noise_dim)
    fake_images = generator(noise)
    real_predicts = discriminator(real_images)
    fake_predicts = discriminator(fake_images)
    d_loss = loss_d_fn(real_predicts, real_labels, fake_predicts, fake_labels)
    # freeze generator
    d_optimizer.zero_grad()
    d_loss.backward()
    d_optimizer.step()
    # unfreeze generator

def update_generator():
    noise = sample_batch_noise(batch_size, noise_dim)
    fake_images = generator(noise)
    fake_predicts = discriminator(fake_images)
    g_loss = loss_g_fn(fake_predicts, real_labels)
    # freeze discriminator
    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()
    # unfreeze discriminator

def train():
    discriminator.train()
    generator.train()
    for i in range(max_iteration):
        for j in range(d_update):
            update_discriminator()
        for j in range(g_update):
            update_generator()

def sample_batch_noise(batch_size = batch_size, noise_dim = noise_dim):
    return 1

def test():
    discriminator.eval()
    generator.eval()
    noise = sample_batch_noise(batch_size, noise_dim)
    output_images = generator(noise)

if __name__ == '__main__':
    import os
    dir2 = '.\\dataset\\anime_data\\testing_tags.txt'
    print(dir2)
    if os.path.exists(dir2):
        print('yes')
    else:
        print('no')
    print('done')
    exit()