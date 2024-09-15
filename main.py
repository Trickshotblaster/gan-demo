import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt

ds = torchvision.datasets.Flowers102("data/", download=True)
ds[0][0]
print("label for element 0 of data:", ds[0][1])
train_len = len(ds)
print("train examples:", train_len)
img_to_tensor = torchvision.transforms.functional.pil_to_tensor
tensor_to_image = torchvision.transforms.functional.to_pil_image
print(img_to_tensor(ds[0][0]).shape)
train_ds = []
for pair in ds:
    train_ds.append(img_to_tensor(pair[0].resize((16, 16))) / 255.0)
train_ds = torch.stack(train_ds)


class Generator(nn.Module):
    def __init__(self, z_size):
        super(Generator, self).__init__()
        self.z_size = z_size
        self.layers = nn.ModuleList([
            nn.Linear(z_size, z_size * 4),
            nn.BatchNorm1d(z_size * 4),
            nn.LeakyReLU(),
            nn.Linear(z_size*4, z_size * 8),
            nn.Tanh(),
        ])
        
    def forward(self, ins):
        for layer in self.layers:
            ins = layer(ins)
        return ins

class Discriminator(nn.Module):
    def __init__(self, image_size):
        super(Discriminator, self).__init__()
        self.image_size = image_size
        self.layers = nn.ModuleList([
            nn.Linear(image_size, image_size // 4),
            nn.BatchNorm1d(image_size // 4),
            nn.LeakyReLU(),
            nn.Linear(image_size // 4, image_size // 8),
            nn.BatchNorm1d(image_size // 8),
            nn.LeakyReLU(),
            nn.Linear(image_size // 8, 1)
        ])
    def forward(self, ins):
        ins = ins.view(-1, self.image_size)
        for layer in self.layers:
            ins = layer(ins)
        return ins
z_size = 4 * 4 * 3
my_generator = Generator(z_size)
generated_image = my_generator(torch.randn(2, z_size))
tensor_to_image(generated_image.view(3, 16, 16)).show()
image_size = z_size * 8
my_discriminator = Discriminator(image_size)
print("discriminator score:", my_discriminator(generated_image))