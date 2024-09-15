import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.adamw
import torchvision
import matplotlib.pyplot as plt

torch.set_default_device(torch.device("cuda:0" if torch.cuda.is_available else "cpu"))

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
            nn.Linear(z_size*4, z_size * 16),
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
            nn.Linear(image_size // 8, 1),
            nn.Sigmoid()
        ])
    def forward(self, ins):
        ins = ins.view(-1, self.image_size)
        for layer in self.layers:
            ins = layer(ins)
        return ins
z_size = 4 * 4 * 3
my_generator = Generator(z_size)
generated_image = my_generator(torch.randn(2, z_size))
image_size = z_size * 16
my_discriminator = Discriminator(image_size)
print("discriminator score:", my_discriminator(generated_image))

optim_g = torch.optim.AdamW(my_generator.parameters(), lr=0.001)
optim_d = torch.optim.AdamW(my_discriminator.parameters(), lr=0.001)


batch_size = 16
max_steps = int((train_len // batch_size) * 10)

my_generator.train()
my_discriminator.train()
for step in range(max_steps):
    idx = torch.randint(0, train_len, (batch_size // 2,))
    dxbr = train_ds[idx]
    noise = torch.randn(batch_size // 2, z_size)
    dxbg = my_generator(noise).view(batch_size //2, image_size)
    dybr = torch.ones(batch_size//2, 1)
    dybg = torch.zeros(batch_size//2, 1)

    g_loss = F.binary_cross_entropy(my_discriminator(dxbg), torch.ones(batch_size // 2, 1))
    
    optim_g.zero_grad()

    g_loss.backward()
    optim_g.step()


    optim_d.zero_grad()
    d_loss_real = F.binary_cross_entropy(my_discriminator(dxbr), dybr)
    d_loss_fake= F.binary_cross_entropy(my_discriminator(dxbg.detach()), dybg)
    d_loss = (d_loss_real + d_loss_fake) / 2
    d_loss.backward()
    optim_d.step()

    if step % 100 == 0:
        print(f"step: {step} | loss g: {g_loss.item():.4f} | loss d: {d_loss.item():.4f}")

my_generator.eval()
my_discriminator.eval()

noise = torch.randn(1, z_size)
img = tensor_to_image(my_generator(noise).view(3, 16, 16))
img.show()
