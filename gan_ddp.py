import math
from typing import Optional, Tuple, Union, List

import torch
from torch import nn
from labml_helpers.module import Module


class Swish(Module):
    """
    ### Swish actiavation function

    $$x \cdot \sigma(x)$$
    """

    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    """
    ### Embeddings for $t$
    """

    def __init__(self, n_channels: int):
        """
        * `n_channels` is the number of dimensions in the embedding
        """
        super().__init__()
        self.n_channels = n_channels
        # First linear layer
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        # Activation
        self.act = Swish()
        # Second linear layer
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        # Create sinusoidal position embeddings
        # [same as those from the transformer](../../transformers/positional_encoding.html)
        #
        # \begin{align}
        # PE^{(1)}_{t,i} &= sin\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg) \\
        # PE^{(2)}_{t,i} &= cos\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg)
        # \end{align}
        #
        # where $d$ is `half_dim`
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        # Transform with the MLP
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)

        #
        return emb


class ResidualBlock(Module):
    """
    ### Residual block

    A residual block has two convolution layers with group normalization.
    Each resolution is processed with two residual blocks.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int,
                 n_groups: int = 32, dropout: float = 0.1):
        """
        * `in_channels` is the number of input channels
        * `out_channels` is the number of input channels
        * `time_channels` is the number channels in the time step ($t$) embeddings
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        * `dropout` is the dropout rate
        """
        super().__init__()
        # Group normalization and the first convolution layer
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # Group normalization and the second convolution layer
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = Swish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        # Linear layer for time embeddings
        self.time_emb = nn.Linear(time_channels, out_channels)
        self.time_act = Swish()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # First convolution layer
        h = self.conv1(self.act1(self.norm1(x)))
        # Add time embeddings
        h += self.time_emb(self.time_act(t))[:, :, None, None]
        # Second convolution layer
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))

        # Add the shortcut connection and return
        return h + self.shortcut(x)


class AttentionBlock(Module):
    """
    ### Attention block

    This is similar to [transformer multi-head attention](../../transformers/mha.html).
    """

    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None, n_groups: int = 32):
        """
        * `n_channels` is the number of channels in the input
        * `n_heads` is the number of heads in multi-head attention
        * `d_k` is the number of dimensions in each head
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        """
        super().__init__()

        # Default `d_k`
        if d_k is None:
            d_k = n_channels
        # Normalization layer
        self.norm = nn.GroupNorm(n_groups, n_channels)
        # Projections for query, key and values
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        # Linear layer for final transformation
        self.output = nn.Linear(n_heads * d_k, n_channels)
        # Scale for dot-product attention
        self.scale = d_k ** -0.5
        #
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        # Get shape
        batch_size, n_channels, height, width = x.shape
        # Change `x` to shape `[batch_size, seq, n_channels]`
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        # Get query, key, and values (concatenated) and shape it to `[batch_size, seq, n_heads, 3 * d_k]`
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        # Split query, key, and values. Each of them will have shape `[batch_size, seq, n_heads, d_k]`
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # Calculate scaled dot-product $\frac{Q K^\top}{\sqrt{d_k}}$
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        # Softmax along the sequence dimension $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = attn.softmax(dim=2)
        # Multiply by values
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        # Reshape to `[batch_size, seq, n_heads * d_k]`
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        # Transform to `[batch_size, seq, n_channels]`
        res = self.output(res)

        # Add skip connection
        res += x

        # Change to shape `[batch_size, in_channels, height, width]`
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)

        #
        return res


class DownBlock(Module):
    """
    ### Down block

    This combines `ResidualBlock` and `AttentionBlock`. These are used in the first half of U-Net at each resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class UpBlock(Module):
    """
    ### Up block

    This combines `ResidualBlock` and `AttentionBlock`. These are used in the second half of U-Net at each resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class MiddleBlock(Module):
    """
    ### Middle block

    It combines a `ResidualBlock`, `AttentionBlock`, followed by another `ResidualBlock`.
    This block is applied at the lowest resolution of the U-Net.
    """

    def __init__(self, n_channels: int, time_channels: int):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
        self.attn = AttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x


class Upsample(nn.Module):
    """
    ### Scale up the feature map by $2 \times$
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.conv(x)


class Downsample(nn.Module):
    """
    ### Scale down the feature map by $\frac{1}{2} \times$
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.conv(x)


class UNet(Module):
    """
    ## U-Net
    """

    def __init__(self, image_channels: int = 3,ret_channels: int = 3, n_channels: int = 64,
                 ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
                 is_attn: Union[Tuple[bool, ...], List[int]] = (False, False, True, True),
                 n_blocks: int = 2):
        """
        * `image_channels` is the number of channels in the image. $3$ for RGB.
        * `n_channels` is number of channels in the initial feature map that we transform the image into
        * `ch_mults` is the list of channel numbers at each resolution. The number of channels is `ch_mults[i] * n_channels`
        * `is_attn` is a list of booleans that indicate whether to use attention at each resolution
        * `n_blocks` is the number of `UpDownBlocks` at each resolution
        """
        super().__init__()

        # Number of resolutions
        n_resolutions = len(ch_mults)

        # Project image into feature map
        self.image_proj = nn.Conv2d(image_channels, n_channels, kernel_size=(3, 3), padding=(1, 1))

        # Time embedding layer. Time embedding has `n_channels * 4` channels
        self.time_emb = TimeEmbedding(n_channels * 4)

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = n_channels
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = in_channels * ch_mults[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(DownBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(out_channels, n_channels * 4, )

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
            # Final block to reduce the number of channels
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        # Final normalization and convolution layer
        self.norm = nn.GroupNorm(8, n_channels)
        self.act = Swish()
        self.final = nn.Conv2d(in_channels, ret_channels, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size]`
        """

        # Get time-step embeddings
        t = self.time_emb(t)

        # Get image projection
        x = self.image_proj(x)

        # `h` will store outputs at each resolution for skip connection
        h = [x]
        # First half of U-Net
        for m in self.down:
            x = m(x, t)
            h.append(x)

        # Middle (bottom)
        x = self.middle(x, t)

        # Second half of U-Net
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, t)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                #
                x = m(x, t)

        # Final normalization and convolution
        return self.final(self.act(self.norm(x)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def get_data():
    transform = transforms.Compose([transforms.ToTensor(),transforms.Pad(2)])
    tr = torchvision.datasets.MNIST(root = './mdata/train',train = True,download = True,transform = transform)
    valid_ds = torchvision.datasets.MNIST(root = './mdata/valid',train = False,download = True,transform = transform)
    tr_split_len = 10000
    train_ds = torch.utils.data.random_split(tr, [tr_split_len, len(tr)-tr_split_len])[0]

    train_loader = torch.utils.data.DataLoader(dataset=train_ds, batch_size = 32, shuffle = True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_ds, batch_size = 32, shuffle = False)

    print(len(train_loader))
    print(len(valid_loader))

    for i in train_loader:
        im, something = i
        #print(im)
        print(im.shape)
        break

    return train_loader, valid_loader



class Generator(nn.Module):
    def __init__(self,in_channel=3):
        super().__init__()
        self.in_ch = in_channel
        self.u = UNet(image_channels=2*in_channel,ret_channels=in_channel)

    def forward(self,x,t,y):
        x = torch.cat((x, y), dim=1)

        return self.u(x,t)





class Discriminator(nn.Module):
    def __init__(self,in_channels=3):
        super().__init__()

        self.num_channel = in_channels
        self.time_embedding = nn.Embedding(32,32*32)

        self.model = nn.Sequential(
            nn.Conv2d(2*in_channels+1, 64, kernel_size=3, stride=2, padding=1),  # Input: x_t-1, x_t
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(in_features=512*4, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512,out_features=1)
        )

    def forward(self, x_t_minus_1,  t,x_t):

        # Concatenate x_t-1, x_t, and t along the channel dimension

        emb = self.time_embedding(t)
        emb = torch.reshape(emb,(x_t.shape[0],1,32,32))

        x = torch.cat((x_t_minus_1, x_t, emb), dim=1)
        return self.model(x)






class DiffusionParams():
    def __init__(self):
        self.T = 4
        self.beta = torch.linspace(start = 1e-4, end = 0.4, steps = self.T, dtype = torch.float32, device = device)
        #print(self.beta)
        self.alpha = 1 - self.beta
        #print(self.alpha)
        self.alpha_bar = torch.cumprod(self.alpha, dim = 0)
        #rint(self.alpha_bar)

    def forward_process(self, x0: torch.Tensor, t: torch.Tensor):
        # Returns x_t
        if t==0:
            return x0
        epsilon = torch.randn_like(x0)
        mean = torch.sqrt(self.alpha_bar[t-1])
        mean = mean.view(-1, 1, 1, 1)
        std_dev = torch.sqrt(1 - self.alpha_bar[t-1])
        std_dev = std_dev.view(-1, 1, 1, 1)
        return mean*x0 + std_dev*epsilon, epsilon

    def posterior_forward(self, x0: torch.Tensor, xt: torch.Tensor, t: torch.Tensor):

        # t-1 has to be >= 1 and hence t>=2

        epsilon = torch.randn_like(x0)
        mean1 = (torch.sqrt(self.alpha_bar[t-2]))*(self.beta[t-1])/(1-self.alpha_bar[t-1])
        mean1 = mean1.view(-1, 1, 1, 1)
        mean2 = torch.sqrt(self.alpha[t-1])*(1-self.alpha_bar[t-2])/(1-self.alpha_bar[t-1])
        mean2 = mean2.view(-1, 1, 1, 1)
        std_dev = torch.sqrt(1 - self.alpha_bar[t-2])*(self.beta[t-1])/(1-self.alpha_bar[t-1])
        std_dev = std_dev.view(-1, 1, 1, 1)
        return mean1*x0 + mean2*xt + std_dev*epsilon, epsilon


    def forward_pairs(self, x0: torch.Tensor, t: torch.Tensor):
        # returns x_t and x_t+1 need t<8
        epsilon = torch.randn_like(x0)
        mean = torch.sqrt(self.alpha_bar[t-1])
        mean = mean.view(-1, 1, 1, 1)
        std_dev = torch.sqrt(1 - self.alpha_bar[t-1])
        std_dev = std_dev.view(-1, 1, 1, 1)

        mean1 = torch.sqrt(self.alpha_bar[t])
        mean1= mean.view(-1, 1, 1, 1)
        std_dev1 = torch.sqrt(1 - self.alpha_bar[t])
        std_dev1 = std_dev.view(-1, 1, 1, 1)
        return mean*x0 + std_dev*epsilon, mean1*x0+std_dev1*epsilon, epsilon

def sample(model,ek=None):
    with torch.no_grad():
        t = d.T
        xt = torch.randn((1,chan,32,32),device=device)
        for i in range(d.T-1):
            z =  torch.randn_like(xt, device=device)
            gen_out = model(xt, torch.tensor([t],device=device), z)
            xt, e = d.posterior_forward( gen_out, xt, t)
            t = t - 1
            #print(xt.shape)

        plt.imshow(xt[0].permute(1,2,0).cpu().numpy(),cmap="gray")
        if ek==None:
            plt.show()
        else:
            plt.savefig(f"{ek+1}_img.png")


chan = 1

d = DiffusionParams()
gen = Generator(in_channel=chan)
dis = Discriminator(in_channels=chan)

gen = gen.to(device)
dis = dis.to(device)
optimizer1 = torch.optim.Adam(gen.parameters(), lr = 2e-4, betas=(0.5,0.9))
optimizer2 = torch.optim.Adam(dis.parameters(), lr = 2e-4, betas=(0.5,0.9))

train_load, test_load = get_data()

num_epochs = 20
generated_images = []

r1_gamma = 0.02
k = 0



for epoch in range(num_epochs):
    mean_train_loss = []
    i = 0
    for images, _ in train_load:
        images = images.to(device)
        t = torch.randint(low = 2, high= d.T+1, size = (images.shape[0],), device = device)

        #Training discriminator
        for p in dis.parameters():
          p.requires_grad = True

        dis.zero_grad()
        xtm1, xt, eps = d.forward_pairs(images, t-1)
        xtm1.requires_grad = True

        output = dis(xtm1, t, xt.detach()).view(-1)
        #print(f"For real correct it predicted {output}")
        err_real = F.softplus(-output)
        err_real = err_real.mean()


        err_real.backward(retain_graph=True)

        temp_0 = err_real.item()

        grad_real = torch.autograd.grad(outputs=output.sum(), inputs=xtm1, create_graph=True)[0]
        grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()


        grad_penalty = r1_gamma / 2 * grad_penalty
        grad_penalty.backward()

        z =  torch.randn_like(xt, device=device)

        gen_out = gen(xt, t, z)


        x_pos_sample, e = d.posterior_forward( gen_out, xt, t-1)

        out1 = dis(x_pos_sample, t, xt.detach()).view(-1)

        #print(f"For fake it predicted {out1}")
        err_fake = F.softplus(out1)
        err_fake = err_fake.mean()
        err_fake.backward()

        temp_1 = err_fake.item()

        optimizer2.step()


        for p in dis.parameters():
            p.requires_grad = False

        gen.zero_grad()

        t = torch.randint(low = 2, high= d.T+1, size = (images.shape[0],), device = device)

        xtm1, xt, eps = d.forward_pairs( images, t-1)
        z =  torch.randn_like(xt, device=device)
        gen_out = gen(xt, t, z)

        post_sample, nois = d.posterior_forward( gen_out, xt, t-1)
        out1 = dis(post_sample, t, xt.detach()).view(-1)

        #print(f"For gen train it predicted {out1}")
        err_fake = F.softplus(-out1)
        err_fake = err_fake.mean()
        err_fake.backward()
        temp3 = err_fake.item()
        optimizer1.step()
        #sscheduler.step()
        gen.zero_grad()
        optimizer1.zero_grad()
        mean_train_loss.append(err_fake.item()/32)
        #print(err_fake.item()/32)


        # print(k)
        # k = k+1
        # if k%2 == 0:
        #     print(f" Loss of D on real : {temp_0}, Loss of D on fake {temp_1}, loss of G on fake {temp3}")

        #     sample(gen,k)
        if(k%100 == 0):
          print(k)
          sample(gen)
        k = k +1



    print(f"Epoch: {epoch+1}/{num_epochs}: Train Loss : {np.mean(np.array(mean_train_loss))}")
    #torch.save(gen.state_dict(),f"{epoch}_gen.pth")
    sample(gen,epoch)
