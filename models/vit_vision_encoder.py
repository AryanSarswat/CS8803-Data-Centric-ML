from collections import namedtuple

import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from packaging import version
from torch import nn
from torchsummary import summary
import torchvision

# constants

Config = namedtuple('FlashAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)

# main class

class Attend(nn.Module):
    def __init__(self, use_flash = False):
        super().__init__()
        self.use_flash = use_flash
        assert not (use_flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = Config(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not use_flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            self.cuda_config = Config(True, False, False)
        else:
            self.cuda_config = Config(False, True, True)

    def flash_attn(self, q, k, v):
        config = self.cuda_config if q.is_cuda else self.cpu_config

        # flash attention - https://arxiv.org/abs/2205.14135
        
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(q, k, v)

        return out

    def forward(self, q, k, v):
        n, device, scale = q.shape[-2], q.device, q.shape[-1] ** -0.5

        if self.use_flash:
            return self.flash_attn(q, k, v)

        # similarity

        sim = torch.einsum("b h i d, b j d -> b h i j", q, k) * scale

        # attention

        attn = sim.softmax(dim=-1)

        # aggregate values

        out = torch.einsum("b h i j, b j d -> b h i d", attn, v)

        return out

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, use_flash = True):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = Attend(use_flash = use_flash)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, use_flash):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, use_flash = use_flash),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class SimpleViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, use_flash = True, include_fc = True):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, use_flash)

        self.to_latent = nn.Identity()
        
        self.include_fc = include_fc
        if self.include_fc:
            self.linear_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, num_classes)
            )

    def forward(self, img):
        *_, h, w, dtype = *img.shape, img.dtype

        x = self.to_patch_embedding(img)
        pe = posemb_sincos_2d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        x = self.transformer(x)
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        
        if self.include_fc:
            x = self.linear_head(x)
        
        return x

def vit_50M(num_classes, include_fc=True):
    model = SimpleViT(
        image_size = 224,
        patch_size = 32,
        num_classes = num_classes,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        use_flash = True,
        include_fc=include_fc
    )
    
    return model    

def vit_100M(num_classes, include_fc=True):
    model = SimpleViT(
        image_size = 224,
        patch_size = 32,
        num_classes = num_classes,
        dim = 1024,
        depth = 12,
        heads = 16,
        mlp_dim = 2048,
        use_flash = True,
        include_fc=include_fc
    )
    
    return model

def vit_200M(num_classes, include_fc=True):
    model = SimpleViT(
        image_size = 224,
        patch_size = 32,
        num_classes = num_classes,
        dim = 1024,
        depth = 24,
        heads = 16,
        mlp_dim = 2048,
        use_flash = True,
        include_fc=include_fc
    )
    
    return model

def vit_base(num_classes, include_fc=False, pretrained=True):
    model = torchvision.models.vit_b_16(weights='IMAGENET1K_V1')

    if not include_fc:
        model.heads = nn.Sequential(
            nn.Linear(768, num_classes)
        )

    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = vit_base(num_classes=80).to(device)
    test_in = torch.randn(5,3,224,224).to(device)
    print(f"Output shape : {model(test_in).shape}")
    #summary(model, (1, 3, 224, 224))
