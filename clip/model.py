from collections import OrderedDict
from typing import Tuple, Union

import os
import math
import copy
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from collections import Counter


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class MultiheadAttention(nn.MultiheadAttention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.attn_gradients = None
        self.attention_map = None
        
    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
        
    def get_attn_gradients(self):
        return self.attn_gradients
    
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map
        
    def get_attention_map(self):
        return self.attention_map        

    def forward(self, query, key, value, register_hook=False, prompt=None, 
                key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True, is_causal=False):
        query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        B, N, C = query.shape
        
        qkv = F.linear(input=query, weight=self.in_proj_weight, bias=self.in_proj_bias)        
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
                                
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        if prompt is not None:
            pk, pv = prompt
            pk = pk.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            pv = pv.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k = torch.cat((pk,k), dim=2)
            v = torch.cat((pv,v), dim=2)
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (pk.shape[2],0))
                
        q = q * math.sqrt(1.0 / float(self.head_dim))
        
        attn = q @ k.transpose(-2, -1)
        if attn_mask is not None:
            attn = attn + attn_mask.unsqueeze(0).unsqueeze(0)
        attn = attn.softmax(dim=-1)
        if self.dropout > 0 and self.training:
            attn = F.dropout(attn, p=self.dropout)

        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)        
               
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = F.linear(input=x, weight=self.out_proj.weight, bias=self.out_proj.bias)    
        
        x = x.transpose(1, 0)
        
        return x, attn



# +
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, text_or_image='text'):
        super().__init__()

        self.text_or_image = text_or_image
#         if text_or_image == 'image':
        self.attn = MultiheadAttention(embed_dim=d_model, num_heads=n_head)
#         else:
#         self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x, register_hook=False, prompt=None):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask, prompt=prompt)[0]

    def forward(self, x, register_hook=False, prompt=None, ssf=None):
        if ssf is not None:
            x = x + self.attention(ssf_ada(self.ln_1(x), ssf[0]), register_hook, prompt)
            x = x + self.mlp(ssf_ada(self.ln_2(x), ssf[1]))
        else:
            x = x + self.attention(self.ln_1(x), register_hook, prompt)
            x = x + self.mlp(self.ln_2(x))
        
        return x
    
def ssf_ada(x, ssf):
    scale, shift = ssf
    assert scale.shape == shift.shape
    if x.shape[0] == scale.shape[0]:
        return x * scale.unsqueeze(dim=1) + shift.unsqueeze(dim=1)
    elif x.shape[-1] == scale.shape[-1]:
        return x * scale.unsqueeze(dim=0) + shift.unsqueeze(dim=0)
    elif x.shape[-1] == scale.shape[0]:
        return x * scale + shift
    elif x.shape[1] == scale.shape[0]:
        return x * scale.unsqueeze(dim=0) + shift.unsqueeze(dim=0)
    else:
        print(x.shape, scale.shape, shift.shape)
        raise ValueError('the input tensor shape does not match the shape of the scale factor.')



# -



class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, text_or_image=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, text_or_image) for _ in range(layers)])

    def forward(self, x, register_blk=-1, prompt=None, ssf=None):
        for i,blk in enumerate(self.resblocks):
            if prompt is None:
                x = blk(x, register_blk==i, prompt=None, ssf=None)
            else:
                x = blk(x, register_blk==i, prompt=prompt[i], ssf=ssf[i])
            
        return x


class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, text_or_image=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        # Added so this info is available. should not change anything.
        self.patch_size = patch_size
        self.width = width
        self.layers = layers
        self.heads = heads

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, text_or_image=text_or_image)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        
    def forward(self, x, register_blk=-1, prompt=None, ssf=None, return_proj=True):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, register_blk, prompt, ssf)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)    
    
        if not return_proj:
            return x, x[:,0,:]
        

        x = x[:,0,:]
        
        if self.proj is not None and return_proj:
            x = x @ self.proj

        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 baseline = False,
                 args=None
                 ):
        super().__init__()
        self.baseline = baseline

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisualTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                text_or_image='image',
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            text_or_image='text'
        )
        
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

        self.prompt_pool = VisualLanguagePrompt(
            vision_emb_dim=vision_width, 
            text_emb_dim=transformer_width, 
            n_tasks=len(args.dataset_order), 
            n_prompt_per_task=20,
            n_prompt_length=8,
        )
        
    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype
    
    def get_sim(self, image, last=True):
        with torch.no_grad():
            v_seq, v_last = self.encode_image(image, return_proj=False)        
        return self.prompt_pool.get_sim(v_seq, v_last, last=last) #self.prompt_pool.get_sim(v_last) #

    def get_taskid(self, image, batch=True):
        with torch.no_grad():
            v_seq, v_last = self.encode_image(image, return_proj=False)        
        return self.prompt_pool.get_taskid(v_seq, v_last, batch) #[0] #self.prompt_pool.get_taskid(v_last, batch) #
            
    def encode_image(self, image, prompt=None, ssf=None, return_proj=True):
        x = self.visual(image.type(self.dtype), prompt=prompt, ssf=ssf, return_proj=return_proj)

        return x    

    def encode_text(self, text, prompt=None, ssf=None, return_proj=True):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, prompt=prompt, ssf=ssf)
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        x = self.ln_final(x).type(self.dtype)

        if not return_proj:
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
            return x
        
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        
        return x
    
    def get_prompt(self, image=None, text=None, taskid=-1, is_train=False, check_novelty=False):
        if taskid == -1:
            taskid = None
            
        assert not ((image is None or text is None) and taskid is None)
        
        v_seq, v_last, t_last = None, None, None
        if taskid is None or is_train:
            with torch.no_grad():
                v_seq, v_last = self.encode_image(image, return_proj=False)
                t_last = self.encode_text(text, return_proj=False)
        else:
            if text is not None:
                with torch.no_grad():
                    t_last = self.encode_text(text, return_proj=False)
            if image is not None:
                with torch.no_grad():
                    v_seq, v_last = self.encode_image(image, return_proj=False)
        v_p, v_s, t_p, t_s, prompt_loss = self.prompt_pool(v_seq, v_last, t_last, is_train, taskid, check_novelty=check_novelty)    
        
        return v_p, v_s, t_p, t_s, prompt_loss

    def forward(self, image, text=None, taskid=-1, is_train=False, use_prompt=True, zeroshot_weights=None, check_novelty=False):
        assert (not (text is None and zeroshot_weights is None)) and not (text is not None and zeroshot_weights is not None) 
        if taskid == -1:
            taskid = None
        
        if self.prompt_pool is not None and use_prompt:
            v_p, v_s, t_p, t_s, prompt_loss = self.get_prompt(image, text, taskid, is_train, check_novelty=check_novelty)                
        else:
            v_p, v_s, t_p, t_s, prompt_loss = None, None, None, None, 0
        
        image_features = self.encode_image(image, v_p, v_s)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        if text is not None:
            text_features = self.encode_text(text, t_p, t_s)            
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        else:
            text_features = zeroshot_weights / zeroshot_weights.norm(dim=-1, keepdim=True)

        # if self.baseline:
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        
        return logits_per_image, logits_per_text, prompt_loss, image_features, text_features




class VisualLanguagePrompt(nn.Module):
    def __init__(self, vision_emb_dim, text_emb_dim, n_tasks, n_prompt_per_task, n_prompt_length):
        super().__init__()
        
        self.vision_emb_dim = vision_emb_dim
        self.text_emb_dim = text_emb_dim
        
        self.task_count = 0
        self.n_tasks = n_tasks
        self.n_prompt_per_task = n_prompt_per_task
        self.n_prompt_length = n_prompt_length

        self.prompt_layers = [0,1,2,3,4,5,6,7,8,9,10,11]
        
        key = nn.Parameter(torch.FloatTensor(self.n_tasks, self.vision_emb_dim), requires_grad=True)
        self.key = self.gram_schmidt(key, pt=1, cross_task=True)

        self.weights = nn.Parameter(torch.FloatTensor(self.n_tasks, 14*14+1), requires_grad=True)
        nn.init.uniform_(self.weights)
        
        for e in self.prompt_layers:
            v_prompt = nn.Parameter(torch.FloatTensor(self.n_tasks*self.n_prompt_per_task, self.n_prompt_length, self.vision_emb_dim), requires_grad=True)
            v_prompt = self.gram_schmidt(v_prompt, pt=self.n_prompt_per_task)
            
            v_att = nn.Parameter(torch.FloatTensor(self.n_tasks*self.n_prompt_per_task, self.vision_emb_dim), requires_grad=True)
            v_att = self.gram_schmidt(v_att, pt=self.n_prompt_per_task)
            
            setattr(self, f'v_prompt_{e}',v_prompt)
            setattr(self, f'v_att_{e}',v_att)
            
            t_prompt = nn.Parameter(torch.FloatTensor(self.n_tasks*self.n_prompt_per_task, self.n_prompt_length, self.text_emb_dim), requires_grad=True)
            t_prompt = self.gram_schmidt(t_prompt, pt=self.n_prompt_per_task)
            
            t_att = nn.Parameter(torch.FloatTensor(self.n_tasks*self.n_prompt_per_task, self.text_emb_dim), requires_grad=True)
            t_att = self.gram_schmidt(t_att, pt=self.n_prompt_per_task)
            
            setattr(self, f't_prompt_{e}',t_prompt)
            setattr(self, f't_att_{e}',t_att)
            
        self.nov_threshold = []
        self.count = np.zeros(self.n_tasks)
        self.train_iter = 0
        
    def process_task_count(self):
        self.task_count += 1
        self.train_iter = 0
        self.key = self.gram_schmidt(self.key, pt=1, cross_task=True, current_only=True)
        
    # code for this function is modified from:
    # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
    def gram_schmidt(self, vv, pt, cross_task=False, current_only=False):

        def projection(u, v):
            denominator = (u * u).sum()

            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u

        # check if the tensor is 3D and flatten the last two dimensions if necessary
        is_3d = len(vv.shape) == 3
        if is_3d:
            shape_2d = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0],-1)

        # swap rows and columns
        vv = vv.T

        # process matrix size
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)

        if current_only: 
            t0, t1 = self.task_count, self.task_count+1
        else:
            t0, t1 = 0, self.n_tasks
            
        for t in range(t0, t1):
            # get starting point
            s = int(t * pt)
            f = int((t + 1) * pt)
            
            if current_only:
                uu[:, 0:s] = vv[:, 0:s].clone()       
                
            for k in range(s, f):
                redo = True
                while redo:
                    redo = False
                    vk = torch.randn_like(vv[:,k]).to(vv.device)
                    uk = 0
                    s0 = 0 if cross_task else s
                    for j in range(s0, k):
                        if not redo:
                            uj = uu[:, j].clone()
                            proj = projection(uj, vk)
                            if proj is None:
                                redo = True
                                print('restarting!!!')
                            else:
                                uk = uk + proj
                    if not redo: uu[:, k] = vk - uk
            for k in range(s, f):
                uk = uu[:, k].clone()
                uu[:, k] = uk / (uk.norm())

        # undo swapping of rows and columns
        uu = uu.T 

        # return from 2D
        if is_3d:
            uu = uu.view(shape_2d)
        
        return torch.nn.Parameter(uu)             

    def set_novelty_threshold(self, threshold, task_id=None):
        if task_id is not None and task_id < len(self.nov_threshold):
            self.nov_threshold[task_id] = threshold
        else:
            self.nov_threshold.append(threshold)
                    
    def get_sim(self, v_seq, v_last, last=True, train=False):  
        
        # cosine similarity to match keys/querries
        n_k = nn.functional.normalize(self.key, dim=1)
        
        weights = F.softmax(self.weights, dim=1)
        
        n_k = torch.einsum('kl,kj->klj', weights, n_k)
                
        n_v_q = nn.functional.normalize(v_seq, dim=2).detach()      
        cos_sim = torch.einsum('blj,klj->bk', n_v_q, n_k)
    
        cos_sim = cos_sim[:,:self.task_count+1]  
        
        if last:
            cos_sim = cos_sim[:,-1]
            
        return cos_sim        
        
    def get_taskid(self, v_seq, v_last, batch=True):
        cos_sim = self.get_sim(v_seq, v_last, last=False, train=False)
        
        if batch:
            top_idx = torch.topk(cos_sim.mean(dim=0), k=1)
            if top_idx.indices.item() >= len(self.nov_threshold):
                return top_idx.indices.item(), None 
            else:
                return top_idx.indices.item(), top_idx.values.item() < self.nov_threshold[top_idx.indices.item()], 
        else:
            top_idx = torch.topk(cos_sim, k=1, dim=1)
            return top_idx.indices.cpu().squeeze(dim=1), top_idx.values.cpu().squeeze(dim=1) < torch.tensor(self.nov_threshold)[top_idx.indices.cpu().squeeze(dim=1)]
                        
    def get_params(self, modality, layer):
        prompt = getattr(self,f'{modality[0]}_prompt_{layer}') # 0 based indexing here
        att = getattr(self,f'{modality[0]}_att_{layer}')
        
        prompt = prompt[:self.n_prompt_per_task*(self.task_count+1)]
        att = att[:self.n_prompt_per_task*(self.task_count+1)]
            
        return prompt, att
    
    def forward(self, v_seq=None, v_last=None, t_last=None, train=False, task_id=None, check_novelty=False):
        loss = 0
        vp_return = []
        vs_return = []
        tp_return = []
        ts_return = []
        
        if task_id is None or train:            
            cos_sim = self.get_sim(v_seq, v_last, last=False, train=train)

        for l in self.prompt_layers:
            v_p, v_a = self.get_params(modality='visual', layer=l)
            t_p, t_a = self.get_params(modality='text', layer=l)
            
            if train:
                if self.task_count > 0:
                    cos_sim = torch.cat([cos_sim[:,:-1].detach(), cos_sim[:,-1:]], dim=1) 
                    v_p = torch.cat([v_p[:-self.n_prompt_per_task].detach(), v_p[-self.n_prompt_per_task:]])
                    v_a = torch.cat([v_a[:-self.n_prompt_per_task].detach(), v_a[-self.n_prompt_per_task:]])
                    t_p = torch.cat([t_p[:-self.n_prompt_per_task].detach(), t_p[-self.n_prompt_per_task:]])
                    t_a = torch.cat([t_a[:-self.n_prompt_per_task].detach(), t_a[-self.n_prompt_per_task:]])
                    
                if l == 0:
                    loss = 1 - cos_sim[:, task_id]
            
                top_idx = self.task_count
                
            else:
                if task_id is not None:
                    top_idx = task_id
                else:
                    top_sim = torch.topk(cos_sim.mean(dim=0), k=1)
                    top_idx = top_sim.indices.squeeze()
                    if check_novelty and top_sim.values.item() < self.nov_threshold[top_idx.item()]:
                        vs_return = None
                        ts_return = None
                        vp_return = None
                        tp_return = None
                        break

                self.count[top_idx] += 1

                
            v_p_ = v_p.reshape(self.task_count+1, self.n_prompt_per_task, self.n_prompt_length, self.vision_emb_dim)[top_idx]
            v_a_ = v_a.reshape(self.task_count+1, self.n_prompt_per_task, self.vision_emb_dim)[top_idx]
            t_p_ = t_p.reshape(self.task_count+1, self.n_prompt_per_task, self.n_prompt_length, self.text_emb_dim)[top_idx]
            t_a_ = t_a.reshape(self.task_count+1, self.n_prompt_per_task, self.text_emb_dim)[top_idx]

            if v_last is not None:
                n_v_q = nn.functional.normalize(v_last, dim=1).detach()                
                n_v_a = nn.functional.normalize(v_a_, dim=1)
                v_qa = torch.einsum('bd,kd->bk', n_v_q, n_v_a)
                v_p_ = torch.einsum('bk,kld->bld', v_qa, v_p_)     
                
                vp_k, vp_v = torch.split(v_p_, int(self.n_prompt_length/2), dim=1)
                vp_return.append([vp_k, vp_v]) 
            
            if t_last is not None:
                n_t_q = nn.functional.normalize(t_last, dim=1).detach()                
                n_t_a = nn.functional.normalize(t_a_, dim=1)
                t_qa = torch.einsum('bd,kd->bk', n_t_q, n_t_a)
                t_p_ = torch.einsum('bk,kld->bld', t_qa, t_p_)
                
                tp_k, tp_v = torch.split(t_p_, int(self.n_prompt_length/2), dim=1)
                tp_return.append([tp_k, tp_v]) 

            vs_return.append(None)
            ts_return.append(None)
                    
        return vp_return, vs_return, tp_return, ts_return, loss


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, args=None):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, args=args
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    model.load_state_dict(state_dict, strict=False)
    for p in model.parameters():
        p.data = p.data.float()
    return model.eval()


# +
# note - ortho init has not been found to help l2p/dual prompt
def tensor_prompt(a, b, c=None, ortho=False):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a,b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a,b,c), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        nn.init.uniform_(p)
    return p    

def ortho_penalty(t):
    return ((t @t.T - torch.eye(t.shape[0]).cuda())**2).mean()
