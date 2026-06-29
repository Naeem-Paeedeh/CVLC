from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, Tuple
import einops as eo
from torch import Tensor as T
from models.coalescent_projection import MultiheadAttention_manual_forward
from models.other_components import LoRA_QKV, DeepPrompts, Prompt
from torch.utils.checkpoint import checkpoint


# The ResNet parts are removed for simplification.


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(
        self,
        x: torch.Tensor,
        CPs_QK: T = None,
        CPs_SV: T = None,
        lora_qkv: LoRA_QKV = None
    ):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        
        res = MultiheadAttention_manual_forward(
            mha=self.attn, 
            query=x, 
            key=x,
            value=x,
            need_weights=False,
            attn_mask=self.attn_mask, 
            CPs_QK=CPs_QK,
            CPs_SV=CPs_SV,
            lora_qkv=lora_qkv
        )[0]
        
        return res

    def forward(
        self,
        x: torch.Tensor,
        CPs_QK: T = None,
        CPs_SV: T = None,
        lora_qkv: LoRA_QKV = None,
    ):
        x = x + self.attention(
            self.ln_1(x),
            CPs_QK=CPs_QK,
            CPs_SV=CPs_SV,
            lora_qkv=lora_qkv
        )
        
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        attn_mask: torch.Tensor = None,
        num_prompts: int = 0,
        use_checkpinting: bool = True
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.num_prompts = num_prompts
        
        self.use_checkpinting = use_checkpinting
        
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(
        self,
        x: torch.Tensor,
        chosen_layer: int = -1,
        coalescent_projections_dict: dict = None,
        LoRAs_dict: dict = None,
        prompts_dict: dict = None
    ):
        
        if chosen_layer == -1:
            chosen_layer = self.layers
        else:
            chosen_layer = min(chosen_layer, self.layers)
        
        for layer_number in range(chosen_layer):
            if coalescent_projections_dict is not None:
                CPs_QK = coalescent_projections_dict[f"QK,{layer_number}"]
                CPs_SV = coalescent_projections_dict[f"SV,{layer_number}"]
            else:
                CPs_QK = None
                CPs_SV = None
                
            if LoRAs_dict is not None and layer_number in LoRAs_dict:
                lora_qkv = LoRAs_dict[layer_number]
            else:
                lora_qkv = None
                
            if prompts_dict is not None:
                prompt_obj: Prompt = prompts_dict[layer_number]     # prompt_obj.shape: (num_prompts, dim_embed)
                prompt = prompt_obj.prompt.unsqueeze(1).expand(-1, x.shape[1], -1)
                
                if layer_number == 0:
                    x = torch.cat([prompt, x], dim=0)
                else:
                    # Drop previous prompts and replace them with current layer's prompts
                    x = torch.cat([prompt, x[self.num_prompts:]], dim=0)
                
            if self.use_checkpinting:
                x = checkpoint(
                    self.resblocks[layer_number],
                    x,
                    CPs_QK,
                    CPs_SV,
                    lora_qkv,
                    use_reentrant=False
                )
            else:
                x = self.resblocks[layer_number].forward(
                    x,
                    CPs_QK=CPs_QK,
                    CPs_SV=CPs_SV,
                    lora_qkv=lora_qkv
                )
        
        # Strip prompts out before returning to preserve expected sequence lengths        
        if prompts_dict is not None:
            x = x[self.num_prompts:]
            
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
        num_prompts: int = 0,
        use_checkpinting: bool = True
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.num_prompts = num_prompts
        
        self.use_checkpinting = use_checkpinting
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width=width, layers=layers, heads=heads, num_prompts=num_prompts, use_checkpinting=use_checkpinting)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(
        self, x: torch.Tensor,
        chosen_layer: int = -1,
        coalescent_projections_dict: dict = None,
        LoRAs_dict: dict = None,
        prompts_dict: dict = None
    ):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        
        x = self.transformer.forward(
            x=x,
            chosen_layer=chosen_layer,
            coalescent_projections_dict=coalescent_projections_dict,
            LoRAs_dict=LoRAs_dict,
            prompts_dict=prompts_dict
        )
        
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
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
                 use_checkpinting: bool = True,
                 num_vision_prompts: int = 0,
                 num_text_prompts: int = 0
                 ):
        super().__init__()
        
        self.transformer_heads = transformer_heads
        
        self.dim_embed_vision = vision_width
        self.dim_embed_text = transformer_width

        self.context_length = context_length
        self.num_text_prompts = num_text_prompts
        self.num_vision_prompts = num_vision_prompts
        
        self.use_checkpinting = use_checkpinting

        self.vision_heads = vision_width // 64
        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=self.vision_heads,
            output_dim=embed_dim,
            num_prompts=num_vision_prompts
        )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            use_checkpinting=use_checkpinting,
            num_prompts=num_text_prompts
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

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
        length = self.context_length + self.num_text_prompts
        mask = torch.empty(length, length)
        mask.fill_(float("-inf"))
        mask.triu_(1) 
        
        # Permit bidirectional attention between prompts so they don't treat each other casually
        if self.num_text_prompts > 0:
            mask[:self.num_text_prompts, :self.num_text_prompts] = 0.0
            
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(
        self, text,
        coalescent_projections_dict: dict = None,
        LoRAs_dict: dict = None,
        prompts_dict: dict = None
    ):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        
        x = self.transformer.forward(
            x,
            coalescent_projections_dict=coalescent_projections_dict,
            LoRAs_dict=LoRAs_dict,
            prompts_dict=prompts_dict
        )
        
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


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


def build_model(
    state_dict: dict,
    use_checkpinting: bool = True,
    num_vision_prompts: int = 0,
    num_text_prompts: int = 0
):
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
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim=embed_dim,
        image_resolution=image_resolution, 
        vision_layers=vision_layers, 
        vision_width=vision_width, 
        vision_patch_size=vision_patch_size,
        context_length=context_length, 
        vocab_size=vocab_size, 
        transformer_width=transformer_width, 
        transformer_heads=transformer_heads, 
        transformer_layers=transformer_layers,
        num_vision_prompts=num_vision_prompts,
        num_text_prompts=num_text_prompts,
        use_checkpinting=use_checkpinting
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()
