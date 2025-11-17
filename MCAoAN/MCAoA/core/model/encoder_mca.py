# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from core.model.net_utils import FC, MLP, LayerNorm

import torch.nn as nn
import torch.nn.functional as F
import torch, math

import torch
from torch import nn, einsum

from einops import rearrange

# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class AttentionOnAttention(nn.Module):  #multi-head attention + gating
    def __init__(
        self,
        *,
        dim = 512,
        dim_head = 64,
        heads = 8,
        dropout = 0.1,
        aoa_dropout = 0.1
    ):
        super().__init__()
        #self.__C = __C    #forward나 다른 메서드에서 __C값 참조 (없으면 __init__안에서만 쓰고 끝남)

        """projection 어텐션을 위한 q,k,v 만들기 입력 차원 dim -> 맵핑 차원 inner_dim"""
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5    #dot-product의 스케일링 팩터 1/sqrt(dim_head)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)  # q = 512*512, 입력차원 dim을 hidden state 512로 projection
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False) #kv = 512*1024 k,v 한번에 계산

        self.dropout = nn.Dropout(dropout)

        """gating(I,G => I*G)"""
        self.aoa = nn.Sequential(
            nn.Linear(2 * inner_dim, 2 * dim),
            nn.GLU(),   # gated linear unit 1.차원 반으로 쪼개기 2. gate 계산 3.dot product  => 출력은 I*G 값 
            nn.Dropout(aoa_dropout)
        )

    def forward(self, x, context = None):
        h = self.heads

        q_ = self.to_q(x)
        
        """attention (self-attention / cross-attention)"""
        context = default(context, x)     #context가 none이면 self-attention 
        kv = self.to_kv(context).chunk(2, dim = -1)

        # split heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q_, *kv))3
        """attention score"""
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        """attention"""
        attn = dots.softmax(dim = -1)
        attn = self.dropout(attn)

        """ weighted average"""
        attn_out = einsum('b h i j, b h j d -> b h i d', attn, v)

        """ concat heads"""
        out = rearrange(attn_out, 'b h n d -> b n (h d)', h = h)

        """ attention on attention"""
        out = self.aoa(torch.cat((out, q_), dim = -1))
        return out

# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, __C):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FF_SIZE,
            out_size=__C.HIDDEN_SIZE,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- SAoA ----
# ------------------------

class SAoA(nn.Module):
    def __init__(self, __C):    #__C : 하이퍼파라미터 모음집
        super().__init__()    # super(): 부모 클래스 -> nn.Module.__init() 호출
        self.aoa_att = AttentionOnAttention(
            dim = __C.HIDDEN_SIZE,
            dim_head = __C.HIDDEN_SIZE // __C.MULTI_HEAD,  # heads는 config에 맞게
            heads = __C.MULTI_HEAD,
            dropout = __C.DROPOUT_R,
            aoa_dropout = __C.DROPOUT_R
        )
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, x_mask):
        x = self.norm1(x + self.dropout1(
            self.aoa_att(x, x, x, x_mask)    # Self: context=None -> 자동으로 x
        ))

        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))
        return x



# -------------------------------
# ---- GAoA ----
# -------------------------------

class GAoA(nn.Module):
    def __init__(self, __C):
        super().__init__()
        self.aoa_self = AttentionOnAttention(
            dim = __C.HIDDEN_SIZE,
            dim_head = __C.HIDDEN_SIZE // __C.MULTI_HEAD,
            heads = __C.MULTI_HEAD,
            dropout = __C.DROPOUT_R,
            aoa_dropout = __C.DROPOUT_R
        )
        self.aoa_cross = AttentionOnAttention(
            dim = __C.HIDDEN_SIZE,
            dim_head = __C.HIDDEN_SIZE // __C.MULTI_HEAD,
            heads = __C.MULTI_HEAD,
            dropout = __C.DROPOUT_R,
            aoa_dropout = __C.DROPOUT_R
        )
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, y, x_mask=None, y_mask=None):
        # 1) Self AoA
        x = self.norm1(x + self.dropout1(
            self.aoa_self(x)
        ))

        # 2) Cross AoA: q=x, k=v=y
        x = self.norm2(x + self.dropout2(
            self.aoa_cross(x, context=y)
        ))

        # 3) Feed-Forward
        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))
        return x



# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------

class MCA_ED(nn.Module):
    def __init__(self, __C):
        super(MCA_ED, self).__init__()

        self.enc_list = nn.ModuleList([SAoA(__C) for _ in range(__C.LAYER)])
        self.dec_list = nn.ModuleList([GAoA(__C) for _ in range(__C.LAYER)])

    def forward(self, x, y, x_mask, y_mask):
        # Get hidden vector
        for enc in self.enc_list:
            x = enc(x, x_mask)

        for dec in self.dec_list:
            y = dec(y, x, y_mask, x_mask)

        return x, y
