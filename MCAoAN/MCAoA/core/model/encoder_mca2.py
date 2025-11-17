# --------------------------------------------------------
# mcaoa-vqa 
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# modified by Hyelee Kim
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

class AttentionOnAttention(nn.Module):
    def __init__(self, __C):
        super().__init__()
        self.__C = __C
        D = __C.HIDDEN_SIZE


        """projection - 어텐션을 위한 q,k,v 만들기 입력 차원 dim -> 맵핑 차원 inner_dim"""
        self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)   # input size = hidden_dim , w = input size * hidden_dim
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)  # 최종 출력 projection

        self.dropout = nn.Dropout(__C.DROPOUT_R)
        
        """gating(I,G => I*G)"""
        self.aoa = nn.Sequential(
            nn.Linear(2 * D, 2 * D),
            nn.GLU(),   # gated linear unit 1.차원 반으로 쪼개기 2. gate 계산 3.dot product  => 출력은 I*G 값 
            nn.Dropout(__C.DROPOUT_R)
        )
    
        
    """ 어텐션 가중합"""    
    
    def forward(self, v, k, q, mask):
        """_summary_

        Args:
            v : [B,N(query 길이),D]
            k : self-attention : [B,N(query 길이),D], cross-attention : [B,M(context/외부입력 길이),D]
            q : self-attention : [B,N(query 길이),D], cross-attention : [B,M(context/외부입력 길이),D]
            mask (_type_): _description_

        Returns:[B,N,D]
        """
        n_batches = q.size(0)
        
        """ v,k,q 만들기 & 헤드 분할"""
        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD, 
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)                       #[B,H,N,Dh]

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)                       #[B,H,N,Dh]

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)                       #[B,H,N/M,Dh]

        """attention"""
        atted = self.att(v, k, q, mask)
        
        """어텐션 가중합 결과(헤드 결합)"""
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.__C.HIDDEN_SIZE                                       # [B,Nq,D]
        )
        
        """ --- (AoA) Q도 [B,Nq,D]로 복원해 concat 후 게이팅 --- """
        q_merged = q.transpose(1, 2).contiguous().view(n_batches, -1, self.__C.HIDDEN_SIZE)   # [B,Nq,D]
        aoa_in = torch.cat([atted, q_merged], dim=-1)                   # [B,Nq,2D]
        aoa_out = self.aoa(aoa_in)                                      # [B,Nq,D]
        
        """최종 출력 프로젝션"""

        out = self.linear_merge(aoa_out)                                # [B,Nq,D]
        
        return out

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        """어텐션 점수 계산"""
        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)      #scaling

        #if mask is not None:
            #scores = scores.masked_fill(mask, -1e9)

        # (수정) [B,Nk] 등 다양한 입력을 안전하게 처리
        if mask is not None:
        # 일반적으로 key/value 쪽 패딩 마스크는 [B, Nk]
            if mask.dim() == 2:                 # [B, Nk]
                mask = mask[:, None, None, :]   # -> [B, 1, 1, Nk], scores=[B,H,Nq,Nk]와 브로드캐스트
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
    
      
        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, __C):
        super().__init__()

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
        
        self.aoa_att = AttentionOnAttention(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, x_mask):
        x = self.norm1(x + self.dropout1(
            self.aoa_att(x, x, x, x_mask)    
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
        
        self.aoa_self = AttentionOnAttention(__C)
        self.aoa_cross = AttentionOnAttention(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, y, x_mask, y_mask):
        # 1) Self AoA
        x = self.norm1(x + self.dropout1(
            self.aoa_self(x, x, x, x_mask)
        ))

        # 2) Cross AoA: q=x, k=v=y
        x = self.norm2(x + self.dropout2(
            self.aoa_cross(y, y, x, y_mask)
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
        super().__init__()
        self.L = __C.LAYER

        # 질문 인코더 : SAOA * L개
        self.y_saoa = nn.ModuleList([SAoA(__C) for _ in range(self.L)])
        
        # 이미지 인코더 : (SAoA -> GAoA) * L개
        self.x_saoa = nn.ModuleList([SAoA(__C) for _ in range(self.L)])    #이미지 SAoA 블록
        self.x_gaoa = nn.ModuleList([GAoA(__C) for _ in range(self.L)])

    def forward(self, x, y, x_mask, y_mask):
        
        # 간단한 가드(마스크 강제, hidden size 크기 매칭 강제)
        assert x_mask is not None and y_mask is not None, "x_mask, y_mask는 필수."
        assert x.size(-1) == y.size(-1), "Hidden size(D)가 x,y에서 같아야 한다."

        # (선택) dtype/device 정리
        x_mask = x_mask.to(torch.bool).to(x.device)
        y_mask = y_mask.to(torch.bool).to(y.device)
        
        for i in range(self.L):
            # 1) 질문 i층: SAoA
            y = self.y_saoa[i](y, y_mask)              # 질문 SAoA  [B, Nq, D]

            # 2) 이미지 i층: SAoA
            x = self.x_saoa[i](x, x_mask)              # [B, Nx, D]

            # 3) 이미지 i층: GAoA (q_i를 컨텍스트로)
            #    GAoA 내부: self-AoA(y) -> cross-AoA(q=x, k=v=context) -> FFN
            #    여기서는 y=x, context=q
            x = self.x_gaoa[i](x, y, x_mask, y_mask)

        return x, y






