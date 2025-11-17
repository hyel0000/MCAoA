# --------------------------------------------------------
# mcAoAN-vqa (
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from core.model.net_utils import FC, MLP, LayerNorm
from core.model.encoder_mca2 import MCA_ED

import torch.nn as nn
import torch.nn.functional as F
import torch


# ------------------------------
# ---- Simple attentive pool ----
# ------------------------------
class SimpleAttPool(nn.Module):
    """
    1. MLP: FC(d) - ReLU - Dropout(0.1) - FC(1)   
    2. softmax over sequence 
    3. weighted sum => x',y'
    4. concat x' + y'
    """
    def __init__(self, d, drop=0.1):
        super().__init__()
        
        def make_mlp():
            return nn.Sequential(
            nn.Linear(d, d), 
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(d, 1)
        )
        

        self.mlp_x = make_mlp()
        self.mlp_y = make_mlp()

    @staticmethod
    def _normalize_mask(mask: torch.Tensor | None, N_expected: int) -> torch.Tensor | None:
        if mask is None:
            return None
        # Accept [B,1,1,N] or [B,N]; return [B,N], dtype=bool
        if mask.dim() == 4:
            mask = mask.squeeze(1).squeeze(1)
        assert mask.dim() == 2 and mask.size(1) == N_expected, "mask must be [B,N]"
        return mask.to(torch.bool)
    
    """Y', X' 만들기"""
    @staticmethod
    def _attentive_sum(seq: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        # seq: [B,N,d], logits: [B,N] -> [B,d]
        a = F.softmax(logits, dim=1)                   # [B,N]
        return torch.bmm(a.unsqueeze(1), seq).squeeze(1)

    
    def forward(
            self,
            XL: torch.Tensor,        # [B, m, d]
            YL: torch.Tensor,        # [B, n, d]
            mask_x: torch.Tensor | None = None,
            mask_y: torch.Tensor | None = None,
            scale_x: torch.Tensor | None = None,   # [B,1] or [B]
            scale_y: torch.Tensor | None = None,   # [B,1] or [B]
        ) -> tuple[torch.Tensor, dict]:
        
        
            B, m, d = XL.size()
            By, n, dy = YL.size()
            assert B == By and d == dy, "Batch/hidden sizes must match between XL_att and YL_att."

            # 마스크 정규화
            mask_x = self._normalize_mask(mask_x, m)  # [B,m] or None
            mask_y = self._normalize_mask(mask_y, n)  # [B,n] or None

            """어텐션 점수 구하기"""
            s_x = self.mlp_x(XL).squeeze(-1)
            s_y = self.mlp_y(YL).squeeze(-1)

            # 마스킹
            if mask_x is not None:
                s_x = s_x.masked_fill(mask_x, torch.finfo(s_x.dtype).min)
            if mask_y is not None:
                s_y = s_y.masked_fill(mask_y, torch.finfo(s_y.dtype).min)

            # optional logit scaling (e.g., α, β from modality attention)
            if scale_x is not None:
                s_x = s_x * scale_x.view(B, 1)
            if scale_y is not None:
                s_y = s_y * scale_y.view(B, 1)

            """X', Y' 만들기 """
            refined_x = self._attentive_sum(XL, s_x)  # [B,d]
            refined_y = self._attentive_sum(YL, s_y)  # [B,d]

            """ concat """
            fused = torch.cat([refined_x, refined_y], dim=1)

            aux = {
                "logits_x": s_x, "logits_y": s_y,     # attention logits
                "refined_x": refined_x, "refined_y": refined_y,
            }
            
            return fused, aux


# -------------------------
# --------- Fuser ---------
# -------------------------
class ModalityAttentionFusion(nn.Module):
    """
    [X', Y' concat] -> FC(1024) - Dropout(0.2) - FC(512) - Dropout(0.2) - FC(2) - Softmax
    => (α, β)
    """
    def __init__(self, __C, p=0.2):
        super().__init__()
        
        d = __C.HIDDEN_SIZE   # X′,Y′ 차원 = backbone hidden size
        
        self.fuser = nn.Sequential(
            nn.Linear(2 * d, 1024), 
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(1024, 512),   
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(512, 2)
        )

    def forward(self, fused):
        w = F.softmax(self.fuser(fused), dim=-1)  # [B, 2]
        alpha = w[:, :1]  # [B,1]
        beta  = w[:, 1:]  # [B,1]
        return alpha, beta, w


# -------------------------
# ---- Main MCAN Model ----
# -------------------------
class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size):
        super().__init__()
        self.__C = __C

        # --- Text encoder ---
        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE
        )
        if __C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        # --- Image projection ---
        self.img_feat_linear = nn.Linear(__C.IMG_FEAT_SIZE, __C.HIDDEN_SIZE)

        # --- Backbone (MCA_ED) ---
        self.backbone = MCA_ED(__C)   # returns (img_seq, lang_seq)

        # --- Eq.(4)(5) attentive pooling (for X′, Y′ and A, B) ---
        self.pool = SimpleAttPool(d=__C.HIDDEN_SIZE, drop=0.1)

        # --- Modality attention fusion -> (α, β) ---
        self.fusion = ModalityAttentionFusion(__C, p=0.2)

        # --- Head: LayerNorm -> FC -> Sigmoid ---
        self.proj_norm = LayerNorm(2 * __C.HIDDEN_SIZE)
        self.proj = nn.Linear(2 * __C.HIDDEN_SIZE, answer_size)

    def forward(self, img_feat, ques_ix):
        # --- masks ---
        lang_mask = self.make_mask(ques_ix.unsqueeze(2))  # [B,1,1,Nq]
        img_mask  = self.make_mask(img_feat)              # [B,1,1,Nx]

        # --- text ---
        lang_seq = self.embedding(ques_ix)                # [B,Nq,W]
        lang_seq, _ = self.lstm(lang_seq)                 # [B,Nq,D]

        # --- image ---
        img_seq = self.img_feat_linear(img_feat)          # [B,Nx,D]

        """backbone"""
        # MCA_ED.forward(x=img_seq, y=lang_seq, ...) -> (img_seq', lang_seq')
        img_seq, lang_seq = self.backbone(
            img_seq, lang_seq, img_mask, lang_mask
        )  # [B,Nx,D], [B,Nq,D]

        """ concat X' + Y' """
        fused, aux = self.pool(img_seq, lang_seq, img_mask, lang_mask)   # [B,D]

        """ 가중값 """
        alpha, beta, _ = self.fusion(fused)  # [B,1], [B,1]

        """가중합된 X',Y' concat """
        # fused: [B, 2D] = concat(X', Y')
        D = fused.size(1) // 2

        Xp = fused[:, :D]   # [B,D]
        Yp = fused[:, D:]   # [B,D]

        # 가중치 곱
        Xp_weighted = alpha * Xp   # [B,D]
        Yp_weighted = beta * Yp    # [B,D]

        # 다시 concat
        fused_final = torch.cat([Xp_weighted, Yp_weighted], dim=-1)  # [B,2D]


        """정답 추론"""
        logits = self.proj(self.proj_norm(fused_final))  # [B, answer_size]                   
        prob = torch.sigmoid(logits)                    # BCE 사용 시
        
        return prob


    # Masking
    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)
