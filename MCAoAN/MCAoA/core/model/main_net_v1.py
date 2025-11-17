# --------------------------------------------------------
# mcAoAN-vqa (
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from core.model.net_utils import FC, MLP, LayerNorm
from core.model.mca import MCA_ED

import torch.nn as nn
import torch.nn.functional as F
import torch


# ------------------------------
# ---- Flatten the sequence ----
# ------------------------------
"""질문/이미지 시퀀스를 요약벡터 X'.Y'으로 만들기"""

class AttFlat(nn.Module):
    """
    x: 시퀀스 임베딩 [B,N,D]
    x_mask: 패당 마스크 [B,1,1,N], true = 패딩
    """
    def __init__(self, __C):
        super().__init__()
        self.__C = __C

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FLAT_MLP_SIZE,
            out_size=__C.FLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )                                           #출력 [B,N,G: 토큰마다 G개의 점수(glimpse) 뽑음

        self.linear_merge = nn.Linear(
            __C.HIDDEN_SIZE * __C.FLAT_GLIMPSES,
            __C.FLAT_OUT_SIZE
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        
        """ 패딩 토큰 가리기 """
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            torch.finfo(att.dtype).min
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)    # [B, G, D]
        
        """flatten + 선형 결합"""
        x_atted = x_atted.view(x_atted.size(0), -1)   # [B, G*D]
        x_atted = self.linear_merge(x_atted)    # [B, FLAT_OUT_SIZE]

        return x_atted

# -------------------------
# --------- Fuser ---------
# -------------------------

class ModalityAttentionFusion(nn.Module):
    """
    Concatenate X', Y' -> FC(1024) - Dropout(0.2) - FC(512) - Dropout(0.2) - FC(2) - Softmax
    Output 2-dim weights (α for image, β for text), then weighted sum α·X' + β·Y'
    """
    def __init__(self, __C, p=0.2):
        super().__init__()
        
        d = __C.FLAT_OUT_SIZE
        self.fuser = nn.Sequential(
            nn.Linear(2*d, 1024), 
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(1024, 512), 
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(512, 2)
        )

    def forward(self, x_att, y_att):
        """
        x_att, y_att: [B, d]
        returns fused: [B, d], weights: [B, 2]
        """
        pair = torch.cat([x_att, y_att], dim=1)      # [B, 2d]
        w = F.softmax(self.fuser(pair), dim=-1)      # [B, 2]
        # w[:,0] -> image weight α, w[:,1] -> text weight β
        fused = w[:, :1] * x_att + w[:, 1:] * y_att  # broadcast to [B, d]
        return fused, w



# -------------------------
# ---- Main MCAN Model ----
# -------------------------

class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size):
        super(Net, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE
        )

        # Loading the GloVe embedding weights
        if __C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.img_feat_linear = nn.Linear(
            __C.IMG_FEAT_SIZE,
            __C.HIDDEN_SIZE
        )

        self.backbone = MCA_ED(__C)

        self.attflat_img = AttFlat(__C)
        self.attflat_lang = AttFlat(__C)

        self.fusion = ModalityAttentionFusion(__C, p=0.2)
        
        self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)


    def forward(self, img_feat, ques_ix):

        # Make mask
        lang_feat_mask = self.make_mask(ques_ix.unsqueeze(2))
        img_feat_mask = self.make_mask(img_feat)

        # Pre-process Language Feature
        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.lstm(lang_feat)

        # Pre-process Image Feature
        img_feat = self.img_feat_linear(img_feat)

        # Backbone Framework
        img_feat, lang_feat = self.backbone(
            img_feat,
            lang_feat,
            img_feat_mask,
            lang_feat_mask
        )                                                          #return x,y = img_feat,lang_feat

        """lang, img 고정 길이 벡터로 요약"""
        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )

        img_feat = self.attflat_img(
            img_feat,
            img_feat_mask
        )

        """fusion"""
        fused, w = self.fusion(img_feat, lang_feat)   # [B, d]
            
            
        """ Fusion + prediction"""
        proj_feat = self.proj_norm(fused)
        proj_feat = torch.sigmoid(self.proj(proj_feat))

        return proj_feat


    # Masking
    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)
