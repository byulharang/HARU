import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def get_1d_sincos_pos_embed(d_model, length):
    """
    d_model 차원의 1D sine-cosine positional encoding 생성
    :param d_model: 임베딩 차원
    :param length: 시퀀스 길이
    :return: [length, d_model] 텐서
    """
    pe = torch.zeros(length, d_model)
    pos = torch.arange(length, dtype=torch.float).unsqueeze(1)
    freq = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * freq)
    pe[:, 1::2] = torch.cos(pos * freq)
    return pe


def PE_offset(Np, patch_W, offset, d_model):
    # offset이 tensor가 아닌 경우 tensor로 변환
    if not isinstance(offset, torch.Tensor):
        offset = torch.tensor([offset], dtype=torch.float)
    B = offset.size(0)
    device = offset.device
    # 패치 그리드 인덱스 계산
    idx = torch.arange(Np, device=device)
    Hp = idx // patch_W
    Wp = idx % patch_W
    # offset(degree) -> 패치 단위 이동량
    dx = (offset / 360.0).unsqueeze(-1) * patch_W  # [B,1]
    dy = torch.zeros_like(dx)
    # 위치 좌표
    Hp_pos = Hp.float().unsqueeze(0).expand(B, -1) + dy  # [B, Np]
    Wp_pos = Wp.float().unsqueeze(0).expand(B, -1) + dx  # [B, Np]
    # d_model을 절반으로 나누어 pe_h, pe_w 각 half_dim 할당
    half_dim = d_model // 2
    # sin-cos pair 차원
    pair_dim = half_dim // 2
    # frequency 벡터
    dim_range = torch.arange(pair_dim, device=device).float()
    freq = 1.0 / (10000 ** (2 * dim_range / half_dim))  # [pair_dim]
    # height sin-cos encoding
    angles_h = Hp_pos.unsqueeze(-1) * freq            # [B, Np, pair_dim]
    pe_h = torch.cat([torch.sin(angles_h), torch.cos(angles_h)], dim=-1)  # [B, Np, half_dim]
    # width sin-cos encoding (offset 적용)
    angles_w = Wp_pos.unsqueeze(-1) * freq            # [B, Np, pair_dim]
    pe_w = torch.cat([torch.sin(angles_w), torch.cos(angles_w)], dim=-1)  # [B, Np, half_dim]
    # height/width 인코딩을 채널 방향으로 concat
    pe = torch.cat([pe_h, pe_w], dim=-1)              # [B, Np, d_model]
    return pe  # [B, Np, d_model]

class DistanceFiLM(nn.Module):
    """
    mic_distance 스칼라를 받아 d_model 크기의 scale(gamma)과 shift(beta) 벡터 생성
    """
    def __init__(self, d_model, hidden=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * d_model)
        )

    def forward(self, mic_dist):  # mic_dist: [B,1]
        gam_beta = self.mlp(mic_dist)
        gamma, beta = gam_beta.chunk(2, dim=-1)
        return gamma.unsqueeze(1), beta.unsqueeze(1)

class RIRproj_FiLM(nn.Module):
    def __init__(self, rir_in_channels, d_model):
        super().__init__()
        # in_channels = RIR feature 채널 개수, out_channels = d_model
        self.rir_conv = nn.Conv1d(rir_in_channels, d_model, kernel_size=1)
        self.dist_film = DistanceFiLM(d_model)

    def forward(self, rir_tokens, mic_dist):
        h = self.rir_conv(rir_tokens)              # [B, d_model, L]
        h = h.transpose(1, 2)                      # [B, L, d_model]
        gamma, beta = self.dist_film(mic_dist)     # [B,1,d_model]
        return gamma * h + beta                   # [B, L, d_model]

class CrossAttnFFNBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_ffn, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ffn),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_ffn, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, query, key_val):
        q = query.transpose(0, 1)       # -> [Nq, B, d_model]
        kv = key_val.transpose(0, 1)     # -> [Nk, B, d_model]
        attn_out, _ = self.mha(q, kv, kv)
        attn_out = attn_out.transpose(0, 1)
        x = self.norm1(query + self.dropout1(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out)

class SelfAttnFFNBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_ffn, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ffn),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_ffn, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x.transpose(0,1)
        attn_out, _ = self.mha(x, x, x)
        x = self.norm1(x + self.dropout1(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out)



class SelfCrossTR(nn.Module):
    def __init__(self, n_head, n_blocks, img_in_channels, rir_in_channels, d_model,
                 img_size, img_patch_size, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        H, W = img_size
        ps = img_patch_size
        self.P = ps
        self.img_Hp = H // ps
        self.img_Wp = W // ps
        self.img_Np = self.img_Hp * self.img_Wp
        self.unfold = nn.Unfold(kernel_size=ps, stride=ps)
        self.fold = nn.Fold(output_size=(H, W), kernel_size=ps, stride=ps)
        self.img_embed = nn.Linear(img_in_channels * ps * ps, d_model)
        self.img_unembed = nn.Linear(d_model, img_in_channels * ps * ps)
        self.img_embed_proj = nn.Linear(d_model, d_model -1) # due to mask token (++1 dimension)
        self.rir_embed = RIRproj_FiLM(rir_in_channels, d_model)
        self.rir_unembed_conv = nn.Conv1d(d_model, rir_in_channels, kernel_size=1) # need change to Linear
        self.input_norm_img  = nn.LayerNorm(d_model)  
        self.inter_norm_img  = nn.LayerNorm(d_model)
        self.output_norm_img = nn.LayerNorm(d_model)
        self.input_norm_rir  = nn.LayerNorm(d_model)  
        self.output_norm_rir = nn.LayerNorm(d_model)
        self.unembed_relu = nn.ReLU(inplace=True)
        self.self_blocks = nn.ModuleList([
            SelfAttnFFNBlock(d_model, n_head, dim_feedforward, dropout)
            for _ in range(n_blocks)
        ])
        self.cross_blocks = nn.ModuleList([
            CrossAttnFFNBlock(d_model, n_head, dim_feedforward, dropout)
            for _ in range(n_blocks)
        ])

    def forward(self, img_features, rir_features, mic_info, mask_pos):
        mic_degree = mic_info[:, 0]
        mic_dist = mic_info[:, 1].unsqueeze(-1)
        B, C, H, W = img_features.shape
        # 이미지 패치 임베딩 + 위치 인코딩
        patches = self.unfold(img_features)
        seq = patches.permute(0, 2, 1)
        seq = self.img_embed(seq)
        # Mask Token prepare
        mask_map = torch.zeros(B, H, W, device=seq.device, dtype=torch.bool)
        feature_scale   = 1024 // W
        mask_pos_scaled = mask_pos // feature_scale  # [B]
        width_scaled    = 256 // feature_scale
        for b in range(B):
            start = mask_pos_scaled[b].item()
            mask_map[b, :, start : start + width_scaled] = True
        
        mask_f = mask_map.float().unsqueeze(1)
        mask_avg = F.avg_pool2d(mask_f, kernel_size=self.P, stride=self.P)
        mask_ratio = mask_avg.view(B, -1)                       # pixel ratio masking 
        mask_feature = mask_ratio.unsqueeze(-1)                 # [B, N, 1]
        
        # IMG PE: Original + mic_degree
        pe_img = PE_offset(self.img_Np, self.img_Wp, mic_degree, seq.size(-1))
        seq = seq + pe_img.to(seq.device)
        seq = self.input_norm_img(seq) 
        
        # Self Attenion (Img)
        for block in self.self_blocks:
            seq = block(seq)
        seq = self.inter_norm_img(seq)
        
        # RIR PE: Original
        rir_seq = self.rir_embed(rir_features, mic_dist)
        pe_rir = get_1d_sincos_pos_embed(rir_seq.size(-1), rir_seq.size(1)).unsqueeze(0)
        rir_seq = rir_seq + pe_rir.to(rir_seq.device)
        rir_seq =self.input_norm_rir(rir_seq)
        
        seq = seq + pe_img.to(seq.device)
        seq = self.img_embed_proj(seq)                      # [B, N, D-1]
        
        # Mask token concat to Img
        seq = torch.cat([seq, mask_feature], dim=-1)        # [B, N, D]        
        
        # Cross Attention 
        for block in self.cross_blocks:
            seq = block(seq, rir_seq)
        
        
        # Fold patch for reconstruction
        out_patch = self.img_unembed(seq).permute(0, 2, 1)
        img_rec = self.fold(out_patch)
 
        # RIR reconstruction
        r = rir_seq.transpose(1, 2)
        r = self.rir_unembed_conv(r)
        r = self.unembed_relu(r)
        return img_rec, r
