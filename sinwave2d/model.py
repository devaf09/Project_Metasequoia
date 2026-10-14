"""SinWave の 2 次元版（実験用の全オプション版）。

    画像 → 透過マスク（振幅 / 位相）→ 照明 → 滲み層 → [Sin の板 → 回折 → 明るさを 1 に戻す] × D → レンズ → 窓

1 次元版（sinwave_minimal.py v1.0.0）との対応
    場 L          → 場 N×N（周期境界）
    区画 P 個     → 区画 P×P（区画ごとに A, φ と 2 次元の縞 (ωx, ωy)）
    角スペクトル法 → 2 次元の角スペクトル法（2 次元 FFT。行と列に分けて計算できる）
    レンズ        → 2 次元 FFT
    tilt 照明     → 画像を nb×nb のブロックに分け、ブロック番号に位相の傾き 2π(jx·bx + jy·by)/nb を与えた nb² 通り
                     （nb² 通りを平均すると、違うブロックどうしの干渉がちょうど 0 になる）
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class SinWave2D(nn.Module):
    def __init__(self, n_classes: int = 10, N: int = 64, scale: int = 2, layers: int = 16, patches: int = 8,
                 z: float = 32.0, input_mode: str = "amp", illum: str = "none", blocks: int = 4, patterns: int = 8,
                 smear: int = 0, freeze_w: bool = True, det_radius: float | None = None, det_size: int = 3,
                 bias: bool = True, seed: int = 0, chunk: int = 0, ref: float = 0.0, ref_tilt: float = 0.0):
        super().__init__()
        self.cfg = dict(n_classes=n_classes, N=N, scale=scale, layers=layers, patches=patches, z=z,
                        input_mode=input_mode, illum=illum, blocks=blocks, patterns=patterns, smear=smear,
                        freeze_w=freeze_w, det_radius=det_radius, det_size=det_size, bias=bias, seed=seed, ref=ref, ref_tilt=ref_tilt)
        self.N, self.scale, self.input_mode, self.smear, self.chunk = N, scale, input_mode, smear, chunk
        # 参照波（ホログラムの考え方）: 物体の光に、明るさ ref の平面波を重ねる。
        # 明るさ |参照 + 物体|² = |参照|² + |物体|² + 2 Re(参照*·物体) の最後の項が物体の場に線形になる。
        # ref_tilt = 0 は同軸（ガボール型）、> 0 は場の幅あたり ref_tilt 周期だけ斜めに入れる（軸外れ型）
        self.ref = ref
        yy0, xx0 = torch.meshgrid(torch.arange(N, dtype=torch.float32), torch.arange(N, dtype=torch.float32), indexing="ij")
        self.register_buffer("refwave", torch.polar(torch.full((N, N), float(ref)), 2 * math.pi * ref_tilt * (xx0 + yy0) / N))
        img = 28 * scale
        if img > N:
            raise ValueError("拡大した画像が場に収まらない")
        self.off = (N - img) // 2

        # --- Sin の板 ------------------------------------------------------------------------
        g = torch.Generator().manual_seed(seed)
        P2 = patches * patches
        lo, hi = math.log(2 * math.pi / N), math.log(math.pi)
        r = torch.rand(4, layers, P2, generator=g)
        mag = torch.exp(lo + (hi - lo) * r[1])                         # 縞の細かさ（対数一様）
        ang = 2 * math.pi * r[2]                                       # 縞の向き
        self.A = nn.Parameter(0.5 + 0.5 * r[0])
        self.phi = nn.Parameter(2 * math.pi * r[3])
        wx, wy = mag * torch.cos(ang), mag * torch.sin(ang)
        if freeze_w:
            self.register_buffer("wx", wx)
            self.register_buffer("wy", wy)
        else:
            self.wx, self.wy = nn.Parameter(wx), nn.Parameter(wy)
        idx = torch.arange(N) * patches // N
        self.register_buffer("pidx", (idx[:, None] * patches + idx[None, :]).reshape(-1))   # 画素 → 区画番号
        yy, xx = torch.meshgrid(torch.arange(N, dtype=torch.float32), torch.arange(N, dtype=torch.float32),
                                indexing="ij")
        self.register_buffer("xx", xx.reshape(-1))
        self.register_buffer("yy", yy.reshape(-1))

        # --- 回折（2 次元の角スペクトル法、波長 1 画素）---------------------------------------
        k = 2 * math.pi * torch.fft.fftfreq(N)
        kz = torch.sqrt((2 * math.pi) ** 2 - k[:, None] ** 2 - k[None, :] ** 2)
        self.register_buffer("H", torch.polar(torch.ones(N, N), kz * z))

        # --- 照明（固定）------------------------------------------------------------------------
        b = torch.clamp((torch.arange(N) - self.off) * blocks // img, 0, blocks - 1)  # 画素 → ブロック番号
        by, bx = b[:, None].float(), b[None, :].float()
        if illum == "none":
            psi = torch.zeros(1, N, N)
        elif illum == "tilt":                                          # nb² 通り、違うブロックの干渉が 0 になる
            j = torch.arange(blocks, dtype=torch.float32)
            psi = 2 * math.pi * (j[:, None, None, None] * bx + j[None, :, None, None] * by) / blocks
            psi = psi.reshape(-1, N, N)
        elif illum == "tiltx":                                         # nb 通り、横に違うブロックだけ干渉を消す
            j = torch.arange(blocks, dtype=torch.float32)
            psi = 2 * math.pi * j[:, None, None] * bx / blocks + 0 * by
        elif illum == "rand":                                          # ブロックごとの乱数位相 patterns 通り
            ph = 2 * math.pi * torch.rand(patterns, blocks, blocks, generator=g)
            psi = ph[:, b[:, None], b[None, :]]
        else:
            raise ValueError(illum)
        self.register_buffer("illum", torch.polar(torch.ones_like(psi), psi))   # (K, N, N)

        # --- 検出器の窓: レンズ後の面で、中心から半径 r の円周上に n_classes 個 ------------------------
        rad = det_radius if det_radius is not None else N / 4
        det = torch.zeros(n_classes, N, N)
        h = det_size // 2
        for c in range(n_classes):
            t = 2 * math.pi * c / n_classes
            cy, cx = int(round(N / 2 + rad * math.sin(t))), int(round(N / 2 + rad * math.cos(t)))
            det[c, cy - h:cy - h + det_size, cx - h:cx - h + det_size] = 1.0 / det_size ** 2
        self.register_buffer("det", det.reshape(n_classes, -1))
        self.gain = nn.Parameter(torch.tensor(5.0))
        self.bias = nn.Parameter(torch.zeros(n_classes)) if bias else None

    # -----------------------------------------------------------------------------------------
    def field(self, x: torch.Tensor) -> torch.Tensor:
        """画像 (B, 28, 28) → 入力面の光の場 (B, N, N)。"""
        x = x.repeat_interleave(self.scale, 1).repeat_interleave(self.scale, 2)
        m = F.pad(x, (self.off, self.N - self.off - x.shape[2], self.off, self.N - self.off - x.shape[1]))
        if self.input_mode == "amp":                                   # 明るい画素ほど光を通す
            u = torch.complex(m, torch.zeros_like(m))
            return u + self.refwave if self.ref else u
        if self.input_mode == "phase":                                 # 明るい画素ほど位相が遅れる（光量は一様）
            return torch.polar(torch.ones_like(m), math.pi * m)
        if self.input_mode == "both":
            return torch.polar(m, math.pi * m)
        raise ValueError(self.input_mode)

    def _propagate(self, u: torch.Tensor) -> torch.Tensor:
        """u (B, N, N) → レンズ後の明るさ (B, N*N)。"""
        if self.smear:                                                 # 滲み層: 上下左右に smear 画素ずらした場との平均
            s = self.smear
            u = (u + u.roll(s, 1) + u.roll(-s, 1) + u.roll(s, 2) + u.roll(-s, 2)) / 5
        B = u.shape[0]
        u = u.reshape(B, -1)
        for d in range(self.A.shape[0]):
            p = self.pidx
            phase = self.A[d, p] * torch.sin(self.wx[d, p] * self.xx + self.wy[d, p] * self.yy + self.phi[d, p])
            u = u * torch.polar(torch.ones_like(phase), phase)
            u = torch.fft.ifft2(torch.fft.fft2(u.reshape(B, self.N, self.N)) * self.H).reshape(B, -1)
            u = u / (u.abs().pow(2).mean(-1, keepdim=True).sqrt() + 1e-12)
        u = torch.fft.fftshift(torch.fft.fft2(u.reshape(B, self.N, self.N), norm="ortho"), dim=(-2, -1))
        return u.abs().pow(2).reshape(B, -1)

    def intensity(self, x: torch.Tensor) -> torch.Tensor:
        """画像 → レンズ後の明るさ (B, N*N)、照明パターンについて平均、平均 1 に正規化。"""
        u0 = self.field(x)
        K = self.illum.shape[0]
        if self.chunk:                                                 # 省メモリ: 照明パターンを chunk 個ずつ、勾配は再計算
            acc = 0
            for k in range(0, K, self.chunk):
                il = self.illum[k:k + self.chunk]
                f = lambda u, il=il: self._propagate((u[:, None] * il[None]).reshape(-1, self.N, self.N)) \
                    .reshape(u.shape[0], il.shape[0], -1).sum(1)
                acc = acc + (checkpoint(f, u0, use_reentrant=False) if torch.is_grad_enabled() else f(u0))
            inten = acc / K
        else:
            B = u0.shape[0]
            inten = self._propagate((u0[:, None] * self.illum[None]).reshape(-1, self.N, self.N)) \
                .reshape(B, K, -1).mean(1)
        return inten / (inten.mean(-1, keepdim=True) + 1e-12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.gain * (self.intensity(x) @ self.det.T)
        return out + self.bias if self.bias is not None else out

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
