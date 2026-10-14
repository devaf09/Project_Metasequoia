import torch

from sinwave2d import SinWave2D


def _imgs(n=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 28, 28, generator=g)


def test_shapes_and_params():
    m = SinWave2D(N=64, layers=4, patches=4)
    out = m(_imgs())
    assert out.shape == (3, 10)
    assert m.n_params() == 4 * 16 * 2 + 1 + 10


def test_chunk_matches_full():
    """照明パターンを少しずつ計算する省メモリ実行は、まとめて計算したのと同じ出力・同じ勾配になる。"""
    x = _imgs()
    a = SinWave2D(N=64, layers=3, patches=4, illum="tilt", blocks=2)
    b = SinWave2D(N=64, layers=3, patches=4, illum="tilt", blocks=2, chunk=1)
    b.load_state_dict(a.state_dict())
    ya, yb = a(x), b(x)
    assert torch.allclose(ya, yb, atol=1e-5)
    ya.sum().backward()
    yb.sum().backward()
    assert torch.allclose(a.phi.grad, b.phi.grad, atol=1e-4)


def test_tilt_cancels_interference_between_blocks():
    """tilt 照明 nb² 通りの平均は、違うブロックどうしの干渉が消え、ブロックごとの明るさの和になる。
    （増幅で明るさを戻す処理は非線形なので、確かめるときは 1 層・増幅なしの線形な光学系で見る）"""
    torch.manual_seed(0)
    m = SinWave2D(N=64, layers=1, patches=4, illum="tilt", blocks=2)
    x = _imgs(1)

    def raw(xx):
        u0 = m.field(xx)
        u = u0[:, None] * m.illum[None]
        p = m.pidx
        ph = (m.A[0, p] * torch.sin(m.wx[0, p] * m.xx + m.wy[0, p] * m.yy + m.phi[0, p])).reshape(64, 64)
        u = torch.fft.ifft2(torch.fft.fft2(u * torch.polar(torch.ones_like(ph), ph)) * m.H)
        return torch.fft.fft2(u).abs().pow(2).mean(1)

    with torch.no_grad():
        whole = raw(x)
        parts = 0
        for by in range(2):
            for bx in range(2):
                xm = torch.zeros_like(x)
                xm[:, by * 14:(by + 1) * 14, bx * 14:(bx + 1) * 14] = x[:, by * 14:(by + 1) * 14, bx * 14:(bx + 1) * 14]
                parts = parts + raw(xm)
    assert torch.allclose(whole, parts, rtol=1e-3, atol=1e-3 * whole.max().item())
