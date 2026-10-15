import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sinwave2d_minimal as sw  # noqa: E402
from sinwave2d import SinWave2D  # noqa: E402


def _imgs(n=4, seed=0):
    return torch.rand(n, 28, 28, generator=torch.Generator().manual_seed(seed))


def test_matches_full_version():
    """最小版は、実験用の全オプション版と同じ設定・同じ seed なら同じ出力になる。"""
    a = sw.SinWave2DNet(N=32, layers=3, patches=4, smear=1, seed=5)
    b = SinWave2D(N=32, scale=1, layers=3, patches=4, smear=1, seed=5)
    x = _imgs()
    assert torch.allclose(a(x), b(x), atol=1e-5)


def test_merge_and_remove():
    """融合は 2 つの光路のスコアの平均。外すと元の技能と完全に同じ。"""
    a = sw.SinWave2DNet(own=[0, 1, 2, 3, 4], layers=2, patches=4)
    b = sw.SinWave2DNet(own=[5, 6, 7, 8, 9], layers=2, patches=4, seed=1)
    x = _imgs()
    m = sw.merge([a, b])
    assert torch.allclose(m(x), (a(x) + b(x)) / 2)
    assert torch.equal(sw.scores(sw.merge([a]), x), sw.scores(a, x))
    assert sw.scores(a, x)[:, 5:].isinf().all()                       # 担当外の窓は選ばれない
    with pytest.raises(ValueError):
        sw.merge([a, a])                                              # 担当が重なる


def test_save_load(tmp_path):
    a = sw.SinWave2DNet(own=[1, 3], layers=2, patches=4, seed=2)
    with torch.no_grad():
        a.phi.add_(0.3)
    p = tmp_path / "skill.pt"
    sw.save(a, str(p))
    b = sw.load(str(p))
    x = _imgs()
    assert b.own == [1, 3]
    assert torch.allclose(a(x), b(x))


def test_train_rejects_foreign_labels():
    a = sw.SinWave2DNet(own=[0, 1], layers=1, patches=2)
    with pytest.raises(ValueError):
        sw.train(a, _imgs(2), torch.tensor([0, 2]), steps=1)
