"""Draft-only k-bit lm_head copies for MTP proposals (draft_head.cuh, ops.DraftHead / draft_logits; T-051).

The copies only choose which candidates get exact BF16N logits, so their contract is: each weight reads back as the
documented level, the GEMV computes the product with what it stores (FP16 partial sums, approximate by design), and
draft_logits gives every candidate its exact FP32 logit and everything else -inf."""
import pytest
import torch

from conftest import encode_upload, weights_like_model

HP = 3  # FMT_HUFPAIR


def readout(ops, bits, q, s, K):
    """The stored matrix [rows, K] as the GEMV sees it: one-hot inputs, four columns per call (exact in FP16)."""
    gemv = ops.int2_gemv if bits == 2 else ops.int4_gemv
    cols = []
    for c0 in range(0, K, 4):
        x = torch.zeros(4, K, dtype=torch.bfloat16, device="cuda")
        x[torch.arange(4), torch.arange(c0, c0 + 4)] = 1.0
        cols.append(gemv(x, q, s))
    return torch.cat(cols).t()  # [rows, K]


def int2_case(K, seed=0):
    """Rows of K weights whose 128-groups hold the level boundaries exactly (s = 1: -1.5, -1, 0, 1, 1.5), an all-zero
    group, and model-like random groups."""
    w = weights_like_model((8, K), seed=seed).float()
    w[0, :128] = 0.0
    w[1, :8] = torch.tensor([-1.5, -1.0, -0.75, -0.5, 0.0, 0.5, 1.0, 1.5])
    w[1, 8:128] = torch.linspace(-1.4, 1.4, 120)
    return w.to(torch.bfloat16)


@pytest.mark.parametrize("K", [2048])
def test_int2_weights_read_back_as_nearest_level(ops, K):
    w = int2_case(K)
    q, s = ops.int2_quantize(w.cuda())
    d = readout(ops, 2, q, s, K).double().cpu()
    # the stored spacing s per group must be amax / 1.5 (FP32), and every weight the nearest level of that spacing
    g = w.float().view(w.shape[0], -1, 128)
    assert torch.equal(s.cpu(), g.abs().amax(-1) / 1.5)                         # correctly rounded
    step = s.cpu().double()[:, :, None].expand_as(g).reshape(w.shape)
    wd = w.double()
    nz = step > 0
    assert torch.equal(d[~nz], torch.zeros_like(d[~nz]))                       # all-zero group: exactly 0
    r2 = d[nz] / step[nz] * 2                                                   # (n - 1.5) s, rounded once to FP32
    assert torch.isin(r2.round(), torch.tensor([-3.0, -1.0, 1.0, 3.0], dtype=torch.float64)).all()
    assert (r2 - r2.round()).abs().max() < 1e-6                                 # one of -1.5 s, -0.5 s, 0.5 s, 1.5 s
    assert ((wd - d).abs() <= step / 2 * (1 + 1e-6))[nz].all()                 # the nearest level
    wf = w.float()
    sf = step.float()
    ties = nz & ((wf == 0) | (wf == sf) | (wf == -sf))                          # exactly on a level boundary
    assert ties[1, :8].sum() == 3                                               # -1, 0, 1 of the s = 1 group
    assert torch.allclose(d[ties], wd[ties] + step[ties] / 2, rtol=1e-6, atol=0)  # ties go to the upper level


@pytest.mark.parametrize("bits, K", [(2, 2048), (4, 2048), (4, 4096)])
@pytest.mark.parametrize("T", [1, 2, 3, 4])
def test_kbit_gemv_is_the_product_with_the_stored_matrix(ops, bits, K, T):
    w = weights_like_model((61, K), seed=K + T)  # 61 rows: not a multiple of the 8 rows per block
    q, s = (ops.int2_quantize if bits == 2 else ops.int4_quantize)(w.cuda())
    d = readout(ops, bits, q, s, K).double()
    x = (torch.randn(T, K, generator=torch.Generator().manual_seed(T)) * 2).to(torch.bfloat16).cuda()
    y = (ops.int2_gemv if bits == 2 else ops.int4_gemv)(x, q, s).double()
    ref = x.double() @ d.t()
    bound = x.double().abs() @ d.abs().t()
    # FP16 partial sums of 16 (2-bit) / 32 (4-bit) products; a wrong column, lane or scale mapping is off by O(bound)
    assert ((y - ref).abs() <= 2e-2 * bound + 1e-6).all()


@pytest.mark.parametrize("fo", [False, True])
def test_draft_logits_rescore_every_candidate_exactly(ops, fo):
    """fo: the lm_head in fragment order (production since T-080): gathered rows are the same bits, so the draft
    logits are bitwise those of the row-major head."""
    V, K = 4096, 2048
    w = weights_like_model((V, K), seed=7)
    h = (torch.randn(3, K, generator=torch.Generator().manual_seed(3)) * 3).to(torch.bfloat16)
    w[777] = (h[2].float() / h[2].float().norm() * 0.5).to(torch.bfloat16)    # row 2: a clear exact winner
    enc = encode_upload(ops, w, HP, 16, False, True) if fo else encode_upload(ops, w, HP, 4)
    q, s = ops.int2_quantize(w.cuda())
    head = ops.DraftHead(2, q, s, 128)
    h = h.cuda()
    out = ops.draft_logits(h, enc, head, V)
    if fo:
        assert torch.equal(out, ops.draft_logits(h, encode_upload(ops, w, HP, 4), head, V))
    cand = head.scores(h).topk(128, dim=-1).indices
    finite = torch.isfinite(out)
    assert torch.equal(finite.sum(1), torch.full((3,), 128, device="cuda"))
    assert finite.gather(1, cand).all()                                         # finite exactly at the candidates
    exact = h.double() @ w.cuda().double().t()                                  # [3, V]
    S = h.double().abs() @ w.cuda().double().abs().t()
    got = out.gather(1, cand).double()
    assert ((got - exact.gather(1, cand)).abs() <= 1e-5 * S.gather(1, cand)).all()   # exact FP32 re-scoring
    best_cand = cand.gather(1, exact.gather(1, cand).argmax(1, keepdim=True)).squeeze(1)
    assert torch.equal(out.argmax(1), best_cand)                                # the exact best of the candidates
    assert out.argmax(1)[2].item() == 777 == exact.argmax(1)[2].item()          # a clear winner is never lost


@pytest.mark.parametrize("bits, qcols, scols, cand, ok", [
    (2, 512, 16, 1, True), (2, 512, 16, 64, True), (2, 512, 16, 65, False), (2, 512, 16, 0, False),
    (4, 1024, 16, 32, True), (4, 512, 16, 32, False), (3, 768, 16, 32, False),
])
def test_draft_head_rejects_inconsistent_shapes(ops, bits, qcols, scols, cand, ok):
    q = torch.zeros(64, qcols, dtype=torch.uint8)
    s = torch.zeros(64, scols)
    if ok:
        assert ops.DraftHead(bits, q, s, cand).candidates == cand
    else:
        with pytest.raises(ValueError):
            ops.DraftHead(bits, q, s, cand)


@pytest.mark.parametrize("env, bits", [(None, 2), ("int2", 2), ("int4", 4), ("full", None)])
def test_draft_head_choice(ops, monkeypatch, env, bits):
    if env is None:
        monkeypatch.delenv("BF16N_DRAFT_HEAD", raising=False)
    else:
        monkeypatch.setenv("BF16N_DRAFT_HEAD", env)
    assert ops.draft_head_bits() == bits


@pytest.mark.parametrize("env", ["int8", "INT2", ""])
def test_draft_head_choice_rejects_unknown_values(ops, monkeypatch, env):
    monkeypatch.setenv("BF16N_DRAFT_HEAD", env)
    with pytest.raises(ValueError):
        ops.draft_head_bits()


@pytest.mark.parametrize("bits, env, cand", [(2, None, 128), (4, None, 32), (2, "16", 16)])
def test_make_draft_head_candidates(ops, monkeypatch, bits, env, cand):
    if env is None:
        monkeypatch.delenv("BF16N_DRAFT_CANDIDATES", raising=False)
    else:
        monkeypatch.setenv("BF16N_DRAFT_CANDIDATES", env)
    head = ops.make_draft_head(weights_like_model((256, 2048), seed=1).cuda(), bits)
    assert (head.bits, head.candidates) == (bits, cand)


@pytest.mark.parametrize("K", [1024, 4096])
def test_int2_gemv_rejects_other_widths(ops, K):
    q, s = ops.int2_quantize(weights_like_model((16, K), seed=2).cuda())
    with pytest.raises(Exception, match="K must be 2048"):
        ops.int2_gemv(torch.zeros(1, K, dtype=torch.bfloat16, device="cuda"), q, s)
