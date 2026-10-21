import pytest

torch = pytest.importorskip("torch")

from trajrisk.diffrec_model import DIFFREC_DIR, build_diffrec, recommend_scores  # noqa: E402
from trajrisk.masking import mask_history, masked_signals  # noqa: E402

pytestmark = pytest.mark.skipif(not DIFFREC_DIR.exists(), reason="external/DiffRec missing")

CFG = {"model": {"dims": [16], "emb_size": 4, "steps": 5, "noise_schedule": "linear-var",
                 "noise_scale": 0.0001, "noise_min": 0.0005, "noise_max": 0.005}}


def _setup(n_items=60, batch=8):
    torch.manual_seed(0)
    model, diffusion = build_diffrec(CFG, n_items, "cpu")
    model.eval()
    history = (torch.rand(batch, n_items) < 0.3).float()
    history[0] = 0
    history[0, 3] = 1                      # a user with a single liked item
    seen = history > 0
    targets = torch.zeros(batch, dtype=torch.long)
    return model, diffusion, history, seen, targets


def test_mask_only_removes_items_and_never_empties_a_row():
    _, _, h, _, _ = _setup()
    torch.manual_seed(1)
    m = mask_history(h, 0.9)
    assert (m <= h).all()
    assert (m.sum(-1) > 0).all()


def test_no_masking_means_no_change():
    model, diffusion, h, seen, t = _setup()
    sig = masked_signals(model, diffusion, h, seen, t, n_masks=3, rate=0.0)
    assert sig["mask_change@10"].abs().max() == 0
    assert sig["mask_unstable@10"].abs().max() == 0


def test_judged_list_is_the_normal_recommendation():
    model, diffusion, h, seen, t = _setup()
    full = recommend_scores(model, diffusion, h)
    top10 = full.masked_fill(seen, float("-inf")).topk(10, dim=-1).indices
    expected_miss = (~(top10 == t[:, None]).any(-1)).float()
    sig = masked_signals(model, diffusion, h, seen, t, n_masks=3, rate=0.5)
    assert torch.equal(sig["miss"], expected_miss)


def test_single_item_history_is_never_masked():
    model, diffusion, h, seen, t = _setup()
    sig = masked_signals(model, diffusion, h, seen, t, n_masks=5, rate=0.9)
    assert sig["mask_change@10"][0] == 0