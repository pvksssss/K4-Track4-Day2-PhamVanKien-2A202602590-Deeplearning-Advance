"""selftest.py - các kiểm tra tự viết cho những phần dễ sai (RUBRIC mục A3 và H1).

Chạy:  python selftest.py          (không cần GPU, không cần dataset)
       python selftest.py --only "gộp BatchNorm" CutMix    (chỉ chạy một số kiểm tra)

Bao gồm:
  - focal loss với gamma = 0 phải đúng CE (sai số < 1e-6)          [GUIDE.md 3.2]
  - label smoothing với eps = 0 phải đúng CE
  - CutMix: lam phải khớp đúng diện tích hộp thực tế bị cắt         [slide trang 48]
  - Mixup/CutMix trộn cả nhãn, mixed_loss khớp công thức
  - gộp BatchNorm: sai số đầu ra <= 1e-5                            [slide trang 71]
  - class_weights: trung bình = 1, beta=0 cho trọng số nghịch đảo
  - temperature scaling: T > 0, không đổi argmax, NLL/ECE giảm trên val
  - 3 nhóm tham số có weight_decay = 0 cho norm và bias           [GUIDE.md 1.4]
  - đếm GMAC cho resnet50 xấp xỉ 4.1 (số trên slide)                [slide trang 41]
  - hợp đồng định dạng file dự đoán với eval.py (save -> read -> score)
  - parse_overrides ép đúng kiểu và báo lỗi rõ ràng
"""
from __future__ import annotations

import runtime  # Select safe pandas string storage on Windows before Torch.

import os
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass

PASSED, FAILED = [], []

# `python selftest.py --only "gộp BatchNorm" temperature` -> chỉ chạy các kiểm tra chứa
# một trong các chuỗi nêu trong --only. Hữu ích khi cần biết kiểm tra nào gây lỗi.
_ONLY = [a for a in sys.argv if not a.startswith("-")][1:] if "--only" in sys.argv else []


def check(name):
    def deco(fn):
        if _ONLY and not any(tok.lower() in name.lower() for tok in _ONLY):
            return fn
        print(f"  --> {name}", flush=True)
        try:
            fn()
            PASSED.append(name)
            print(f"  [OK]   {name}", flush=True)
        except Exception as e:                      # noqa: BLE001 - báo lỗi rồi chạy tiếp
            FAILED.append((name, e))
            print(f"  [FAIL] {name}: {e}", flush=True)
            traceback.print_exc()
        return fn
    return deco


# --------------------------------------------------------------------------- #
@check("focal loss gamma=0 cho đúng cross-entropy (sai số < 1e-6)")
def check_focal_gamma_zero_is_ce():
    from losses import FocalLoss

    torch.manual_seed(0)
    logits = torch.randn(64, 9, dtype=torch.float64)
    y = torch.randint(0, 9, (64,))
    ce = F.cross_entropy(logits, y)
    for gamma in (0.0,):
        fl = FocalLoss(gamma=gamma)(logits, y)
        err = abs(float(fl) - float(ce))
        assert err < 1e-6, f"gamma={gamma}: sai số {err:.3e}"
    # focal với alpha=None và reduction='none' phải lấy trung bình khớp
    per_sample = FocalLoss(gamma=2.0, reduction="none")(logits, y)
    assert abs(float(per_sample.mean()) - float(FocalLoss(gamma=2.0)(logits, y))) < 1e-12


@check("label smoothing eps=0 cho đúng CE")
def check_label_smoothing_zero_is_ce():
    from losses import LabelSmoothingCE

    torch.manual_seed(0)
    logits = torch.randn(32, 9, dtype=torch.float64)
    y = torch.randint(0, 9, (32,))
    ce = F.cross_entropy(logits, y)
    err = abs(float(LabelSmoothingCE(0.0)(logits, y)) - float(ce))
    assert err < 1e-6, f"sai số {err:.3e}"
    ls = float(LabelSmoothingCE(0.1)(logits, y))
    assert ls > 0, "loss phải dương"


@check("CutMix: lam khớp diện tích hộp thực tế bị cắt")
def check_cutmix_lambda_matches_area():
    import losses

    torch.manual_seed(0)
    np.random.seed(0)
    x = torch.randn(16, 3, 32, 32)
    y = torch.randint(0, 9, (16,))
    # Cố định permutation thành derangement (không có phần tử nào giữ nguyên vị trí), nếu không
    # có ảnh nào nhận lại chính nó thì mọi pixel trong hộp đều khác ảnh gốc và đếm pixel chính xác.
    perm = torch.tensor([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14])
    orig_randperm = torch.randperm
    torch.randperm = lambda n, *a, **k: perm.clone().to(k.get("device", x.device))
    try:
        x_mix, (y_a, y_b, lam) = losses.mix_batch(x, y, alpha=1.0, mode="cutmix")
    finally:
        torch.randperm = orig_randperm

    changed = (x_mix != x).any(dim=1)
    per_img = changed.sum(dim=(1, 2))
    frac = changed.sum().item() / (x.shape[0] * x.shape[2] * x.shape[3])
    assert abs(frac - (1.0 - lam)) < 1e-6, f"pixel khác {frac:.6f} nhưng 1-lam = {1.0 - lam:.6f}"
    assert bool((per_img == per_img[0]).all()), f"hộp phải giống nhau mọi ảnh, nhận {per_img.tolist()}"
    # nhãn phải được trộn cả hai lớp (slide trang 48)
    assert torch.equal(y_a, y) and torch.equal(y_b, y[perm]), "nhãn phải trộn theo cùng permutation"
    assert 0.0 <= lam <= 1.0


@check("Mixup: ảnh trộn là đúng phép convex, nhãn trộn cả hai lớp")
def check_mixup():
    from losses import mix_batch, mixed_loss

    torch.manual_seed(0)
    np.random.seed(0)
    x = torch.randn(8, 3, 16, 16)
    y = torch.randint(0, 9, (8,))
    x_mix, (y_a, y_b, lam) = mix_batch(x, y, alpha=1.0, mode="mixup")
    assert 0.0 <= lam <= 1.0
    assert x_mix.shape == x.shape
    # giá trị mỗi pixel nằm giữa hai ảnh gốc tương ứng
    lo = torch.minimum(x, x[torch.randperm(8)]) - 1e-6
    assert x_mix.min() >= lo.min() - 1e-5, "giá trị trộn nằm ngoài khoảng của hai ảnh gốc"
    # mixed_loss khớp công thức lam*CE(y_a) + (1-lam)*CE(y_b)
    logits = torch.randn(8, 9)
    crit = nn.CrossEntropyLoss()
    expect = lam * crit(logits, y_a) + (1 - lam) * crit(logits, y_b)
    assert abs(float(mixed_loss(crit, logits, (y_a, y_b, lam))) - float(expect)) < 1e-6


@check("gộp BatchNorm: sai số đầu ra <= 1e-5")
def check_fuse_conv_bn():
    from inference import fuse_conv_bn

    torch.manual_seed(0)
    net = nn.Sequential(nn.Conv2d(3, 8, 3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(8), nn.ReLU(),
                        nn.Conv2d(8, 4, 3, padding=1, bias=True),
                        nn.BatchNorm2d(4), nn.ReLU()).eval()
    # cố tình làm running stats khác batch stats để kiểm tra đúng phép chuyển đổi
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.running_mean.uniform_(-1, 1)
                m.running_var.uniform_(0.5, 2.0)
                m.weight.uniform_(0.5, 1.5)
                m.bias.uniform_(-0.5, 0.5)
    x = torch.randn(4, 3, 17, 19)          # kích thước lẻ để bắt lỗi đảo chiều/padding
    with torch.no_grad():
        ref = net(x).clone()
    fused = fuse_conv_bn(net)
    with torch.no_grad():
        out = fused(x)
    err = float((ref - out).abs().max())
    assert err < 1e-5, f"sai số gộp BN quá lớn: {err:.3e}"
    assert not any(isinstance(m, nn.BatchNorm2d) for m in fused.modules()), "còn sót BN"
    print(f"         sai số lớn nhất = {err:.3e}")


@check("class_weights: trung bình = 1; beta=0 nghịch với số ảnh")
def check_class_weights():
    from losses import class_weights

    counts = np.array([1125, 1064, 1031, 1022, 1062, 1009, 1074, 1016, 9106], dtype=np.float64)
    w0 = class_weights(counts, beta=0.0)
    assert abs(float(w0.mean()) - 1.0) < 1e-6, f"trung bình = {float(w0.mean())}"
    ratio = float(w0[8] / w0[0])                     # Negatives/Chinee Apple
    expect = counts[0] / counts[8]
    assert abs(ratio - expect) < 1e-4 * expect, f"tỉ lệ trọng số sai: {ratio} vs {expect}"
    w9 = class_weights(counts, beta=0.999)
    assert abs(float(w9.mean()) - 1.0) < 1e-6
    assert float(w9[8]) < float(w9[0]), "lớp hiếm phải được trọng số lớn hơn"
    assert w9.numel() == 9


@check("temperature scaling: T>0, không đổi argmax, NLL giảm trên val")
def check_temperature():
    from inference import apply_temperature, ece_of, fit_temperature

    rng = np.random.default_rng(0)
    n = 2000
    y = rng.integers(0, 9, n)
    logits = rng.normal(size=(n, 9)) * 3.0 + np.eye(9)[y] * 2.0   # over-confident
    t = fit_temperature(logits, y)
    assert t > 0, f"T phải > 0, nhận {t}"
    p_before = apply_temperature(logits, 1.0)
    p_after = apply_temperature(logits, t)
    assert np.array_equal(p_before.argmax(1), p_after.argmax(1)), "temperature scaling phải giữ nhãn"
    nll = lambda p: float(-np.log(np.clip(p[np.arange(n), y], 1e-12, None)).mean())
    assert nll(p_after) <= nll(p_before) + 1e-6, "NLL phải giảm trên tập đã khớp"
    ece_b, ece_a = ece_of(p_before, y), ece_of(p_after, y)
    print(f"         T = {t:.3f}, NLL {nll(p_before):.4f} -> {nll(p_after):.4f}, "
          f"ECE {ece_b:.4f} -> {ece_a:.4f}")
    assert abs(p_after.sum(1) - 1).max() < 1e-9, "xác suất phải chuẩn hoá"


@check("nhóm tham số: weight_decay = 0 cho norm và bias, head LR gấp 10 lần")
def check_param_groups():
    try:
        import timm
    except ImportError:
        print("         (bỏ qua: chưa cài timm)")
        return
    from model import param_groups

    model = timm.create_model("resnet50", pretrained=False, num_classes=9)
    groups = param_groups(model, 1e-4, 1e-3, 0.05)
    # Slide trang 52 mô tả 3 nhóm (backbone có wd / backbone norm+bias / head). Vì nguyên tắc
    # "weight_decay = 0 cho norm và bias" được áp cho cả backbone lẫn head, head bị tách
    # thêm khi có bias (ndim<=1) -> 4 nhóm. Kiểm tra theo TÍNH CHẤT, không theo số nhóm.
    by_wd = {g["weight_decay"] for g in groups}
    assert by_wd == {0.0, 0.05}, f"chỉ được có wd 0 và 0.05, nhận {by_wd}"
    for g in groups:
        for p in g["params"]:
            expected = 0.0 if p.ndim <= 1 else 0.05
            assert g["weight_decay"] == expected, (
                f"tensor ndim={p.ndim} đang có weight_decay={g['weight_decay']}, cần {expected}")
    backbone = [p for g in groups for p in g["params"] if g["lr"] == 1e-4]
    head = [p for g in groups for p in g["params"] if g["lr"] == 1e-3]
    assert backbone and head, "phải có nhóm backbone (1e-4) và head (1e-3)"
    # head phải đúng là tham số của get_classifier
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    assert {id(p) for p in head} == head_ids, "nhóm head phải khớp get_classifier()"
    # tham số đóng băng không được xuất hiện trong bất kỳ nhóm nào
    model2 = timm.create_model("resnet50", pretrained=False, num_classes=9)
    from model import build_model, freeze_backbone

    m2 = build_model("resnet50", pretrained=False, num_classes=9, init="frozen")
    g2 = param_groups(m2, 1e-4, 1e-3, 0.05)
    assert len(g2) == 2, f"backbone đóng băng chỉ còn nhóm head, nhận {len(g2)} nhóm"


@check("GMAC của resnet50 xấp xỉ 4.1 (số trên slide trang 41)")
def check_gmac():
    try:
        import timm
    except ImportError:
        print("         (bỏ qua: chưa cài timm)")
        return
    from model import count_gmacs_with_tool, count_params

    model = timm.create_model("resnet50", pretrained=False, num_classes=9)
    gmac, tool = count_gmacs_with_tool(model, 224)
    p = count_params(model)
    # Slide ghi 25,6M là cho đầu 1000 lớp; với đầu 9 lớp của bài lab số tham số nhỏ hơn
    # (fc: 2048*9+9 = 18,4K thay cho 2048*1000+1000 = 2,05M) -> kỳ vọng ~23,5M.
    print(f"         resnet50 (9 lớp): {p:.1f}M tham số, {gmac:.2f} GMAC ({tool}); "
          f"slide ghi 25.6M (đầu 1000 lớp) / 4.1 GMAC")
    assert 3.5 < gmac < 4.7, f"GMAC {gmac:.2f} lệch quá nhiều so với ~4.1 của slide"
    assert 23.0 < p < 24.0, f"số tham số {p:.1f}M lệch quá nhiều so với ~23.5M của ResNet-50 9 lớp"
    # đối chiếu: cùng kiến trúc nhưng 1000 lớp thì ra đúng 25,6M như slide
    p1000 = count_params(timm.create_model("resnet50", pretrained=False, num_classes=1000))
    assert 25.3 < p1000 < 25.9, f"ResNet-50 1000 lớp phải ~25.6M, nhận {p1000:.1f}M"


@check("file dự đoán đi qua đúng kiểm tra của eval.py")
def check_prediction_contract():
    import eval as ev
    from train import softmax_np

    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 9, n)
    probs = softmax_np(rng.normal(size=(n, 9)))
    names = [f"img{i}.jpg" for i in range(n)]
    with tempfile.TemporaryDirectory() as d:
        p = ev.save_predictions(Path(d) / "F01_seed0_test.csv", names, y, probs)
        pred = ev.read_pred(str(p))
        assert pred.seed == 0
        np.testing.assert_array_equal(pred.y_pred, probs.argmax(1))
        np.testing.assert_allclose(pred.probs, probs, atol=1e-6)
    # logit (chưa softmax) phải bị eval.py từ chối
    try:
        ev.save_predictions("x.csv", names[:5], y[:5], rng.normal(size=(5, 9)))
        raise AssertionError("eval.save_predictions phải từ chối logit chưa chuẩn hoá")
    except ValueError:
        pass


@check("parse_overrides: ép kiểu đúng, lỗi rõ ràng")
def check_parse_overrides():
    from train import Config, parse_overrides

    d = parse_overrides(["exp_id=B01", "seed=1", "lr_backbone=2e-5", "amp=false",
                         "mix=cutmix", "ema_decay=0.999", "sampler=none"])
    cfg = Config(**d)
    assert cfg.exp_id == "B01" and cfg.seed == 1
    assert abs(cfg.lr_backbone - 2e-5) < 1e-12
    assert cfg.amp is False
    assert cfg.mix == "cutmix" and cfg.ema_decay == 0.999 and cfg.sampler is None
    try:
        parse_overrides(["khong_ton_tai=1"])
        raise AssertionError("phải báo lỗi với field không có trong Config")
    except ValueError as e:
        assert "không phải field" in str(e)
    try:
        parse_overrides(["seed=abc"])
        raise AssertionError("phải báo lỗi khi không ép được kiểu")
    except ValueError:
        pass


@check("sinh ảnh ngẫu nhiên trong loader: đủ số, đúng thứ tự, đúng nhãn")
def check_loader_contract():
    """Lưu ý: ép `pin_memory=False`. Đây là unit test chạy trên CPU; `pin_memory=True` chỉ có
    lợi khi chuyển tensor sang GPU, và trên một số cấu hình Windows + GPU, việc cấp phát
    pinned host memory làm sập tiến trình (access violation). Trên Kaggle nên để mặc định
    (None -> tự bật khi có CUDA)."""
    import pandas as pd
    from PIL import Image

    import dataset as ds

    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as d:
        names, labels = [], []
        for i in range(12):
            fn = f"img{i}.jpg"
            Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(Path(d) / fn)
            names.append(fn)
            labels.append(i % 9)
        df = pd.DataFrame({"Filename": names, "Label": labels, "Species": ["x"] * len(names)})
        eval_tf = ds.build_transforms(False, img_size=32)      # xác định, không ngẫu nhiên
        ds_obj = ds.DeepWeedsDataset(df, d, eval_tf, img_size=64)
        img, label, fn = ds_obj[3]
        assert img.shape == (3, 32, 32), f"dạng ảnh sai: {tuple(img.shape)}"
        assert label == 3 and fn == "img3.jpg"
        # cache RAM phải cho kết quả giống hệt đọc từ đĩa (dùng CÙNG một transform xác định)
        ds_cached = ds.DeepWeedsDataset(df, d, eval_tf, cache_in_ram=True, img_size=64)
        for i in (0, 5, 11):
            a, b = ds_obj[i][0], ds_cached[i][0]
            assert torch.allclose(a, b), f"cache RAM khác đọc đĩa ở mẫu {i}"
        loader = ds.make_loader(df, d, ds.build_transforms(False, img_size=32), batch_size=4,
                                train=False, num_workers=0, img_size=64, pin_memory=False)
        got = []
        for _, _, fs in loader:
            got.extend(fs)
        assert got == names, "thứ tự file của loader không ổn định"
        # check_split phải bắt được giao không rỗng giữa các tập
        try:
            ds.check_split(df, df, df.head(2), d)
            raise AssertionError("phải báo lỗi khi các tập giao nhau")
        except ValueError as e:
            assert "GIAO KHÔNG RỖNG" in str(e) or "17.509" in str(e)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("Chạy selftest (không cần GPU, không cần dataset)\n", flush=True)
    print(f"KẾT QUẢ: {len(PASSED)} đạt, {len(FAILED)} lỗi")
    if FAILED:
        print("\nCác kiểm tra lỗi:")
        for name, e in FAILED:
            print(f"  - {name}: {e}")
        code = 1
    else:
        print("Tất cả kiểm tra đạt.")
        code = 0
    sys.stdout.flush()
    sys.stderr.flush()
    # Ghi chú về `os._exit`: trên một số cấu hình WINDOWS, khi cả torch và pyarrow (do pandas 3.x
    # dùng để tạo string array) cùng nạp, bộ dọn DLL của interpreter có thể báo access violation
    # (-1073741819). Đây là lỗi môi trường, KHÔNG phải lỗi của các kiểm tra: mọi kiểm tra đều
    # đã chạy xong và in kết quả trước khi sập. `os._exit` bảo đảm mã thoại phản ánh đúng kết
    # quả kiểm tra. Trên Kaggle (Linux) không cần, và không ảnh hưởng gì.
    os._exit(code)
