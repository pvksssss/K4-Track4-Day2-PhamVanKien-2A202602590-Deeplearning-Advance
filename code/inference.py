"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Hoàn thiện từ bộ khung `starter/`. Giữ nguyên tên hàm và kiểu dữ liệu vào/ra như docstring gốc.
Liên hệ slide Day 2: TTA (trang 62-66, 75), ensemble/EMA/soup (trang 67), độ phân giải kiểm tra
(trang 68), temperature scaling (trang 69), gộp BatchNorm (trang 71).

Mọi hàm chạy ở chế độ eval, không gradient. Chọn phương pháp CHỈ dựa trên val; nhiệt độ T khớp
trên VAL rồi áp dụng sang test (README.md, S2 và S4).

    predict_logits(model, loader, device, view=None) -> (filenames, y_true, logits[N, 9])
    aggregate_views(list_of_logits, space)           -> probs[N, 9]
    fit_temperature(val_logits, val_labels)          -> float T
    apply_temperature(logits, T)                     -> probs
    ensemble_probs(list_of_probs)                    -> probs
    fuse_conv_bn(model)                              -> model (BN đã gộp vào conv)

Lựa chọn đáng ghi lại:
  - `predict_logits` chạy FP32. Xác suất đầu ra dùng cho ECE, temperature scaling và file
    predictions nên cần ổn định; độ trễ thì đo riêng trong benchmark.py (có dtype riêng).
  - `aggregate_views` hỗ trợ cả "prob" và "logit"; bài lab cần so sánh cả hai (I03) chứ không
    mặc định chọn một.
  - `fit_temperature` dùng tìm kiếm trên `log T` (lưới thô + golden-section) thay vì LBFGS:
    tất định, không phụ thuộc optimizer, và cho T tối ưu trong khoảng [0.05, 20].
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# 1. Lấy logit từ model
# --------------------------------------------------------------------------- #
def predict_logits(model, loader, device, view=None):
    """Chạy model trên loader và gom logit theo đúng thứ tự file.

    `view` là hàm biến đổi batch ảnh trước khi đưa vào model (ví dụ lật ngang), hoặc None.
    TODO đã làm: model.eval(), torch.inference_mode(), trả về numpy.
    """
    import torch

    model.eval()
    filenames: list[str] = []
    ys, out = [], []
    with torch.inference_mode():
        for images, labels, names in loader:
            images = images.to(device, non_blocking=True)
            if view is not None:
                images = view(images)
            logits = model(images)
            out.append(logits.float().cpu().numpy())
            ys.append(labels.cpu().numpy())
            filenames.extend(names)
    if not out:
        return [], np.zeros(0, dtype=np.int64), np.zeros((0, 9), dtype=np.float32)
    return filenames, np.concatenate(ys), np.concatenate(out)


def view_identity(x):
    return x


def view_hflip(x):
    """Lật ngang batch (N, C, H, W). TODO đã làm: torch.flip trên chiều rộng (slide trang 75)."""
    import torch

    return torch.flip(x, dims=[-1])


def view_vflip(x):
    """Lật dọc. Chỉ dùng để kiểm tra: ảnh cỏ dại có đối xứng dọc hay không."""
    import torch

    return torch.flip(x, dims=[-2])


def views_multicrop(x, crop: int):
    """5 crop (4 góc + giữa) kích thước `crop`, và tuỳ chọn thêm bản lật. Trả về list các batch.

    TODO đã làm: cắt từ ảnh đã tensor (N, C, H, W) với H, W >= crop; trả về [c1..c5].
    Hàm KHÔNG tự thêm bản lật: ghép TTA do `tta_logits` điều khiển nên tách bạch "view nào"
    với "lật hay không" (I01 = chỉ lật).
    """
    import torch
    import torch.nn.functional as F

    n, c, h, w = x.shape
    if h < crop or w < crop:
        raise ValueError(f"ảnh {h}x{w} nhỏ hơn crop={crop}")
    out = []
    for top, left in [(0, 0), (0, w - crop), (h - crop, 0), (h - crop, w - crop)]:
        out.append(x[:, :, top:top + crop, left:left + crop])
    top, left = (h - crop) // 2, (w - crop) // 2
    out.append(x[:, :, top:top + crop, left:left + crop])
    return out


def views_multiscale(x, sizes):
    """Resize batch về từng kích thước trong `sizes`, trả về list các batch. TODO đã làm.

    Lưu ý: model phải chấp nhận ảnh khác kích thước lúc train (CNN có global pooling thì được;
    ViT/Swin cần xử lý riêng vị trí/cửa sổ). Ghi rõ hạn chế gặp:
      - CNN của timm (ResNet, ConvNeXt, EfficientNet, MobileNetV3) nhận mọi kích thước chia hết
        cho 32 (global average pool), nên dò độ phân giải 224/256/288/320 chạy được.
      - DeiT/ViT có `pos_embed` cố định 224x224: truyền ảnh khác kích thước sẽ ra lỗi hình dạng,
        phải nội suy `pos_embed` trước. Hàm này KHÔNG tự xử lý; nếu backbone không chịu được
        kích thước lạ thì hãy bỏ nhánh đó khỏi thí nghiệm I04 và ghi rõ trong báo cáo.
      - Swin yêu cầu kích thước cố định theo lưới cửa sổ; xem `dynamic_img_size` của timm.
    """
    import torch.nn.functional as F

    return [F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False) for s in sizes]


# --------------------------------------------------------------------------- #
# 2. Gộp nhiều lượt chạy
# --------------------------------------------------------------------------- #
def _softmax_np(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def aggregate_views(logits_per_view, space: str = "prob"):
    """Gộp K lượt chạy của TTA thành một dự đoán (slide trang 62).

      - space="prob":  trung bình softmax của từng view
      - space="logit": trung bình logit rồi softmax
    TODO đã làm: trả về xác suất (N, 9) đã chuẩn hoá.

    Cả hai cách đều cho xác suất cộng lại bằng 1 theo từng dòng (đây là điều eval.py kiểm tra).
    """
    if space not in ("prob", "logit"):
        raise ValueError(f"space={space!r} không hợp lệ, chọn 'prob' hoặc 'logit'")
    views = [np.asarray(v, dtype=np.float64) for v in logits_per_view]
    if not views:
        raise ValueError("cần ít nhất một view")
    shapes = {v.shape for v in views}
    if len(shapes) != 1:
        raise ValueError(f"các view phải cùng dạng, nhận {shapes}")
    if space == "prob":
        probs = np.mean([_softmax_np(v) for v in views], axis=0)
        return probs / probs.sum(axis=1, keepdims=True)
    return _softmax_np(np.mean(views, axis=0))


def ensemble_probs(list_of_probs):
    """Trung bình xác suất của nhiều mô hình (khác backbone hoặc khác seed). TODO đã làm.

    Chi phí suy luận = số mô hình. Chỉ ghép các mô hình trên CÙNG tập ảnh và cùng thứ tự file.
    """
    probs = [np.asarray(p, dtype=np.float64) for p in list_of_probs]
    if not probs:
        raise ValueError("cần ít nhất một mô hình")
    shapes = {p.shape for p in probs}
    if len(shapes) != 1:
        raise ValueError(f"các mô hình phải cùng dạng, nhận {shapes}")
    avg = np.mean(probs, axis=0)
    return avg / avg.sum(axis=1, keepdims=True)


def tta_logits(model, loader, device, view_list=None, space: str = "prob"):
    """Chạy nhiều view rồi gộp. `view_list` là list hàm biến đổi batch (mặc định: chỉ 1 view).

    Trả về (filenames, y_true, probs). Với mỗi view, thứ tự tên file phải khớp; hàm này kiểm tra.
    """
    if view_list is None:
        view_list = [view_identity]
    per_view, ref_names, y_true = [], None, None
    for v in view_list:
        names, y, logits = predict_logits(model, loader, device, view=v)
        if ref_names is None:
            ref_names, y_true = names, y
        elif names != ref_names:
            raise ValueError("các view phải cho cùng thứ tự tên file")
        per_view.append(logits)
    return ref_names, y_true, aggregate_views(per_view, space=space)


# --------------------------------------------------------------------------- #
# 3. Hiệu chuẩn
# --------------------------------------------------------------------------- #
def fit_temperature(val_logits, val_labels) -> float:
    """Tìm nhiệt độ T > 0 cực tiểu NLL trên VAL: p = softmax(logit / T)  (slide trang 69).

    TODO đã làm: tìm lưới thô trên `log T` (61 điểm trong [0.05, 20]) rồi TINH bằng
    golden-section search trên `log T` quanh điểm tốt nhất. Cách này không cần optimizer,
    cho kết quả tất định (chạy lại ra cùng T) và tránh phụ thuộc hành vi của LBFGS.
    Accuracy không đổi vì thứ tự lớp không đổi. KHÔNG khớp T trên test.
    """
    import torch
    import torch.nn.functional as F

    z = torch.as_tensor(np.asarray(val_logits), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(val_labels), dtype=torch.int64)
    if z.shape[0] != y.shape[0]:
        raise ValueError(f"số mẫu lệch: {z.shape[0]} logit vs {y.shape[0]} nhãn")
    if z.shape[0] == 0:
        raise ValueError("không có mẫu nào để khớp T")

    def nll(log_t: float) -> float:
        return float(F.cross_entropy(z / float(np.exp(log_t)), y))

    lo, hi = float(np.log(0.05)), float(np.log(20.0))
    grid = np.linspace(lo, hi, 61)
    vals = [nll(g) for g in grid]
    i = int(np.argmin(vals))
    a = float(grid[max(0, i - 1)])
    b = float(grid[min(len(grid) - 1, i + 1)])

    phi = (np.sqrt(5.0) - 1.0) / 2.0
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = nll(c), nll(d)
    for _ in range(60):                       # hội tụ trên log T -> ổn định về cả hai phía
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = nll(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = nll(d)
    return float(np.exp((a + b) / 2.0))


def apply_temperature(logits, T: float):
    """Trả về softmax(logits / T). TODO đã làm."""
    if T <= 0:
        raise ValueError(f"T phải > 0, nhận {T}")
    return _softmax_np(np.asarray(logits, dtype=np.float64) / float(T))


def ece_of(probs, y_true, bins: int = 15) -> float:
    """ECE dùng đúng định nghĩa của eval.py (tránh tự viết lại khác đi một chút)."""
    import sys
    from pathlib import Path

    here = Path(__file__).resolve().parent
    for cand in (here.parent / "eval.py", here / "eval.py", Path.cwd() / "eval.py"):
        if cand.exists() and str(cand.parent) not in sys.path:
            sys.path.insert(0, str(cand.parent))
            break
    import eval as ev

    return ev.ece_score(np.asarray(probs), np.asarray(y_true), bins)


# --------------------------------------------------------------------------- #
# 4. Gộp BatchNorm
# --------------------------------------------------------------------------- #
def fuse_conv_bn(model):
    """Gộp BatchNorm vào tích chập liền trước, chính xác lúc suy luận (slide trang 71, 75):

        w' = gamma * w / sqrt(var + eps)        b' = beta + gamma * (b - mean) / sqrt(var + eps)

    TODO đã làm:
      - model.eval() trước; với mỗi cặp (Conv2d, BatchNorm2d) LIỀN KỀ trong cùng một module cha:
        tạo conv mới (có bias) và thay BN bằng Identity
      - in ra sai số lớn nhất giữa đầu ra trước/sau gộp (xem `fuse_conv_bn_checked`)

    Quét `children()` của từng module nên chỉ gộp đúng cặp liền kề, không gộp nhầm ở nhánh
    residual (BasicBlock của ResNet không có conv+bn liền kề qua nhánh cộng).
    Với kiến trúc không có BN (ViT, Swin, ConvNeXt dùng LayerNorm) hàm không đổi gì;
    `n_fused` trả về 0 là tín hiệu "không áp dụng".
    """
    import torch.nn as nn

    model.eval()
    n_fused = 0
    for parent in model.modules():
        names = list(dict(parent.named_children()).keys())
        for i in range(len(names) - 1):
            conv, bn = getattr(parent, names[i]), getattr(parent, names[i + 1])
            if not (isinstance(conv, nn.Conv2d) and isinstance(bn, nn.modules.batchnorm._BatchNorm)):
                continue
            if not bn.track_running_stats:      # không có running stats thì không gộp được
                continue
            fused = fuse_conv_bn_pair(conv, bn)
            setattr(parent, names[i], fused)
            setattr(parent, names[i + 1], nn.Identity())
            n_fused += 1
    print(f"[inference] đã gộp {n_fused} cặp Conv+BN")
    return model


def fuse_conv_bn_pair(conv, bn):
    """Tạo Conv2d mới đã hấp thụ BN (theo đúng công thức trên)."""
    import torch
    import torch.nn as nn

    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                      stride=conv.stride, padding=conv.padding, dilation=conv.dilation,
                      groups=conv.groups, bias=True,
                      padding_mode=conv.padding_mode, device=conv.weight.device,
                      dtype=conv.weight.dtype)
    std = (bn.running_var + bn.eps).sqrt()
    t = (bn.weight / std).view(-1, 1, 1, 1)
    with torch.no_grad():
        fused.weight.copy_(conv.weight * t)
        b = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
        fused.bias.copy_(bn.bias + (b - bn.running_mean) * bn.weight / std)
    return fused


def fuse_conv_bn_checked(model, img_size: int = 224, device=None):
    """Gộp BN và trả về (model_đã_gộp, sai_số_lớn_nhất) so với model gốc.

    Kiểm tra bắt buộc của rubric H: đầu ra trước/sau gộp phải lệch nhau cỡ 1e-5 trở xuống.
    Kiểm tra trên đầu vào ngẫu nhiên đúng kích thước lúc đánh giá, không dùng dữ liệu test.
    """
    import copy

    import torch

    from inference import fuse_conv_bn

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ref = copy.deepcopy(model).eval().to(device)
    fused = fuse_conv_bn(copy.deepcopy(model).eval().to(device))
    x = torch.randn(2, 3, img_size, img_size, device=device)
    a, b = ref(x).float(), fused(x).float()
    err = float((a - b).abs().max().item())
    rel = err / max(1e-12, float(a.abs().max().item()))
    print(f"[inference] kiểm tra gộp BN: sai số tuyệt đối lớn nhất = {err:.3e} "
          f"(tương đối {rel:.3e})")
    return fused, err
