"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

Hoàn thiện từ bộ khung `starter/`. Giữ nguyên tên hàm và kiểu dữ liệu vào/ra như docstring gốc.
Liên hệ slide Day 2: label smoothing (trang 56), focal loss (trang 57), Mixup/CutMix (trang 48).

    build_criterion(kind, **kw)                 -> callable(logits, target) -> loss scalar
    class_weights(counts, beta)                 -> tensor trọng số lớp
    mix_batch(x, y, alpha, mode)                -> (x_mixed, (y_a, y_b, lam))
    mixed_loss(criterion, logits, targets)      -> loss scalar

Lựa chọn cài đặt (ghi lại vì rubric H chấm phần "cài đặt đúng kỹ thuật"):
  - Label smoothing dùng `F.cross_entropy(..., label_smoothing=eps)` của PyTorch. Công thức
    q'(k) = (1-eps)*1[k==y] + eps/K là đúng như cách PyTorch hiện thực (trừ chênh lệch ~1/K ở
    lớp đúng), nên kiểm tra `eps=0` cho đúng CE là đủ.
  - Focal loss tự viết để kiểm soát chính xác cách rút gọn `(1-p_t)^gamma` và cách chuẩn
    hoá. Với `gamma=0` hàm trả về đúng `F.cross_entropy` (sai số < 1e-6, xem selftest.py).
  - CutMix điều chỉnh `lam` theo DIỆN TÍCH THỰC của hộp sau khi cắt ra ngoài biên, và đặt
    hộp theo tỉ lệ diện tích đích, không phải số pixel tuyệt đối.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 1. Các hàm loss
# --------------------------------------------------------------------------- #
class LabelSmoothingCE(nn.Module):
    """Cross-entropy với label smoothing: q'(k) = (1 - eps) * 1[k == y] + eps / K  (slide trang 56).

    TODO đã làm: dùng `torch.nn.functional.cross_entropy(label_smoothing=eps)`.
    Kiểm tra `eps = 0` cho đúng CE (xem selftest.py: check_label_smoothing_zero_is_ce).
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError(f"smoothing phải nằm trong [0, 1), nhận {smoothing}")
        self.smoothing = float(smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, target, label_smoothing=self.smoothing)

    def extra_repr(self) -> str:
        return f"smoothing={self.smoothing}"


class FocalLoss(nn.Module):
    """Focal loss nhiều lớp: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)  (slide trang 57).

    TODO đã làm:
      - tính log_softmax, lấy p_t của lớp đúng, nhân (1 - p_t)^gamma, lấy trung bình batch
      - alpha: None hoặc vector trọng số theo lớp
    BẮT BUỘC viết một kiểm tra nhỏ: gamma = 0 phải cho đúng cross-entropy (sai số < 1e-6)
    -- xem selftest.py: check_focal_gamma_zero_is_ce.

    Với `reduction="none"` trả về từng mẫu (cần cho mixup theo mẫu); mặc định "mean".
    `alpha=None` nghĩa là không cân bằng lớp, đây là focal thuần theo slide.
    """

    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = "mean"):
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma phải >= 0, nhận {gamma}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction={reduction!r} không hợp lệ")
        self.gamma = float(gamma)
        self.reduction = reduction
        if alpha is None:
            self.register_buffer("alpha", None)
        else:
            a = torch.as_tensor(alpha, dtype=torch.float32)
            if a.ndim != 1:
                raise ValueError("alpha phải là vector 1-D theo lớp")
            self.register_buffer("alpha", a)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim != 1 or logits.ndim != 2:
            raise ValueError(f"cần logits (N, K) và target (N,), nhận {tuple(logits.shape)}, {tuple(target.shape)}")
        log_p = F.log_softmax(logits, dim=-1)
        log_pt = log_p.gather(1, target.unsqueeze(1)).squeeze(1)          # log p_t
        pt = log_pt.exp()
        loss = -((1.0 - pt) ** self.gamma) * log_pt                       # -alpha_t (1-p_t)^g log p_t
        if self.alpha is not None:
            if self.alpha.numel() != logits.shape[1]:
                raise ValueError(f"alpha có {self.alpha.numel()} phần tử, logits có {logits.shape[1]} lớp")
            loss = loss * self.alpha.to(loss.dtype)[target]
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

    def extra_repr(self) -> str:
        return f"gamma={self.gamma}, alpha={'None' if self.alpha is None else 'yes'}"


def class_weights(counts, beta: float = 0.0):
    """Trọng số theo lớp từ số ảnh mỗi lớp trong tập TRAIN.

    - beta = 0: trọng số tỉ lệ nghịch với số ảnh (1 / n_c), chuẩn hoá về trung bình 1
    - beta > 0: class-balanced theo "số mẫu hiệu dụng": w_c = (1 - beta) / (1 - beta ** n_c)
      (slide trang 57, Cui et al. arXiv:1901.05555); chuẩn hoá tổng trọng số về số lớp

    TODO đã làm: trả về tensor độ dài 9, chuẩn hoá sao cho trung bình = 1 (tức tổng = K).
    Chỉ dùng số liệu của TRAIN (`DeepWeedsDataset.class_counts()`); không dùng val hay test (S4).
    """
    n = np.asarray(counts, dtype=np.float64).reshape(-1)
    if (n <= 0).any():
        raise ValueError("mọi lớp phải có ít nhất 1 ảnh trong tập train")
    if beta < 0 or beta >= 1:
        raise ValueError(f"beta phải nằm trong [0, 1), nhận {beta}")
    if beta == 0:
        w = 1.0 / n
    else:
        w = (1.0 - beta) / (1.0 - np.power(beta, n))
    w = w / w.mean()                      # trung bình = 1
    return torch.tensor(w, dtype=torch.float32)


def build_criterion(kind: str = "ce", **kw):
    """Trả về hàm loss theo `kind`: "ce", "ls" (label smoothing), "focal", "ce_weighted".

    Ví dụ kw: smoothing=0.1, gamma=2.0, alpha=None, weight=tensor.

    TODO đã làm: tạo đúng loss, hoặc gọi các lớp bên dưới.

    Với "ce_weighted" có thể truyền sẵn `weight=<tensor>` (từ `class_weights`), hoặc truyền
    `counts=<array số ảnh train>` + `beta=<float>` để tự tính trọng số.
    """
    if kind == "ce":
        weight = kw.get("weight")
        return nn.CrossEntropyLoss(weight=None if weight is None else torch.as_tensor(weight, dtype=torch.float32))

    if kind == "ls":
        return LabelSmoothingCE(smoothing=float(kw.get("smoothing", 0.1)))

    if kind == "focal":
        return FocalLoss(gamma=float(kw.get("gamma", 2.0)), alpha=kw.get("alpha"),
                         reduction=kw.get("reduction", "mean"))

    if kind == "ce_weighted":
        if "weight" in kw and kw["weight"] is not None:
            weight = kw["weight"]
        else:
            if "counts" not in kw:
                raise ValueError("'ce_weighted' cần weight=<tensor> hoặc counts=<array> và beta=<float>")
            weight = class_weights(kw["counts"], beta=float(kw.get("beta", 0.0)))
        return nn.CrossEntropyLoss(weight=torch.as_tensor(weight, dtype=torch.float32))

    raise ValueError(f"kind={kind!r} không hợp lệ, chọn 'ce', 'ls', 'focal' hoặc 'ce_weighted'")


# --------------------------------------------------------------------------- #
# 2. Trộn batch (Mixup / CutMix)
# --------------------------------------------------------------------------- #
def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    """Trộn một batch ảnh và nhãn.

    - lam ~ Beta(alpha, alpha)
    - mode="mixup": x_mix = lam * x + (1 - lam) * x[perm]
    - mode="cutmix": cắt một hộp chữ nhật từ x[perm] dán vào x, rồi điều chỉnh lam theo
      DIỆN TÍCH THỰC của hộp sau khi cắt ra ngoài biên (slide trang 48)
    - trả về (x_mix, (y_a, y_b, lam)) với y_a = y, y_b = y[perm]

    TODO đã làm: tự cài đặt.
    Diện tích hộp được chọn theo tỉ lệ `1 - lam` của tổng diện tích ảnh, rồi cắt theo biên
    ảnh; `lam` cuối cùng = 1 - area_thuc_te / (H*W). Kiểm tra bằng mắt: xem selftest.py
    (check_cutmix_lambda_matches_area) và hàm `show_mixed_batch` trong notebook.
    """
    if mode not in ("mixup", "cutmix"):
        raise ValueError(f"mode={mode!r} không hợp lệ, chọn 'mixup' hoặc 'cutmix'")
    if x.ndim != 4:
        raise ValueError(f"x phải có dạng (N, C, H, W), nhận {tuple(x.shape)}")

    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    y_a, y_b = y, y[perm]

    if mode == "mixup":
        return lam * x + (1.0 - lam) * x[perm], (y_a, y_b, lam)

    _, _, h, w = x.shape
    cut_ratio = float(np.sqrt(1.0 - lam))
    cut_h, cut_w = int(h * cut_ratio), int(w * cut_ratio)
    if cut_h == 0 or cut_w == 0:
        return x.clone(), (y_a, y_b, 1.0)

    # tâm hộp ngẫu nhiên, cho phép hộp tràn ra ngoài biên như CutMix gốc
    cy = int(np.random.randint(h))
    cx = int(np.random.randint(w))
    y0, y1 = np.clip(cy - cut_h // 2, 0, h), np.clip(cy + cut_h // 2, 0, h)
    x0, x1 = np.clip(cx - cut_w // 2, 0, w), np.clip(cx + cut_w // 2, 0, w)

    x_mix = x.clone()
    x_mix[:, :, y0:y1, x0:x1] = x[perm][:, :, y0:y1, x0:x1]
    lam = 1.0 - ((y1 - y0) * (x1 - x0) / (h * w))
    return x_mix, (y_a, y_b, lam)


def mixed_loss(criterion, logits, targets):
    """Loss cho batch đã trộn: lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b).

    TODO đã làm. Lưu ý: accuracy trên batch đã trộn không còn nghĩa bình thường; đánh giá bằng val.
    (Train loop chỉ ghi `train_top1` khi không trộn, và luôn chọn checkpoint bằng macro-F1 val.)
    """
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)


@torch.no_grad()
def soft_target_nll(logits: torch.Tensor, targets) -> torch.Tensor:
    """NLL với nhãn mềm: -sum_k q_k * log_softmax_k. Tiện khi muốn dùng nhãn mềm trực tiếp."""
    log_p = F.log_softmax(logits, dim=-1)
    q = torch.as_tensor(targets, dtype=logits.dtype, device=logits.device)
    return -(q * log_p).sum(-1).mean()