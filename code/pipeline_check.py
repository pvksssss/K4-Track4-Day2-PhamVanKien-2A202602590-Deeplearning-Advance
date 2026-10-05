"""pipeline_check.py - kiểm tra pipeline trước khi chạy thí nghiệm thật (GUIDE.md mục 1.3).

Chạy:  python pipeline_check.py --images data/images --labels data/labels

Làm đúng checklist gỡ lỗi trên slide (trang 59):
  1. Cố định seed (random, numpy, torch, DataLoader worker)
  2. Mất lạt ban đầu: loss CE của head mới phải xấp xỉ -ln(1/9) = 2.197
  3. Quá khớp một batch nhỏ tới loss gần 0 (nếu không được thì lỗi ở code/model)
  4. Vẽ ảnh sau augmentation (đã giải chuẩn hoá) kèm nhãn để chắc ảnh và nhãn khớp
  5. Kiểm tra model.train()/model.eval() dùng đúng lúc (BatchNorm, dropout)
  6. Vẽ đường LR theo bước: phải thấy đúng hình warmup rồi cosine

Dừng ngay (raise) nếu một trong các kiểm tra định lượng (2, 3) sai.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from dataset import (CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES, build_transforms,
                     load_split, make_loader)
from model import bn_modules, build_model, set_bn_eval
from train import Config, build_optimizer, build_scheduler, set_seed

EXPECTED_INITIAL_LOSS = float(np.log(NUM_CLASSES))   # 2.1972


def show_batch(images, labels, filenames, path, title):
    """Vẽ một lô ảnh SAU khi đã giải chuẩn hoá (đảo lại mean/std) kèm tên lớp và tên file."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    n = len(images)
    cols = min(8, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.1 * cols, 2.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= n:
            continue
        img = (images[i].unsqueeze(0) * std + mean).clamp(0, 1)[0]
        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.set_title(f"{CLASS_NAMES[labels[i]]}\n{filenames[i][:26]}", fontsize=7)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[pipeline] đã lưu {path}")


def main() -> int:
    for _s in (sys.stdout, sys.stderr):
        if hasattr(_s, "reconfigure"):
            try:
                _s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/images")
    ap.add_argument("--labels", default="data/labels")
    ap.add_argument("--out", default="curves/_pipeline")
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--steps", type=int, default=60, help="số bước để overfit một batch nhỏ")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="không tải trọng số ImageNet (chạy được khi máy offline hoặc hết dung lượng). "
                         "Kiểm tra loss ban đầu vẫn có ý nghĩa: đầu 9 lớp ngẫu nhiên luôn cho loss ~ln(9).")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                    help="'auto' dùng CUDA nếu có; chọn 'cpu' để kiểm tra logic khi GPU lỗi.")
    args = ap.parse_args()

    pretrained = not args.no_pretrained
    init = "finetune" if pretrained else "scratch"
    out = Path(args.out)
    set_seed(0)

    # ---------------- 4. xem ảnh sau augmentation ----------------
    train_df, val_df, _ = load_split(args.labels, fold=0)
    for aug in ("trivial", "basic", "color"):
        tf = build_transforms(train=True, img_size=224, aug=aug)
        loader = make_loader(train_df.head(16), args.images, tf, batch_size=16, train=False,
                             num_workers=0, cache_in_ram=False, img_size=256)
        images, labels, names = next(iter(loader))
        show_batch(images, labels, names, out / f"aug_{aug}.png",
                   f"Ảnh sau augmentation '{aug}' (đã đảo chuẩn hoá, kèm nhãn và tên file)")
    tf_eval = build_transforms(train=False, img_size=224)
    loader = make_loader(val_df.head(16), args.images, tf_eval, batch_size=16, train=False,
                         num_workers=0, img_size=256)
    images, labels, names = next(iter(loader))
    show_batch(images, labels, names, out / "aug_eval.png", "Tiền xử lý val/test (không ngẫu nhiên)")

    # ---------------- 2. loss ban đầu xấp xỉ ln(9) ----------------
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"\n[pipeline] device={device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else " (không dùng GPU)"))
    model = build_model(args.backbone, pretrained=pretrained, num_classes=NUM_CLASSES, init=init)
    model = model.to(device)
    print(f"[pipeline] khởi tạo = {init}, tag trọng số = {model.pretrained_tag}")

    loader = make_loader(train_df, args.images, tf_eval, batch_size=64, train=False,
                         num_workers=0, img_size=256)
    model.eval()
    ce = torch.nn.CrossEntropyLoss()
    losses = []
    with torch.inference_mode():
        for i, (images, labels, _) in enumerate(loader):
            losses.append(float(ce(model(images.to(device)), labels.to(device))))
            if i >= 5:
                break
    mean_loss = float(np.mean(losses))
    print(f"[pipeline] loss ban đầu trên {len(losses)} batch đầu: {mean_loss:.4f} "
          f"(kỳ vọng ~{EXPECTED_INITIAL_LOSS:.4f})")
    assert abs(mean_loss - EXPECTED_INITIAL_LOSS) < 0.25, (
        f"loss ban đầu {mean_loss:.4f} lệch quá xa ln(9)={EXPECTED_INITIAL_LOSS:.4f}; "
        "kiểm tra lại cách dựng head hoặc chuẩn hoá đầu vào")

    # ---------------- 3. overfit một batch nhỏ ----------------
    set_seed(0)
    small = make_loader(train_df.head(16), args.images, tf_eval, batch_size=16, train=True,
                        num_workers=0, img_size=256)
    images, labels, _ = next(iter(small))
    images, labels = images.to(device), labels.to(device)

    model = build_model(args.backbone, pretrained=pretrained, num_classes=NUM_CLASSES, init=init)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    first, hist = None, []
    for _ in range(args.steps):
        opt.zero_grad(set_to_none=True)
        loss = ce(model(images), labels)
        loss.backward()
        opt.step()
        v = float(loss.detach())
        hist.append(v)
        if first is None:
            first = v
    print(f"[pipeline] overfit 16 ảnh: loss {first:.4f} -> {hist[-1]:.6f} sau {args.steps} bước")
    assert hist[-1] < 0.05, (
        f"không overfit được một batch nhỏ (loss cuối {hist[-1]:.4f}); "
        "theo slide trang 59, lỗi nằm ở code hoặc model, chưa nên chạy thí nghiệm thật")

    # ---------------- 5. train/eval mode ----------------
    model.train()
    assert all(m.training for m in bn_modules(model)), "model.train() phải đưa BatchNorm sang train"
    model.eval()
    assert not any(m.training for m in bn_modules(model)), "model.eval() phải đưa BatchNorm sang eval"

    frozen = build_model(args.backbone, pretrained=pretrained, num_classes=NUM_CLASSES,
                         init="frozen" if pretrained else "scratch")
    frozen.train()
    set_bn_eval(frozen)
    assert not any(m.training for m in bn_modules(frozen)), \
        "backbone đóng băng: BatchNorm phải ở eval dù đã gọi model.train()"
    print("[pipeline] train()/eval() và đóng băng BatchNorm: đúng")

    # ---------------- 6. đường LR warmup + cosine ----------------
    cfg = Config(epochs=12, warmup_epochs=1.0, batch_size=64)
    steps_per_epoch = len(make_loader(train_df, args.images, tf_eval, 64, train=True,
                                      num_workers=0, img_size=256))
    opt = build_optimizer(build_model(args.backbone, pretrained=False, init=init), cfg)
    sched = build_scheduler(opt, cfg, steps_per_epoch)
    lrs = []
    for _ in range(cfg.epochs * steps_per_epoch):
        lrs.append(opt.param_groups[0]["lr"])
        sched.step()
    warm = int(round(cfg.warmup_epochs * steps_per_epoch))
    assert lrs[0] < lrs[warm - 1] <= cfg.lr_backbone + 1e-12, "LR phải tăng tuyến tính ở warmup"
    assert lrs[-1] < 0.05 * cfg.lr_backbone, f"LR cuối {lrs[-1]:.2e} chưa về gần 0"
    print(f"[pipeline] LR: bước 0 {lrs[0]:.2e} -> đỉnh {max(lrs):.2e} -> cuối {lrs[-1]:.2e} "
          f"(warmup {warm} bước, {steps_per_epoch} bước/epoch)")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(out).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(lrs, color="tab:purple")
    ax.axvline(warm, ls=":", c="k", label=f"hết warmup ({warm} bước)")
    ax.set_xlabel("step")
    ax.set_ylabel("lr (nhóm backbone)")
    ax.set_title(f"LR schedule kiểm tra: warmup tuyến tính rồi cosine ({args.backbone})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "lr_schedule_check.png", dpi=140)
    plt.close(fig)

    print("\n[pipeline] TẤT CẢ KIỂM TRA ĐẠT. Có thể chạy Bước 1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())