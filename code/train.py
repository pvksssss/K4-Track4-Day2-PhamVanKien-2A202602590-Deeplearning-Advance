"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

Hoàn thiện từ bộ khung `starter/`. Giữ nguyên tên hàm và kiểu dữ liệu vào/ra như docstring gốc.
Dùng MỘT hàm `run(cfg)` cho mọi cấu hình (RUBRIC mục H): đổi thí nghiệm chỉ bằng cách đổi `Config`.

Chạy một thí nghiệm từ dòng lệnh:
    python train.py --set exp_id=B01 backbone=resnet50 seed=0
Chỉ số dùng để chọn checkpoint (macro-F1 val) tính bằng eval.compute_metrics của repo gốc:
    sys.path.insert(0, "<thư mục chứa eval.py>");  from eval import compute_metrics

Ghi chú thiết kế cho môi trường Kaggle (Tesla T4, 2 vCPU):
  - AMP dùng **fp16** (T4 là sm_75, không hỗ trợ bf16) nên luôn cần GradScaler.
  - `cache_images=True` nạp ảnh vào RAM để không bị nghẽn giải mã JPEG trên 2 vCPU.
  - `resume=True` nối tiếp từ `last.pt` nếu phiên bị đóng giữa chừng.
  - Mỗi epoch ghi `last.pt` và `best.pt`; sau khi huấn luyện xong thì xoá `last.pt`
    (chỉ chứa optimizer state, không cần cho suy luận) để tiết kiệm ~200 MB mỗi lần chạy.
  - Logit val/test được lưu ra `.npy` (GUIDE.md mục 4) để phần lớn thí nghiệm suy luận ở
    Bước 3 chạy được không cần GPU.
"""
from __future__ import annotations

import runtime  # Windows pandas compatibility shared with the notebook.

import argparse
import dataclasses
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# Ghi file dự đoán đúng định dạng bằng hàm có sẵn trong eval.py (repo gốc):
#     from eval import save_predictions, compute_metrics
# Log theo epoch (history.csv) và config.json bạn tự ghi bằng pandas/json.

_HERE = Path(__file__).resolve().parent


def _setup_console() -> None:
    """Cho phép in tiếng Việt khi chạy trên Windows (console cp1252).

    Trên Kaggle/Colab (Linux) hàm này không có tác dụng gì.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


_setup_console()


def _import_eval():
    """Import `eval.py` của repo gốc mà không cần sửa file đó.

    Thử lần lượt: thư mục cha (code/ nằm trong repo), cùng thư mục, rồi thư mục làm việc.
    """
    if "eval" in sys.modules:
        import eval as ev
        return ev
    for cand in (_HERE.parent / "eval.py", _HERE / "eval.py", Path.cwd() / "eval.py"):
        if cand.exists():
            sys.path.insert(0, str(cand.parent))
            import eval as ev
            return ev
    raise ImportError("Không tìm thấy eval.py. Hãy đặt code/ cạnh eval.py của repo bài lab.")


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # trivial | color | basic | randaug | basic_vflip
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    # --- tuỳ chọn đã thêm (được phép: "bạn được thêm hàm, tham số, file mới") ---
    grad_clip: float = 1.0            # 0 = không clip
    min_lr_ratio: float = 0.01        # cosine về còn min_lr_ratio * lr ban đầu
    deterministic: bool = False       # True: cudnn.deterministic, chậm hơn nhưng tái lập tốt hơn
    cache_images: bool = True         # nạp ảnh vào RAM (xem dataset.py)
    pin_memory: bool | None = None     # None = tự bật khi có CUDA; tắt nếu máy lỗi pinned memory
    check_split: bool = True          # chạy kiểm tra S1-S4 mỗi lần (cache kết quả vào json)
    resume: bool = False              # nối tiếp từ last.pt nếu có (Kaggle bị ngắt phiên)
    save_val_predictions: bool = True # luôn ghi predictions/<exp>_seed<k>_val.csv (cần cho --final-val)
    curves_dir: str = "curves"
    logit_dir: str = "logits"         # lưu val/test logits ra .npy để Bước 3 không cần GPU
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"             # config.json, history.csv, checkpoint, logit của từng lần chạy
    pred_dir: str = "predictions"     # file dự đoán đúng định dạng eval.py (nộp cùng bài)
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False
    # Isolated pipeline verification only; never a reportable experiment.
    debug_samples_per_class: int | None = None


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def logit_path(cfg: Config, split: str) -> Path:
    """Đường dẫn file logit: <logit_dir>/<exp_id>/seed<k>_<split>.npz"""
    return Path(cfg.logit_dir) / cfg.exp_id / f"seed{cfg.seed}_{split}.npz"


# --------------------------------------------------------------------------- #
# 1. Seed, optimizer, scheduler, EMA
# --------------------------------------------------------------------------- #
def set_seed(seed: int, deterministic: bool = False) -> None:
    """Cố định mọi nguồn ngẫu nhiên.

    TODO đã làm: random, numpy, torch (CPU và CUDA), os.environ PYTHONHASHSEED, seed cho
    worker của DataLoader (xem dataset._worker_init_fn).

    Về mức độ tái lập: seed cố định head, thứ tự batch và augmentation. Với
    `deterministic=False`, `cudnn.benchmark=True` để T4 chọn thuật toán nhanh nhất; thuật toán
    đã chọn có thể khác giữa hai lần chạy nên kết quả KHÔNG giống nhau từng bit, nhưng sai
    số ở mức ~1e-6 và không đổi kết luận. Bật `deterministic=True` (Config) khi cần đối chiếu
    một cấu hình cụ thể. Điều này được ghi lại trong báo cáo.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not deterministic


def build_optimizer(model, cfg: Config):
    """AdamW với 3 nhóm tham số (xem model.param_groups).

    TODO đã làm: groups từ `param_groups`, AdamW với betas (0.9, 0.999) mặc định của AdamW.
    """
    import torch

    from model import param_groups

    groups = param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    return torch.optim.AdamW(groups, betas=(0.9, 0.999))


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine về ~0 (slide trang 55).

    TODO đã làm: cập nhật theo BƯỚC (iteration) bằng `LambdaLR`; mỗi nhóm tham số giữ đúng
    tỉ lệ lr của nó vì LambdaLR nhân với lr gốc của từng nhóm.
    Vẽ đường LR theo bước để thấy đúng hình warmup + cosine: `curves/<exp_id>_lr.png`.
    """
    import torch

    total = max(1, cfg.epochs * steps_per_epoch)
    warmup = max(0, int(round(cfg.warmup_epochs * steps_per_epoch)))
    floor = float(cfg.min_lr_ratio)

    def fn(step: int) -> float:
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup
        p = (step - warmup) / max(1, total - warmup)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


class EMA:
    """Trung bình động trọng số: W_ema <- d * W_ema + (1 - d) * W  (slide trang 56).

    TODO đã làm:
      - __init__(model, decay): sao chép trọng số (bản sao riêng, ở chế độ eval)
      - update(model): sau mỗi bước tối ưu
      - copy_to(model): chép sang model để đánh giá/suy luận bằng trọng số EMA
      - BatchNorm: mọi tensor float trong state_dict (kể cả running_mean/var) đều được
        trung bình như tham số -- đây cũng là cách timm làm; tensor nguyên (ví dụ
        num_batches_tracked) được chép thẳng. Nếu muốn giữ nguyên thống kê BN của model
        đang huấn luyện thì truyền `copy_buffers=True`.
    """

    def __init__(self, model, decay: float, copy_buffers: bool = False):
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay phải nằm trong [0, 1), nhận {decay}")
        import copy

        import torch

        self.decay = float(decay)
        self.copy_buffers = copy_buffers
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self._ema_state = self.module.state_dict()
        self._model_state = model.state_dict()
        for v in self._ema_state.values():
            if torch.is_floating_point(v):
                v.requires_grad_(False)

    @torch.no_grad()
    def update(self, model) -> None:
        import torch

        msd = model.state_dict()
        for k, e in self._ema_state.items():
            m = msd[k].detach()
            if torch.is_floating_point(e) and not self.copy_buffers:
                e.mul_(self.decay).add_(m, alpha=1.0 - self.decay)
            else:
                e.copy_(m)

    @torch.no_grad()
    def copy_to(self, model) -> None:
        model.load_state_dict(self.module.state_dict(), strict=True)

    def state_dict(self) -> dict:
        return self.module.state_dict()


# --------------------------------------------------------------------------- #
# 2. Vòng huấn luyện và đánh giá
# --------------------------------------------------------------------------- #
def _amp_on(cfg: Config, device) -> bool:
    return bool(cfg.amp) and getattr(device, "type", "cpu") == "cuda"


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None, lr_trace: list | None = None) -> dict:
    """Một epoch huấn luyện. Trả về dict, ví dụ {"train_loss": ..., "lr": ...}.

    TODO đã làm:
      - model.train() (nếu init == "frozen": giữ phần backbone ở eval, xem model.set_bn_eval)
      - nếu cfg.mix: mix_batch rồi mixed_loss (losses.py)
      - AMP (autocast fp16 + GradScaler), clip gradient trước optimizer.step(), scheduler.step()
      - nếu có EMA: ema.update(model)
      - `lr_trace` (nếu truyền) ghi (global_step, lr_backbone) để vẽ đường LR theo bước.
    """
    import torch

    model.train()
    if cfg.init == "frozen":
        from model import set_bn_eval
        set_bn_eval(model)

    amp_on = _amp_on(cfg, device)
    loss_sum = 0.0
    top1_sum, n_seen = 0.0, 0
    t0 = time.perf_counter()

    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        targets = None
        if cfg.mix:
            from losses import mix_batch
            images, targets = mix_batch(images, labels, alpha=cfg.mix_alpha, mode=cfg.mix)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_on):
            logits = model(images)
            if targets is not None:
                from losses import mixed_loss
                loss = mixed_loss(criterion, logits, targets)
            else:
                loss = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if cfg.grad_clip and cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"] if p.grad is not None],
                cfg.grad_clip)
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        # AMP overflow skips optimizer.step; schedule/EMA must skip the same update.
        if scaler.get_scale() >= scale_before:
            scheduler.step()
            if ema is not None:
                ema.update(model)

        bs = images.size(0)
        loss_sum += loss.detach().item() * bs
        n_seen += bs
        if targets is None:
            # accuracy trên batch đã trộn không còn nghĩa bình thưng -> không ghi khi mix
            top1_sum += (logits.detach().argmax(1) == labels).sum().item()
        if lr_trace is not None:
            lr_trace.append((scheduler.last_epoch, optimizer.param_groups[0]["lr"]))

    return {
        "train_loss": loss_sum / max(1, n_seen),
        "train_top1": (top1_sum / n_seen) if (cfg.mix is None and n_seen) else None,
        "lr": optimizer.param_groups[0]["lr"],
        "seconds": time.perf_counter() - t0,
    }


def evaluate(model, loader, criterion, device):
    """Chạy model trên một loader ở chế độ eval, KHÔNG tính gradient.

    Trả về (filenames: list[str], y_true: ndarray[N], logits: ndarray[N, 9], loss: float).
    Giữ đúng thứ tự của loader để ghép logit với tên file.

    TODO đã làm: model.eval(), torch.inference_mode(), gom kết quả.
    Đánh giá chạy FP32 (không autocast) để xác suất ổn định, vì xác suất này dùng cho ECE,
    temperature scaling và cả file predictions. Nếu muốn tiết kiệm thời gian thì chấp nhận
    fp16 nhưng phải ghi rõ trong báo cáo.
    """
    import torch

    model.eval()
    filenames: list[str] = []
    ys, logits_all = [], []
    loss_sum, n_seen = 0.0, 0
    with torch.inference_mode():
        for images, labels, names in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss_sum += criterion(logits, labels).item() * images.size(0)
            n_seen += images.size(0)
            logits_all.append(logits.float().cpu().numpy())
            ys.append(labels.cpu().numpy())
            filenames.extend(names)
    return (filenames,
            np.concatenate(ys) if ys else np.zeros(0, dtype=np.int64),
            np.concatenate(logits_all) if logits_all else np.zeros((0, 9), dtype=np.float32),
            loss_sum / max(1, n_seen))


def softmax_np(logits: np.ndarray) -> np.ndarray:
    """softmax ổn định trên numpy (trừ max trước để không tràn số)."""
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def metrics_from_logits(y_true: np.ndarray, logits: np.ndarray) -> dict:
    """Chỉ số theo ĐÚNG định nghĩa của eval.py (dùng eval.compute_metrics)."""
    ev = _import_eval()
    probs = softmax_np(logits)
    return ev.compute_metrics(y_true, probs.argmax(1), probs)


def save_logits(path: str | Path, filenames, logits: np.ndarray, y_true: np.ndarray) -> Path:
    """Lưu logit + tên file + nhãn ra .npz để Bước 3 xử lý TTA/ensemble/T mà không cần GPU."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, filenames=np.asarray(list(filenames), dtype=object),
                        logits=np.asarray(logits, dtype=np.float32),
                        y_true=np.asarray(y_true, dtype=np.int64))
    return path


def load_logits(path: str | Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Đọc lại file .npz của save_logits -> (filenames, logits, y_true)."""
    d = np.load(path, allow_pickle=True)
    return [str(x) for x in d["filenames"]], d["logits"], d["y_true"]


def save_checkpoint(path: Path, model, ema: EMA | None, epoch: int,
                    best_macro_f1: float, best_epoch: int, extra: dict | None = None) -> None:
    """Lưu trọng số (và trọng số EMA nếu có). `extra` dành cho state của optimizer/scheduler."""
    payload = {"model": model.state_dict(), "epoch": epoch,
               "best_macro_f1": best_macro_f1, "best_epoch": best_epoch,
               "pretrained_tag": getattr(model, "pretrained_tag", "unknown")}
    if ema is not None:
        payload["ema"] = ema.state_dict()
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch_save(payload, path)


def torch_save(payload, path):
    import torch

    torch.save(payload, path)


def load_state(cfg: Config, device=None, prefer_ema: bool = False):
    """Dựng lại model của một lần chạy đã có và nạp checkpoint tốt nhất (dùng cho Bước 3).

    Trả về (model, best_epoch, best_macro_f1_val).
    """
    import torch

    from model import build_model

    rdir = run_dir(cfg)
    ckpt_path = rdir / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Không có checkpoint {ckpt_path}; hãy chạy run(Config(...)) trước")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(cfg.backbone, pretrained=False, num_classes=9,
                        drop_rate=cfg.drop_rate, init=cfg.init)
    key = "ema" if (prefer_ema and "ema" in ckpt) else "model"
    model.load_state_dict(ckpt[key])
    model.pretrained_tag = ckpt.get("pretrained_tag", model.pretrained_tag)
    if device is not None:
        model.to(device)
    return model, ckpt.get("best_epoch", -1), ckpt.get("best_macro_f1", float("nan"))


# --------------------------------------------------------------------------- #
# 3. Biểu đồ
# --------------------------------------------------------------------------- #
def plot_curves(history: list[dict], path: str | Path, title: str,
                lr_trace: list | None = None) -> None:
    """Vẽ đường cong training của một thí nghiệm -> curves/<exp_id>_<mota>.png (GUIDE.md mục 6.2).

    TODO đã làm: 3 khung — (1) loss train/val, (2) macro-F1 val (kèm top-1 val),
    (3) LR theo bước để thấy warmup + cosine. Có tiêu đề, nhãn trục, chú thích, lưới.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    ax.plot(ep, [h["train_loss"] for h in history], "o-", label="train loss", color="tab:blue")
    ax.plot(ep, [h["val_loss"] for h in history], "s-", label="val loss", color="tab:orange")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(ep, [h["val_macro_f1"] for h in history], "s-", label="val macro-F1", color="tab:green")
    if any(h.get("val_top1") is not None for h in history):
        ax.plot(ep, [h["val_top1"] for h in history], "^--", label="val top-1", color="tab:red")
    if any(h.get("train_top1") is not None for h in history):
        tr = [(e, t) for e, t in zip(ep, [h.get("train_top1") for h in history]) if t is not None]
        if tr:
            ax.plot([a for a, _ in tr], [b for _, b in tr], "o--", label="train top-1",
                    color="tab:purple", alpha=0.7)
    best = max(history, key=lambda h: h["val_macro_f1"])
    ax.axvline(best["epoch"], ls=":", c="k", lw=1,
               label=f"best epoch = {best['epoch']} (macro-F1 {best['val_macro_f1']:.4f})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("score")
    ax.set_title("Val metric")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    if lr_trace:
        steps = [s for s, _ in lr_trace]
        lrs = [v for _, v in lr_trace]
        ax.plot(steps, lrs, color="tab:purple")
    ax.set_xlabel("step")
    ax.set_ylabel("lr (backbone group)")
    ax.set_title(f"LR schedule (warmup {history[0].get('warmup_steps', '?')} bước)")
    ax.grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 4. run(cfg) - hàm duy nhất cho mọi thí nghiệm
# --------------------------------------------------------------------------- #
def _build_criterion(cfg: Config, train_df):
    """Dựng criterion từ Config. Trọng số lớp chỉ tính từ TRAIN (S2)."""
    from losses import build_criterion

    if cfg.loss == "ls":
        return build_criterion("ls", smoothing=cfg.label_smoothing or 0.1)
    if cfg.loss == "focal":
        return build_criterion("focal", gamma=cfg.focal_gamma)
    if cfg.loss == "ce_weighted":
        counts = np.bincount(train_df["Label"].to_numpy(), minlength=9)
        beta = 0.0 if cfg.class_weight_beta is None else float(cfg.class_weight_beta)
        return build_criterion("ce_weighted", counts=counts, beta=beta)
    return build_criterion(cfg.loss)


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình và lưu mọi thứ cần thiết. Trả về dict kết quả tóm tắt.

    TODO đã làm, theo thứ tự:
      1. set_seed; tạo thư mục run_dir(cfg); ghi config.json (dataclasses.asdict(cfg))
      2. dataset.load_split + dataset.check_split (dừng nếu vi phạm S1-S6)
      3. dựng train/val loader (test loader chỉ tạo khi cfg.save_test_predictions)
      4. model.build_model, criterion (losses.build_criterion), optimizer, scheduler, scaler, EMA
      5. với mỗi epoch: train_one_epoch -> evaluate(val) -> ghi history (loss, macro-F1 val, lr...)
         và lưu checkpoint tốt nhất theo MACRO-F1 VAL (hòa thì lấy epoch sớm hơn)
      6. cuối: nạp checkpoint tốt nhất, lưu val logits và eval.save_predictions(pred_path(cfg, "val"), ...)
      7. NẾU cfg.save_test_predictions (chỉ ở Bước 4): đánh giá test đúng MỘT lần,
         lưu logits và eval.save_predictions(pred_path(cfg, "test"), ...)
      8. ghi history.csv, plot_curves(...), trả về dict tóm tắt
         (best_epoch, macro-F1 val, thời gian train mỗi epoch, số tham số, GMAC)
    Quy tắc: KHÔNG dùng test để chọn checkpoint hay bất kỳ quyết định nào (README.md, S4).
    """
    import torch

    import dataset as ds
    from model import build_model, count_gmacs_with_tool, count_params

    if cfg.debug_samples_per_class is not None:
        if cfg.debug_samples_per_class < 1:
            raise ValueError("debug_samples_per_class must be positive")
        if cfg.save_test_predictions:
            raise ValueError("debug runs cannot evaluate test")

    ev = _import_eval()
    rdir = run_dir(cfg)
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2, ensure_ascii=False),
                                      encoding="utf-8")

    set_seed(cfg.seed, cfg.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== {cfg.exp_id} | seed {cfg.seed} | {cfg.backbone} | init={cfg.init} | "
          f"loss={cfg.loss} | aug={cfg.aug} | mix={cfg.mix} | sampler={cfg.sampler} ===")
    print(f"[run] device={device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else " (KHÔNG có GPU!)"))

    # --- 2. dữ liệu ---
    train_df, val_df, test_df = ds.load_split(cfg.labels_dir, fold=cfg.fold)
    if cfg.check_split:
        cache_json = Path(cfg.out_dir) / f"split_check_fold{cfg.fold}.json"
        if cache_json.exists():
            print(f"[run] dùng kết quả kiểm tra chia đã lưu: {cache_json}")
        else:
            report = ds.check_split(train_df, val_df, test_df, cfg.images_dir)
            report.pop("class_names", None)
            cache_json.parent.mkdir(parents=True, exist_ok=True)
            cache_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    train_tf = ds.build_transforms(train=True, img_size=cfg.img_size, aug=cfg.aug)
    val_tf = ds.build_transforms(train=False, img_size=cfg.img_size)

    if cfg.debug_samples_per_class is not None:
        n = cfg.debug_samples_per_class
        train_df = train_df.groupby("Label", group_keys=False).head(n).reset_index(drop=True)
        val_df = val_df.groupby("Label", group_keys=False).head(n).reset_index(drop=True)
        print(f"[DEBUG ONLY] train={len(train_df)}, val={len(val_df)}; excluded from results.xlsx")

    train_loader = ds.make_loader(train_df, cfg.images_dir, train_tf, cfg.batch_size, train=True,
                                  sampler=cfg.sampler, num_workers=cfg.num_workers,
                                  cache_in_ram=cfg.cache_images, seed=cfg.seed, img_size=cfg.img_size,
                                  pin_memory=cfg.pin_memory)
    val_loader = ds.make_loader(val_df, cfg.images_dir, val_tf, cfg.batch_size, train=False,
                                num_workers=cfg.num_workers, cache_in_ram=cfg.cache_images,
                                seed=cfg.seed, img_size=cfg.img_size, pin_memory=cfg.pin_memory)
    test_loader = None
    if cfg.save_test_predictions:
        # chỉ tạo ở Bước 4; không dùng test_loader cho bất kỳ quyết định nào (S4)
        test_loader = ds.make_loader(test_df, cfg.images_dir, val_tf, cfg.batch_size, train=False,
                                     num_workers=cfg.num_workers, cache_in_ram=cfg.cache_images,
                                     seed=cfg.seed, img_size=cfg.img_size, pin_memory=cfg.pin_memory)

    # --- 3. model và tối ưu ---
    model = build_model(cfg.backbone, pretrained=True, num_classes=ds.NUM_CLASSES,
                        drop_rate=cfg.drop_rate, init=cfg.init).to(device)
    params_m = count_params(model)
    gmacs, gmac_tool = count_gmacs_with_tool(model, cfg.img_size)
    print(f"[run] tag trọng số = {model.pretrained_tag} | params {params_m:.2f}M | "
          f"{gmacs:.2f} GMAC ({gmac_tool})")

    criterion = _build_criterion(cfg, train_df).to(device)
    optimizer = build_optimizer(model, cfg)
    steps_per_epoch = len(train_loader)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)
    scaler = torch.amp.GradScaler("cuda", enabled=_amp_on(cfg, device))
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay else None

    # --- 4. resume ---
    start_epoch, best_f1, best_epoch = 1, -1.0, -1
    last_pt = rdir / "last.pt"
    if cfg.resume and last_pt.exists():
        ck = torch.load(last_pt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        if ema is not None and "ema" in ck:
            ema.module.load_state_dict(ck["ema"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1
        best_f1, best_epoch = ck["best_macro_f1"], ck["best_epoch"]
        print(f"[run] resume từ {last_pt}: bắt đầu lại epoch {start_epoch}, best macro-F1 {best_f1:.4f}")

    history: list[dict] = []
    lr_trace: list = []
    if (rdir / "history.csv").exists() and cfg.resume:
        import pandas as pd
        history = pd.read_csv(rdir / "history.csv").to_dict("records")
        print(f"[run] đã đọc {len(history)} epoch trong history.csv")

    warmup_steps = int(round(cfg.warmup_epochs * steps_per_epoch))
    for epoch in range(start_epoch, cfg.epochs + 1):
        tr = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler,
                             cfg, device, ema=ema, lr_trace=lr_trace)
        names, y_val, logits_val, val_loss = evaluate(model, val_loader, criterion, device)
        m = metrics_from_logits(y_val, logits_val)
        row = {
            "epoch": epoch, "train_loss": tr["train_loss"], "train_top1": tr["train_top1"],
            "val_loss": val_loss, "val_top1": m["top1"], "val_macro_f1": m["macro_f1"],
            "val_balanced_acc": m["balanced_acc"], "val_ece": m["ece"], "val_nll": m["nll"],
            "lr": tr["lr"], "seconds": tr["seconds"], "warmup_steps": warmup_steps,
        }
        history.append(row)
        star = ""
        if m["macro_f1"] > best_f1:          # ">" nghĩa là hòa thì lấy epoch sớm hơn
            best_f1, best_epoch = m["macro_f1"], epoch
            save_checkpoint(rdir / "best.pt", model, ema, epoch, best_f1, best_epoch)
            star = "  <- best"
        print(f"[{cfg.exp_id} s{cfg.seed}] ep {epoch:>2}/{cfg.epochs} "
              f"train_loss {tr['train_loss']:.4f} val_loss {val_loss:.4f} "
              f"val_macroF1 {m['macro_f1']:.4f} val_top1 {m['top1']:.4f} "
              f"ece {m['ece']:.4f} lr {tr['lr']:.2e} {tr['seconds']:.1f}s{star}")

        save_checkpoint(rdir / "last.pt", model, ema, epoch, best_f1, best_epoch,
                        extra={"optimizer": optimizer.state_dict(),
                               "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict()})
        _write_history(rdir / "history.csv", history)

    # --- 5. nạp lại checkpoint tốt nhất ---
    model, best_epoch, best_f1 = load_state(cfg, device=device, prefer_ema=False)
    names, y_val, logits_val, _ = evaluate(model, val_loader, criterion, device)
    save_logits(logit_path(cfg, "val"), names, logits_val, y_val)
    val_m = metrics_from_logits(y_val, logits_val)
    if cfg.save_val_predictions:
        ev.save_predictions(pred_path(cfg, "val"), names, y_val, softmax_np(logits_val))

    summary = {
        "exp_id": cfg.exp_id, "seed": cfg.seed, "backbone": cfg.backbone,
        "debug_run": cfg.debug_samples_per_class is not None,
        "init": cfg.init, "aug": cfg.aug, "img_size": cfg.img_size, "epochs": cfg.epochs,
        "batch_size": cfg.batch_size, "loss": cfg.loss, "mix": cfg.mix, "sampler": cfg.sampler,
        "lr_backbone": cfg.lr_backbone, "lr_head": cfg.lr_head, "weight_decay": cfg.weight_decay,
        "ema_decay": cfg.ema_decay, "pretrained_tag": model.pretrained_tag,
        "params_m": round(params_m, 3), "gmacs": round(gmacs, 3), "gmac_tool": gmac_tool,
        "best_epoch": best_epoch,
        "val_macro_f1": float(val_m["macro_f1"]), "val_top1": float(val_m["top1"]),
        "val_balanced_acc": float(val_m["balanced_acc"]), "val_ece": float(val_m["ece"]),
        "val_recall": [float(x) for x in val_m["recall"]],
        "sec_per_epoch": float(np.mean([h["seconds"] for h in history])) if history else None,
        "total_sec": float(sum(h["seconds"] for h in history)) if history else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
    }

    # --- 6. test: đúng MỘT lần, chỉ khi bật cờ ở Bước 4 ---
    if cfg.save_test_predictions and test_loader is not None:
        names_t, y_test, logits_test, _ = evaluate(model, test_loader, criterion, device)
        save_logits(logit_path(cfg, "test"), names_t, logits_test, y_test)
        ev.save_predictions(pred_path(cfg, "test"), names_t, y_test, softmax_np(logits_test))
        mt = metrics_from_logits(y_test, logits_test)
        summary.update({"test_macro_f1": float(mt["macro_f1"]), "test_top1": float(mt["top1"]),
                        "test_balanced_acc": float(mt["balanced_acc"]), "test_ece": float(mt["ece"]),
                        "test_recall": [float(x) for x in mt["recall"]]})
        print(f"[{cfg.exp_id} s{cfg.seed}] TEST (một lần): top1 {mt['top1']:.4f} "
              f"macroF1 {mt['macro_f1']:.4f} ece {mt['ece']:.4f}")

    _write_history(rdir / "history.csv", history)
    plot_curves(history, Path(cfg.curves_dir) / f"{cfg.exp_id}_seed{cfg.seed}.png",
                title=f"{cfg.exp_id} | {cfg.backbone} (tag={model.pretrained_tag}) | seed={cfg.seed} "
                      f"| aug={cfg.aug} loss={cfg.loss} mix={cfg.mix}",
                lr_trace=lr_trace)
    (rdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
    if last_pt.exists() and not cfg.resume:
        last_pt.unlink()                      # optimizer state không cần cho suy luận
    return summary


def _write_history(path: Path, history: list[dict]) -> None:
    """Ghi history.csv (mỗi epoch một dòng, không kèm lr_trace dài)."""
    import pandas as pd

    pd.DataFrame(history).to_csv(path, index=False)


# --------------------------------------------------------------------------- #
# 7. CLI
# --------------------------------------------------------------------------- #
def _coerce(value: str, field_type: str):
    """Ép kiểu giá trị string theo kiểu của field trong Config."""
    if value.lower() in ("none", "null", ""):
        return None
    t = field_type
    try:
        if "bool" in t:
            return value.lower() in ("1", "true", "yes", "y", "on")
        if t.startswith("int") or t == "int":
            return int(float(value))
        if t.startswith("float") or t == "float":
            return float(value)
        return value
    except ValueError as e:
        raise ValueError(f"không ép được '{value}' sang {t}") from e


def parse_overrides(pairs: list[str]) -> dict:
    """Biến ['seed=1', 'loss=focal', 'ema_decay=none'] thành dict, ép kiểu theo field của Config.

    TODO đã làm: tách key/value (theo dấu `=` đầu tiên), báo lỗi rõ nếu key không có trong
    Config, ép int/float/bool/None theo kiểu field. `none` -> None (dùng cho `sampler`,
    `mix`, `ema_decay`, `class_weight_beta`).
    """
    types = {f.name: str(f.type) for f in dataclasses.fields(Config)}
    out: dict = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"override '{pair}' phải có dạng KEY=VALUE")
        key, value = pair.split("=", 1)
        key, value = key.strip(), value.strip()
        if key not in types:
            raise ValueError(f"'{key}' không phải field của Config. "
                             f"Các field hợp lệ: {', '.join(sorted(types))}")
        out[key] = _coerce(value, types[key])
    return out


def main() -> None:
    """Điểm vào dòng lệnh: `python train.py --set exp_id=B01 backbone=resnet50 seed=0`.

    TODO đã làm: argparse nhận `--set KEY=VALUE ...`, dựng Config qua parse_overrides,
    gọi run(cfg), in kết quả tóm tắt dạng JSON.
    """
    ap = argparse.ArgumentParser(description="Huấn luyện một cấu hình của lab Day 2")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="ghi đè field của Config, ví dụ: --set exp_id=B01 backbone=resnet50 seed=0")
    args = ap.parse_args()

    cfg = Config(**parse_overrides(args.set))
    summary = run(cfg)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
