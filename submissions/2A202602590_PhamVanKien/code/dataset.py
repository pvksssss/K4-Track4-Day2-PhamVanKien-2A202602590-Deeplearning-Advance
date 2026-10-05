"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Hoàn thiện từ bộ khung `starter/`. Giữ nguyên tên hàm và kiểu dữ liệu vào/ra như docstring gốc.

Quy tắc chia dữ liệu bắt buộc (S1-S6) nằm ở README.md, mục 2.1. Không sửa, không lọc, không chia lại.

    load_split(labels_dir, fold=0)                        -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir)   -> dict  (số liệu để ghi báo cáo)
    build_transforms(train, img_size, aug)                -> torchvision transform
    DeepWeedsDataset[i]                                   -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)

Ba lựa chọn thiết kế ghi rõ ở đây vì ảnh hưởng tới kết luận của bài:
  1. Tiền xử lý val/test: `Resize(img_size)` + `CenterCrop(img_size)`. Ảnh DeepWeeds đã là
     hình vuông 256x256, nên với ảnh vuông CenterCrop là thao tác thừa; đường này tương đương
     resize toàn bộ ảnh về img_size. Không dùng augmentation ngẫu nhiên khi đánh giá (S2).
  2. `RandomResizedCrop` dùng `scale=(0.7, 1.0)` thay vì mặc định (0.08, 1.0): đây là bài
     phân loại 9 lớp ở mức loài, nhiều lớp chỉ ~1.100 ảnh; crop quá aggressive làm mất
     chi tiết đặc trưng của lá/cụm hoa và thường hại hơn giúp. Giá trị này là biến của
     trục augmentation (trục B, GUIDE.md mục 3).
  3. `cache_in_ram`: Kaggle T4 chỉ có 2 vCPU, giải mã JPEG thường là nút thắt. Toàn bộ 17.509
     ảnh uint8 256x256x3 chiếm ~3,2 GiB, nạp vào RAM một lần để DataLoader không phải
     giải mã lại mỗi epoch. Trên Linux, tiến trình con được `fork` nên mảng được chia sẻ
     bản sao-on-write, không nhân đôi bộ nhớ.
"""
from __future__ import annotations

import runtime  # Windows pandas compatibility, before constructing DataFrames.

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # đổi nếu trọng số timm bạn dùng yêu cầu mean/std khác
IMAGENET_STD = (0.229, 0.224, 0.225)

# --- hằng số kiểm tra chia dữ liệu (README.md mục 2.1) ---
EXPECTED_TOTAL = 17509           # hợp ba tập phải bằng đúng số này
NOMINAL_SHARE = {"train": 0.60, "val": 0.20, "test": 0.20}
SHARE_TOL = 0.01                 # lệch quá 1 điểm phần trăm thì cảnh báo (không raise)
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# --- các mức augmentation của trục B ---
# trivial     : không augmentation ngẫu nhiên (chỉ resize) -> "không có augmentation"
# basic       : RandomResizedCrop + lật ngang
# color       : basic + ColorJitter
# randaug     : basic + RandAugment
# basic_vflip : basic + lật dọc -> dùng để trả lời câu hỏi "lật dọc có hợp lệ với ảnh cỏ dại không?"
AUG_LEVELS = ("trivial", "basic", "color", "randaug", "basic_vflip")


# --------------------------------------------------------------------------- #
# 1. Đọc split
# --------------------------------------------------------------------------- #
def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1).

    Mỗi file có cột `Filename, Label, Species`. Trả về ba DataFrame.
    KHÔNG sửa, lọc hay chia lại dữ liệu: hàm này chỉ đọc và ép kiểu `Label` về int.

    TODO đã làm:
      - đọc ba file CSV bằng pandas
      - trả về (train_df, val_df, test_df)
    """
    labels_dir = Path(labels_dir)
    frames = []
    for split in ("train", "val", "test"):
        path = labels_dir / f"{split}_subset{fold}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy {path}. Phải dùng fold chia sẵn của tác giả (S1); "
                f"tải từ github.com/AlexOlsen/DeepWeeds, thư mục labels/."
            )
        df = pd.read_csv(path)
        # File chia sẵn của tác giả có 2 cột (Filename, Label); labels.csv có thêm Species.
        # Vì vậy chỉ bắt buộc Filename và Label.
        missing = {"Filename", "Label"} - set(df.columns)
        if missing:
            raise ValueError(f"{path}: thiếu cột {sorted(missing)}")
        df = df.copy()
        df["Label"] = df["Label"].astype(np.int64)
        if df["Label"].min() < 0 or df["Label"].max() >= NUM_CLASSES:
            raise ValueError(f"{path}: Label phải nằm trong 0..{NUM_CLASSES - 1}")
        if df["Filename"].duplicated().any():
            raise ValueError(f"{path}: có tên file trùng nhau")
        frames.append(df)
    return tuple(frames)


# --------------------------------------------------------------------------- #
# 2. Kiểm tra chia dữ liệu
# --------------------------------------------------------------------------- #
def resolve_images_dir(images_dir: str | Path) -> Path:
    """Tìm thư mục thực sự chứa file ảnh.

    `images.zip` giải nén ra `data/images/*.jpg` (phẳng). Nếu bạn giải nén lệch một cấp,
    thư mục con duy nhất chứa ảnh cũng được chấp nhận. Trả về đường dẫn dạng Path.
    """
    d = Path(images_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"{d} không phải thư mục")
    if any(d.glob("*")) and any(d.glob(f"*{ext}") for ext in IMG_EXTS):
        return d
    subdirs = sorted(p for p in d.iterdir() if p.is_dir())
    for sub in subdirs:
        if any(sub.glob(f"*{ext}") for ext in IMG_EXTS):
            print(f"[dataset] dùng thư mục ảnh con: {sub}")
            return sub
    raise FileNotFoundError(f"Không tìm thấy file ảnh (*.jpg) nào trong {d} hay thư mục con của nó")


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path, strict_ratio: bool = False) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). In ra và trả về dict số liệu.

    Kiểm tra (mỗi ý lỗi thì raise để dừng ngay):
      1. số ảnh mỗi tập và số ảnh mừng lớp trong từng tập (kỳ vọng xấp xỉ 60/20/20)
      2. giao của từng cặp tập theo Filename phải RỖNG (train∩val, train∩test, val∩test)
      3. hợp ba tập phải bằng đúng 17.509 ảnh
      4. mọi Filename đều tồn tại trong `images_dir`

    Ý (1) chỉ *cảnh báo* nếu lệch quá 1 điểm phần trăm so với 60/20/20, vì lớp `Negative`
    không được phân tầng khi chia nên tỉ lệ có thể lệch tự nhiên. Đặt `strict_ratio=True`
    để biến cảnh báo thành lỗi.
    """
    splits = {"train": train_df, "val": val_df, "test": test_df}
    img_dir = resolve_images_dir(images_dir)

    n = {k: int(len(v)) for k, v in splits.items()}
    per_class = {k: v["Label"].value_counts().reindex(range(NUM_CLASSES), fill_value=0)
                    .sort_index().astype(int).tolist()
                 for k, v in splits.items()}
    total = sum(n.values())

    sets = {k: set(v["Filename"]) for k, v in splits.items()}
    overlap = {
        "train_val": sorted(sets["train"] & sets["val"]),
        "train_test": sorted(sets["train"] & sets["test"]),
        "val_test": sorted(sets["val"] & sets["test"]),
    }

    missing = {k: sorted(s - {p.name for p in img_dir.glob("*") if p.is_file()})
               for k, s in sets.items()}

    report = {
        "images_dir": str(img_dir),
        "n": n,
        "per_class": per_class,
        "class_names": CLASS_NAMES,
        "overlap": {k: len(v) for k, v in overlap.items()},
        "union": len(sets["train"] | sets["val"] | sets["test"]),
        "expected_total": EXPECTED_TOTAL,
        "missing_files": {k: len(v) for k, v in missing.items()},
    }

    # --- in báo cáo ---
    print(f"[check_split] thư mục ảnh: {img_dir}")
    print(f"{'tập':<6}{'số ảnh':>8}{'tỉ lệ':>9}   (kỳ vọng {', '.join(f'{k} {v:.0%}' for k, v in NOMINAL_SHARE.items())})")
    for k, v in n.items():
        share = v / total if total else 0.0
        dev = share - NOMINAL_SHARE[k]
        flag = ""
        if abs(dev) > SHARE_TOL:
            flag = f"  <-- LỆCH {dev:+.1%} so với {NOMINAL_SHARE[k]:.0%}"
            if strict_ratio:
                raise ValueError(f"[check_split] tỉ lệ tập {k} lệch {dev:+.1%} (strict_ratio=True)")
        print(f"{k:<6}{v:>8}{share:>8.1%}{flag}")
    print(f"{'TỔNG':<6}{total:>8}")
    print("\n[check_split] số ảnh từng lớp (đối chiếu Table 1 của bài báo):")
    hdr = "".join(f"{name[:11]:>13}" for name in CLASS_NAMES)
    print(f"{'lớp':<6}" + hdr)
    for k in ("train", "val", "test"):
        print(f"{k:<6}" + "".join(f"{c:>13}" for c in per_class[k]))
    allc = [sum(per_class[k][i] for k in splits) for i in range(NUM_CLASSES)]
    print(f"{'tất cả':<6}" + "".join(f"{c:>13}" for c in allc))

    # --- các kiểm tra cứng ---
    for pair, files in overlap.items():
        if files:
            raise ValueError(f"[check_split] GIAO KHÔNG RỖNG {pair}: {len(files)} file, ví dụ {files[:5]}")
    if total != EXPECTED_TOTAL:
        raise ValueError(f"[check_split] hợp ba tập có {total} ảnh, phải bằng {EXPECTED_TOTAL}")
    if len(sets["train"] | sets["val"] | sets["test"]) != EXPECTED_TOTAL:
        raise ValueError("[check_split] hợp ba tập theo tên file khác 17.509 (trùng tên giữa các tập)")
    for k, files in missing.items():
        if files:
            raise ValueError(f"[check_split] {len(files)} file của tập {k} không tồn tại trong {img_dir}, "
                             f"ví dụ {files[:5]}")
    print(f"\n[check_split] OK: giao rỗng, hợp đúng {EXPECTED_TOTAL} ảnh, mọi file đều tồn tại.")
    return report


# --------------------------------------------------------------------------- #
# 3. Transform
# --------------------------------------------------------------------------- #
def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Tạo transform. `aug` chọn mức augmentation (xem `AUG_LEVELS`).

    Train (basic): RandomResizedCrop(img_size) + lật ngang + ToTensor + Normalize.
    Val/test: Resize(img_size) + CenterCrop(img_size) + ToTensor + Normalize.
               KHÔNG augmentation ngẫu nhiên khi đánh giá (S2).

    Lựa chọn: dùng `bilinear` cho mọi phép resize để khớp với hầu hết tag pretrained của
    timm; nếu bạn đổi sang tag nào yêu cầu bicubic thì đổi cả hai nhánh cho khớp.
    Lật dọc (basic_vflip) được tách riêng để kiểm chứng, không đưa vào `basic`.
    """
    from torchvision import transforms as T

    norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    eval_tf = T.Compose([
        T.Resize(img_size, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(img_size),
        T.ToTensor(),
        norm,
    ])
    if not train:
        return eval_tf

    if aug not in AUG_LEVELS:
        raise ValueError(f"aug={aug!r} không hợp lệ, chọn một trong {AUG_LEVELS}")

    ops = [
        T.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(3 / 4, 4 / 3),
                            interpolation=T.InterpolationMode.BILINEAR),
        T.RandomHorizontalFlip(),
    ]
    if aug == "color":
        ops += [T.ColorJitter(0.4, 0.4, 0.4, 0.1)]
    elif aug == "randaug":
        ops += [T.RandAugment(num_ops=2, magnitude=9)]
    elif aug == "basic_vflip":
        ops += [T.RandomVerticalFlip()]
    # aug == "trivial": không thêm phép ngẫu nhiên nào -> thay RandomResizedCrop bằng Resize
    if aug == "trivial":
        ops[0] = T.Resize(img_size, interpolation=T.InterpolationMode.BILINEAR)

    return T.Compose(ops + [T.ToTensor(), norm])


# --------------------------------------------------------------------------- #
# 4. Dataset
# --------------------------------------------------------------------------- #
class DeepWeedsDataset(Dataset):
    """Dataset đọc ảnh từ `images_dir` theo DataFrame (Filename, Label).

    `__getitem__(i)` trả về (ảnh đã transform, nhãn int, tên file str).
    Tên file cần có để ghi `predictions/*.csv` đúng định dạng của eval.py.

    `cache_in_ram=True` nạp trước toàn bộ ảnh dưới dạng mảng uint8 [N, H, W, 3].
    Toàn bộ ảnh DeepWeeds (17.509, 256x256) chiếm ~3,2 GiB. `max_cache_gb` để tự phòng vệ:
    nếu ảnh lớn hơn ngân sách thì tắt cache và in cảnh báo thay vì làm treo máy.
    """

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None,
                 cache_in_ram: bool = False, max_cache_gb: float = 6.0,
                 img_size: int = 256):
        self.df = df.reset_index(drop=True)
        self.images_dir = resolve_images_dir(images_dir)
        self.transform = transform
        self.files = self.df["Filename"].astype(str).tolist()
        self.labels = self.df["Label"].astype(np.int64).to_numpy()
        self._cache: np.ndarray | None = None

        if cache_in_ram:
            self._build_cache(img_size, max_cache_gb)

    def _build_cache(self, img_size: int, max_cache_gb: float) -> None:
        """Nạp ảnh vào RAM dưới dạng uint8. Dùng song song 2 luồng cho nhanh."""
        from concurrent.futures import ThreadPoolExecutor

        n = len(self.files)
        if not n:
            return
        # Cache original pixels; img_size is the transform's output size and
        # must not change the input to RandomResizedCrop or validation resize.
        with Image.open(self.images_dir / self.files[0]) as first:
            width, height = first.size
        est_gb = n * height * width * 3 / 1024 ** 3
        if est_gb > max_cache_gb:
            print(f"[dataset] BỎ cache RAM: ước lượng {est_gb:.1f} GiB > ngân sách {max_cache_gb} GiB")
            return

        buf = np.empty((n, height, width, 3), dtype=np.uint8)
        paths = [self.images_dir / f for f in self.files]

        def _read(i: int) -> None:
            with Image.open(paths[i]) as im:
                if im.size != (width, height):
                    raise ValueError("RAM cache requires equal original image dimensions")
                buf[i] = np.asarray(im.convert("RGB"), dtype=np.uint8)

        with ThreadPoolExecutor(max_workers=4) as pool:
            for _ in pool.map(_read, range(n)):
                pass
        self._cache = buf
        print(f"[dataset] đã cache {n} ảnh vào RAM ({est_gb:.2f} GiB, uint8)")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int):
        if self._cache is not None:
            img = Image.fromarray(self._cache[i])       # dùng chung buffer, không copy
        else:
            with Image.open(self.images_dir / self.files[i]) as im:
                img = im.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[i]), self.files[i]

    def class_counts(self) -> np.ndarray:
        """Số ảnh của từng lớp trong tập này (chỉ dùng cho tập TRAIN, S2)."""
        return np.bincount(self.labels, minlength=NUM_CLASSES).astype(np.float64)


# --------------------------------------------------------------------------- #
# 5. DataLoader
# --------------------------------------------------------------------------- #
def _worker_init_fn(worker_id: int) -> None:
    """Cố định nguồn ngẫu nhiên của từng worker để chạy lại cho ra kết quả giống nhau."""
    seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2,
                cache_in_ram: bool = False, seed: int = 0, drop_last: bool | None = None,
                img_size: int = 256, pin_memory: bool | None = None) -> DataLoader:
    """Tạo DataLoader.

      - train=True: shuffle (hoặc dùng sampler); train=False: KHÔNG shuffle, giữ thứ tự df
        (thứ tự phải ổn định để ghép logit với Filename)
      - sampler=None | "balanced": "balanced" dùng WeightedRandomSampler với trọng số
        1/(số ảnh của lớp) (trục D của GUIDE.md mục 3)
      - drop_last=True khi train: batch cuối quá nhỏ làm BatchNorm không ổn định.
        Với sampler="balanced" thì KHÔNG drop_last, vì mỗi batch đã phản ánh đúng phân phối
        lớp và batch cuối vẫn đủ lớn nếu batch_size thuộc loại nào đó.
      - pin_memory: None = tự động bật khi có CUDA (chỉ có lợi khi chuyển tensor sang GPU),
        tắt khi chạy CPU. Truyền True/False để ép.
      - num_workers=2 vì Kaggle T4 chỉ có 2 vCPU; seed cho worker để tái lập.
    """
    if sampler not in (None, "balanced"):
        raise ValueError(f"sampler={sampler!r} không hợp lệ, chọn None hoặc 'balanced'")
    if drop_last is None:
        drop_last = bool(train) and sampler is None
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    ds = DeepWeedsDataset(df, images_dir, transform, cache_in_ram=cache_in_ram, img_size=img_size)

    gen = torch.Generator()
    gen.manual_seed(seed)
    sampler_obj = None
    shuffle = bool(train)          # val/test: giữ nguyên thứ tự df để ghép logit với Filename
    if sampler == "balanced":
        counts = ds.class_counts()
        w = 1.0 / np.maximum(counts, 1.0)
        weights = torch.as_tensor(w[ds.labels], dtype=torch.double)
        sampler_obj = WeightedRandomSampler(weights, num_samples=len(ds), replacement=True, generator=gen)
        shuffle = False

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler_obj,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=_worker_init_fn,
        generator=gen,
        persistent_workers=bool(num_workers),   # giữ worker sống để không tải lại ảnh mỗi epoch
    )
