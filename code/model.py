"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

Hoàn thiện từ bộ khung `starter/`. Giữ nguyên tên hàm và kiểu dữ liệu vào/ra như docstring gốc.

    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                   -> None
    param_groups(model, lr_backbone, lr_head, weight_decay)  -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float

Hai điểm cần nêu trong báo cáo:
  - Tag trọng số: mỗi kiến trúc có nhiều tag pretrained (khác công thức huấn luyện). Hàm này
    ghi tag thực sự được tải vào `model.pretrained_tag` để đưa vào results.xlsx; không ghi
    tên kiến trúc cũng là so sánh không công bằng (GUIDE.md mục 2.1).
  - GMAC: công cụ khác nhau cho kết quả chênh nhau vài phần trăm, nên hàm đếm sẵn và trả về
    tên công cụ đã dùng (`count_gmacs_with_tool`). Mặc định dùng bộ đếm hook tự viết (chỉ đếm
    Conv2d và Linear) để không phụ thuộc thư viện ngoài trên Kaggle.
"""
from __future__ import annotations

# Gợi ý backbone (GUIDE.md mục 2.1). Tag trọng số của timm có thể đổi theo phiên bản:
# dùng timm.list_pretrained("resnet50*") để xem, và GHI LẠI tag bạn dùng trong results.xlsx.
SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",      # hoặc vit_small_patch16_224
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",        # mạng nhẹ
    "mobilenetv3": "mobilenetv3_large_100",      # mạng nhẹ
}

INIT_MODES = ("scratch", "frozen", "finetune")
BN_TYPES = ("_BatchNorm", "SyncBatchNorm", "BatchNorm")


# --------------------------------------------------------------------------- #
# 1. Tạo model
# --------------------------------------------------------------------------- #
def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune"):
    """Tạo model phân loại 9 lớp.

    `init` (trục A của GUIDE.md mục 3):
      - "scratch"  : pretrained=False, huấn luyện toàn bộ
      - "frozen"   : pretrained=True, đóng băng backbone, chỉ train head
      - "finetune" : pretrained=True, train toàn bộ

    TODO đã làm:
      - timm.create_model(name, pretrained=..., num_classes=num_classes, drop_rate=...)
        (timm tự thay head mới; head khởi tạo ngẫu nhiên)
      - nếu init == "frozen": gọi freeze_backbone(model)
      - ghi lại tên tag trọng số thực sự được tải (model.pretrained_cfg)

    `init="scratch"` ép `pretrained=False` bất kể tham số truyền vào, để không vô tình dùng
    trọng số tiền huấn luyện trong thí nghiệm "từ đầu".
    """
    import timm
    import torch.nn as nn

    if init not in INIT_MODES:
        raise ValueError(f"init={init!r} không hợp lệ, chọn một trong {INIT_MODES}")
    if init == "scratch":
        pretrained = False

    tname = SUGGESTED_BACKBONES.get(name, name)   # chấp nhận cả tên viết tắt lẫn tên timm đầy đủ
    model = timm.create_model(tname, pretrained=pretrained, num_classes=num_classes,
                              drop_rate=drop_rate)

    cfg = getattr(model, "pretrained_cfg", None) or {}
    model.pretrained_tag = (cfg.get("tag") or cfg.get("hf_hub_id") or "?") if pretrained else "random_init"
    model.timm_name = tname
    model.init_mode = init

    if init == "frozen":
        freeze_backbone(model)
    return model


def get_classifier(model):
    """Lấy module head của model (timm đặt tên là `get_classifier`)."""
    if hasattr(model, "get_classifier"):
        try:
            head = model.get_classifier()
            if head is not None:
                return head
        except (AttributeError, NotImplementedError):
            pass
    return list(model.children())[-1]              # dự phòng cho model tự viết


def bn_modules(model) -> list:
    """Danh sách module BatchNorm trong model (dùng cho chế độ đóng băng và cho EMA)."""
    import torch.nn as nn
    return [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]


def freeze_backbone(model) -> None:
    """Đóng băng mọi tham số trừ head.

    TODO đã làm:
      - requires_grad = False cho tham số backbone; head (model.get_classifier()) vẫn train
      - BatchNorm của backbone cũng phải ở chế độ eval (xem set_bn_eval)

    Vì sao phải giữ BN ở eval: BatchNorm ở chế độ train dùng/thống kê theo *batch* hiện tại
    để chuẩn hoá, nên đặc tính đầu ra thay đổi theo batch và gradient sẽ chảy vào thống kê
    BN — trong khi backbone đóng băng thì các thống kê đó đã bị "đóng băng" theo ý nghĩa là
    phải cố định. Xem `set_bn_eval`: train loop gọi `model.train()` (để head train) rồi gọi
    lại `set_bn_eval(model)` để đưa các BN về eval.
    """
    head = get_classifier(model)
    head_ids = {id(p) for p in head.parameters()}
    n_frozen = 0
    for p in model.parameters():
        if id(p) not in head_ids:
            p.requires_grad_(False)
            n_frozen += 1
    set_bn_eval(model)
    print(f"[model] đóng băng {n_frozen} tensor tham số; head "
          f"({sum(p.numel() for p in head.parameters()) / 1e6:.2f}M tham số) vẫn được huấn luyện")


def set_bn_eval(model) -> None:
    """Đặt mọi BatchNorm của model về chế độ eval (dùng chung cho backbone đóng băng)."""
    for m in bn_modules(model):
        m.eval()


def set_dropout_eval(model) -> None:
    """Đặt dropout về eval. Không dùng trong huấn luyện, chỉ để tiện khi suy luận."""
    import torch.nn as nn
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.eval()


# --------------------------------------------------------------------------- #
# 2. Nhóm tham số
# --------------------------------------------------------------------------- #
def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """Chia tham số thành 3 nhóm như slide Day 2, trang 52.

    - backbone có ndim > 1: lr = lr_backbone, weight_decay = weight_decay
    - norm và bias của backbone (ndim <= 1): lr = lr_backbone, weight_decay = 0
    - head mới: lr = lr_head (thường gấp 10 lần backbone), weight_decay = weight_decay

    TODO đã làm:
      - bỏ qua tham số requires_grad == False
      - trả về list[dict] dạng {"params": [...], "lr": ..., "weight_decay": ...}
      - (trục E) mở rộng: LR theo tầng nếu bạn muốn thử

    Về số nhóm: slide mô tả 3 nhóm. Vì nguyên tắc "weight_decay = 0 cho norm và bias" được
    áp cho cả backbone lẫn head, mà head hầu hết là Linear (weight ndim=2, bias ndim=1), nên
    head cũng bị tách làm 2 nhóm -> thực tế 4 nhóm cho kiến trúc điển hình. Đây là chi tiết
    làm đúng nguyên tắc hơn so với slide, được ghi rõ để không bị hiểu nhầm là lỗi.
    `names` giữ lại tên tham số để tiện kiểm tra.
    """
    head = get_classifier(model)
    head_ids = {id(p) for p in head.parameters()}
    groups: dict[tuple, dict] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_head = id(p) in head_ids
        wd = 0.0 if p.ndim <= 1 else weight_decay
        key = ("head" if is_head else "backbone", wd > 0.0)
        if key not in groups:
            lr = lr_head if is_head else lr_backbone
            groups[key] = {"params": [], "lr": lr, "weight_decay": wd,
                           "names": []}
        groups[key]["params"].append(p)
        groups[key]["names"].append(name)

    out = [g for g in groups.values() if g["params"]]
    desc = ", ".join(f"{'head' if k[0] == 'head' else 'backbone'}/wd={k[1]} "
                     f"({sum(p.numel() for p in g['params']) / 1e6:.2f}M, lr={g['lr']})"
                     for k, g in groups.items())
    print(f"[model] nhóm tham số ({len(out)} nhóm): {desc}")
    return out


# --------------------------------------------------------------------------- #
# 3. Đếm tham số và GMAC
# --------------------------------------------------------------------------- #
def count_params(model) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def _builtin_macs(model, img_size: int = 224) -> float:
    """Đếm MAC bằng forward hook: chỉ tính Conv2d và Linear (bỏ qua norm/activation/pooling).

    Đây là cùng quy ước với số trên slide (ResNet-50 ~4,1 GMAC). Với transformer, phép
    matmul trong lớp attention không đi qua Linear nên bị bỏ sót; phần đó nhỏ hơn đáng kể
    so với tổng (khoảng 5-10% với DeiT-S), và điều này được ghi rõ khi báo cáo.
    """
    import torch
    import torch.nn as nn

    total = 0
    handles = []

    def conv_hook(m, inp, out):
        nonlocal total
        oh, ow = out.shape[-2:]
        total += int(oh * ow * out.shape[1] * (m.in_channels // m.groups) * m.kernel_size[0]
                     * m.kernel_size[1])

    def linear_hook(m, inp, out):
        nonlocal total
        total += int((inp[0].numel() // m.in_features) * m.out_features * m.in_features)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))

    was_training = model.training
    model.eval()
    try:
        parameter = next(model.parameters())
        with torch.no_grad():
            model(torch.zeros(1, 3, img_size, img_size,
                              device=parameter.device, dtype=parameter.dtype))
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    return float(total) / 1e9


def count_gmacs_with_tool(model, img_size: int = 224, tool: str = "builtin") -> tuple[float, str]:
    """Đếm GMAC và trả về (gmacs, tên_công_cụ). `tool` ∈ {"builtin", "fvcore", "ptflops"}.

    Ghi rõ công cụ đã dùng; số có thể lệch vài phần trăm giữa các công cụ.
    """
    import torch
    import torch.nn as nn

    if tool == "builtin":
        return _builtin_macs(model, img_size), "builtin(hook: Conv2d+Linear)"

    if tool == "fvcore":
        try:
            from fvcore.nn import FlopCountAnalysis
        except ImportError as e:
            raise ImportError("Cần `pip install fvcore`") from e
        was_training = model.training
        model.eval()
        parameter = next(model.parameters())
        try:
            flops = FlopCountAnalysis(model, torch.zeros(1, 3, img_size, img_size,
                                                       device=parameter.device, dtype=parameter.dtype)).total()
        finally:
            model.train(was_training)
        return float(flops) / 1e9, "fvcore"          # fvcore đếm 1 MAC = 1 flop

    if tool == "ptflops":
        try:
            from ptflops import get_model_complexity_info
        except ImportError as e:
            raise ImportError("Cần `pip install ptflops`") from e
        macs, _ = get_model_complexity_info(model, (3, img_size, img_size), as_strings=False,
                                            print_per_layer_stat=False, verbose=False)
        return float(macs) / 1e9, "ptflops"

    raise ValueError(f"tool={tool!r} không hợp lệ")


def count_gmacs(model, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size (slide tính MAC, không phải FLOPs 2x).

    TODO đã làm: bộ đếm hook tự viết (không cần cài thêm thư viện); nếu muốn đối chiếu với
    công cụ khác thì dùng `count_gmacs_with_tool(model, img_size, tool="fvcore")`.
    Ghi rõ công cụ đã dùng; số có thể lệch vài phần trăm giữa các công cụ.
    """
    return count_gmacs_with_tool(model, img_size, tool="builtin")[0]
