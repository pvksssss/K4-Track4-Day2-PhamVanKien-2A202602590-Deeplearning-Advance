"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

Hoàn thiện từ bộ khung `starter/`. Giữ nguyên tên hàm và kiểu dữ liệu vào/ra như docstring gốc.

Quy tắc đo (vi phạm bị trừ điểm, RUBRIC mục 3):
  - warmup: bỏ >= 10 lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() (hoặc CUDA event) TRƯỚC và SAU đoạn cần đo
  - >= 50 lần đo, báo cáo p50, p95, p99 (không chỉ trung bình)
  - ghi rõ GPU, dtype (FP32/AMP/FP16), batch, độ phân giải, có/không gộp BN, phiên bản torch
  - chọn và ghi rõ có tính tiền xử lý hay không

Lựa chọn đo: `latency_report` đo **chỉ phần forward** của model trên tensor đầu vào đã nằm
sẵn trên GPU (`includes_preprocess=False`), vì tiền xử lý (resize/normalize) giống nhau ở mọi
phương pháp nên tách ra sẽ dễ so sánh hơn. Nếu muốn tính cả tiền xử lý, truyền
`preprocess=<callable>` và báo cáo sẽ ghi `includes_preprocess=True`.
"""
from __future__ import annotations

import time

import numpy as np


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo thời gian một hàm `fn()` (không tham số), trả về mili-giây.

    `sync` là hàm đồng bộ (ví dụ torch.cuda.synchronize) hoặc None trên CPU.

    TODO đã làm:
      - chạy warmup lần đầu rồi bỏ
      - với mỗi lần đo: sync(); t0 = time.perf_counter(); fn(); sync(); lấy hiệu * 1000
      - trả về {"p50": ..., "p95": ..., "p99": ..., "mean": ..., "n": iters}
    Gợi ý: dùng numpy.percentile. Ngoài ra trả kèm `min` và `max` để thấy đuôi phân phối.
    """
    if warmup < 10:
        print(f"[bench] CẢNH BÁO: chỉ warmup {warmup} lần; GUIDE.md yêu cầu >= 10")
    if iters < 50:
        print(f"[bench] CẢNH BÁO: chỉ {iters} lần đo; GUIDE.md yêu cầu >= 50")

    for _ in range(warmup):
        fn()
    if sync is not None:
        sync()

    times = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        if sync is not None:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        times[i] = (time.perf_counter() - t0) * 1000.0
    return {
        "p50": float(np.percentile(times, 50)),
        "p95": float(np.percentile(times, 95)),
        "p99": float(np.percentile(times, 99)),
        "mean": float(times.mean()),
        "min": float(times.min()),
        "max": float(times.max()),
        "n": int(iters),
        "warmup": int(warmup),
    }


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100, preprocess=None,
                   channels_last: bool = False) -> dict:
    """Đo độ trễ forward của `model` với đầu vào ngẫu nhiên (batch_size, 3, img_size, img_size).

    Trả về dict có thể ghi thẳng vào sheet `Latency` của results.xlsx:
        {"gpu": ..., "dtype": ..., "batch": ..., "img_size": ..., "p50": ..., "p95": ..., "p99": ...,
         "images_per_s": batch_size / (p50 / 1000), "torch": torch.__version__}

    TODO đã làm:
      - model.eval(), torch.inference_mode()
      - dtype: "fp32" | "amp" (autocast fp16) | "fp16" (model.half())
      - gọi bench(...) với sync phù hợp; lấy tên GPU bằng torch.cuda.get_device_name
      - Nhớ: ở batch 1, AMP có thể CHẬM hơn FP32 (slide trang 73): đo thật, đừng giả định

    `channels_last=True` đặt tensor theo layout NHWC, thường nhanh hơn với tensor core fp16.
    """
    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Yêu cầu đo trên CUDA nhưng máy không có GPU")
    dev = torch.device(device)
    sync = torch.cuda.synchronize if dev.type == "cuda" else None

    model = model.to(dev).eval()
    x = torch.randn(batch_size, 3, img_size, img_size, device=dev)

    autocast_on = False
    if dtype == "fp16":
        model = model.half()
        x = x.half()
    elif dtype == "amp":
        autocast_on = True
    elif dtype != "fp32":
        raise ValueError(f"dtype={dtype!r} không hợp lệ, chọn 'fp32', 'amp' hoặc 'fp16'")

    if channels_last:
        model = model.to(memory_format=torch.channels_last)
        x = x.to(memory_format=torch.channels_last)

    def step():
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=autocast_on):
            model(preprocess(x) if preprocess is not None else x)

    r = bench(step, warmup=warmup, iters=iters, sync=sync)
    r.update({
        "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU",
        "dtype": dtype,
        "batch": int(batch_size),
        "img_size": int(img_size),
        "channels_last": bool(channels_last),
        "includes_preprocess": preprocess is not None,
        "images_per_s": batch_size / (r["p50"] / 1000.0),
        "torch": torch.__version__,
    })
    print(f"[bench] {r['gpu']} | {dtype} | batch {batch_size} @ {img_size} | "
          f"p50 {r['p50']:.2f} ms | p95 {r['p95']:.2f} ms | p99 {r['p99']:.2f} ms | "
          f"{r['images_per_s']:.1f} img/s")
    return r


def tta_latency(model, k_views: int, **kw) -> dict:
    """Độ trễ của TTA K view: xấp xỉ K lần một lượt chạy (slide trang 63). TODO đã làm.

    Đo thật phần "chạy K view rồi gộp" (dùng lật ngang + trung bình xác suất) và so với
    `k_views * p50` của một lượt chạy đơn, để kiểm chứng nhận định "chi phí gần tuyến tính".
    """
    import torch

    from inference import aggregate_views, view_hflip, view_identity

    device = kw.pop("device", "cuda")
    img_size = kw.pop("img_size", 224)
    batch_size = kw.pop("batch_size", 1)
    warmup = kw.pop("warmup", 10)
    iters = kw.pop("iters", 100)
    dev = torch.device(device)
    sync = torch.cuda.synchronize if dev.type == "cuda" else None

    model = model.to(dev).eval()
    x = torch.randn(batch_size, 3, img_size, img_size, device=dev)
    # K view: view đầu là ảnh gốc, các view sau xen kẽ lật ngang (đủ để đo chi phí K lượt chạy)
    views = [view_identity if i % 2 == 0 else view_hflip for i in range(k_views)]

    def step():
        with torch.inference_mode():
            logits = [model(v(x)) for v in views]
        aggregate_views([t.detach().cpu().numpy() for t in logits], space="prob")

    r = bench(step, warmup=warmup, iters=iters, sync=sync)
    single = latency_report(model, batch_size, img_size, dtype=kw.get("dtype", "fp32"),
                            device=device, warmup=warmup, iters=iters)
    r.update({
        "k_views": int(k_views),
        "single_view_p50": single["p50"],
        "k_times_single_p50": k_views * single["p50"],
        "overhead_vs_linear": r["p50"] - k_views * single["p50"],
        "gpu": single["gpu"], "dtype": single["dtype"], "batch": batch_size,
        "img_size": img_size, "torch": single["torch"],
    })
    print(f"[bench] TTA K={k_views}: p50 {r['p50']:.2f} ms (K x p50 = "
          f"{r['k_times_single_p50']:.2f} ms, chênh {r['overhead_vs_linear']:+.2f} ms)")
    return r


def sweep_latency(model, batch_sizes=(1, 32), img_size: int = 224, dtypes=("fp32", "amp"),
                  device: str = "cuda", iters: int = 100) -> list[dict]:
    """Quét nhiều tổ hợp batch x dtype để lấp sheet `Latency` của results.xlsx.

    Chạy ở cả hai điều kiện mà GUIDE.md mục 4.1 yêu cầu: batch 1 (giống robot) và một batch
    lớn hơn để đo thông lượng ảnh/giây.
    """
    rows = []
    for bs in batch_sizes:
        for dt in dtypes:
            rows.append(latency_report(model, bs, img_size, dtype=dt, device=device, iters=iters))
    return rows