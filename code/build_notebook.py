"""Regenerate the runnable, output-free lab notebook from maintained cell sources."""
from pathlib import Path
import nbformat as nb

cells = []


def md(text):
    cells.append(nb.v4.new_markdown_cell(text.strip()))


def py(text):
    cells.append(nb.v4.new_code_cell(text.strip()))


md("""
# DeepWeeds — thực nghiệm backbone, huấn luyện và suy luận

Notebook sử dụng phần triển khai trong `code/`. Chạy từng giai đoạn bằng `STAGE`, rồi
chạy lại notebook từ đầu với cùng thư mục output. Kết quả hoàn tất được tái sử dụng nếu
cấu hình không đổi. Mọi lựa chọn dựa trên **val**; test dùng ở chung kết.

Thứ tự: `setup` → `smoke` → `backbones` → `training` → `inference` → `final` → `export`.
Chạy thử không phải kết quả chính thức. Đọc `README.md`, `GUIDE.md`, `RUBRIC.md` ở thư mục gốc.
""")
md("""
## 0. Môi trường và cấu hình

Mở notebook từ repo hoặc `code/`. Cài PyTorch phù hợp GPU trước, sau đó cài các thư viện
trong `code/requirements.txt`. Colab/Kaggle có thể dùng `%pip install -r code/requirements.txt`
từ thư mục gốc repo. Đặt profile `local` cho RTX 2050 4 GB; `kaggle` dùng batch 32.
Giữ cùng batch size, epoch và seed khi so backbone. Profile local dùng batch 4 và worker 0.
""")
py("""
import os, sys, json, subprocess, dataclasses, platform
from pathlib import Path

ROOT = next((p for p in (Path.cwd(), *Path.cwd().parents)
             if (p / 'eval.py').is_file() and (p / 'code').is_dir()), None)
if ROOT is None:
    raise FileNotFoundError('Mở notebook từ thư mục repo hoặc code/')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'code'))
import runtime
import numpy as np, pandas as pd, torch, timm, matplotlib.pyplot as plt
import dataset as ds
import train
from lab_workflow import (base_config, experiment_plan, run_experiments, best_config,
                         compare_inference, lock_selection, run_final, export_results, smoke)
if Path(train.__file__).resolve().parent != ROOT / 'code':
    raise RuntimeError('Kernel đã import starter/train.py; restart kernel rồi chạy lại')

STAGE = os.environ.get('LAB_STAGE', 'setup')
PROFILE = 'local'                 # local | kaggle
ALLOW_FINAL_TEST = False          # bật khi đã chốt cấu hình trên val
SEEDS = (0, 1, 2)
SELECTED_METHOD = None            # None: chọn theo macro-F1 val, ECE rồi độ trễ
assert STAGE in ('setup', 'smoke', 'backbones', 'training', 'inference', 'final', 'export')
BASE = base_config(ROOT, PROFILE)
print('Repo:', ROOT, '| Stage:', STAGE, '| Profile:', PROFILE)
print('python:', platform.python_version(), '| torch:', torch.__version__, '| timm:', timm.__version__)
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
print('Ảnh:', BASE.images_dir, '| batch:', BASE.batch_size, '| workers:', BASE.num_workers)
(ROOT / 'runs').mkdir(exist_ok=True)
(ROOT / 'runs/environment.json').write_text(json.dumps({
    'python': platform.python_version(), 'torch': torch.__version__, 'timm': timm.__version__,
    'pandas': pd.__version__, 'profile': PROFILE, 'seeds': SEEDS,
    'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
}, indent=2), encoding='utf-8')
""")
md("""
### Dữ liệu

Cần ảnh ở `images/` hoặc `data/images/`, cùng 4 CSV nguyên bản tại `data/labels/`:
`labels.csv`, `train_subset0.csv`, `val_subset0.csv`, `test_subset0.csv`.
Nếu chưa có, tải theo README; kiểm tra MD5 của `images.zip` trước giải nén.
Không chỉnh sửa CSV hoặc chia lại theo seed.
""")
py("""
train_df, val_df, test_df = ds.load_split(BASE.labels_dir, fold=0)
split_report = ds.check_split(train_df, val_df, test_df, BASE.images_dir, strict_ratio=True)
(ROOT / 'runs/eda.json').write_text(json.dumps(split_report, indent=2, ensure_ascii=False), encoding='utf-8')
counts = pd.DataFrame({name: frame['Label'].value_counts().reindex(range(9), fill_value=0)
                      for name, frame in [('train', train_df), ('val', val_df), ('test', test_df)]})
counts.index = ds.CLASS_NAMES
display(counts)
ax = counts.plot.bar(figsize=(12, 4), rot=30)
ax.set_ylabel('Số ảnh'); ax.set_title('Fold 0 — phân bố lớp'); plt.tight_layout()
eda_dir = ROOT / 'curves/_eda'; eda_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(eda_dir / 'class_distribution.png', dpi=140); plt.show()
""")
py("""
from PIL import Image
fig, axes = plt.subplots(9, 3, figsize=(9, 23))
image_dir = ds.resolve_images_dir(BASE.images_dir)
for label, name in enumerate(ds.CLASS_NAMES):
    examples = train_df[train_df.Label == label].head(3)
    for ax, (_, row) in zip(axes[label], examples.iterrows()):
        with Image.open(image_dir / row.Filename) as image:
            ax.imshow(image.convert('RGB'))
        ax.set_title(f'{name} | {row.Filename}', fontsize=8); ax.axis('off')
plt.tight_layout(); plt.savefig(eda_dir / 'examples.png', dpi=120); plt.show()
""")
md("""
## 1. Kiểm tra pipeline và chạy thử

Self-test kiểm tra loss, Mixup/CutMix, BN, temperature, nhóm tham số và hợp đồng dự đoán.
Stage `smoke` chạy 1 epoch MobileNetV3 scratch trên 2 ảnh/lớp train và val ở 64×64;
không dùng test. Log nằm ở `runs/_smoke/`, biểu đồ ở `curves/_smoke/`.
Đây là kiểm tra đường đi dữ liệu và checkpoint, không phải thí nghiệm so sánh.
""")
py("""
if STAGE in ('setup', 'smoke'):
    subprocess.run([sys.executable, '-X', 'utf8', str(ROOT / 'code/selftest.py')], check=True, cwd=ROOT)
if STAGE == 'smoke':
    smoke_summary = smoke(ROOT, PROFILE)
    display(pd.DataFrame([smoke_summary]))
""")
md("""
## 2. So sánh 5 backbone

ResNet50, ConvNeXt-Tiny, DeiT-Small, Swin-Tiny và MobileNetV3; cùng recipe, seed, 12 epoch.
Chọn backbone có macro-F1 val tốt nhất để đi tiếp. Xem cả thời gian và kích thước trước khi
viết kết luận. Trọng số ImageNet được tải lần đầu, cần Internet hoặc cache sẵn.
""")
py("""
backbone_configs = experiment_plan(BASE, 'backbones')
display(pd.DataFrame([dataclasses.asdict(c) for c in backbone_configs])[
    ['exp_id', 'backbone', 'init', 'epochs', 'batch_size', 'seed']])
if STAGE == 'backbones':
    backbone_results = run_experiments(backbone_configs)
    display(backbone_results.sort_values('val_macro_f1', ascending=False))
""")
md("""
## 3. Công thức huấn luyện

Chỉ đổi một yếu tố so với T00: khởi tạo (finetune/scratch/frozen), augmentation
(basic/trivial/color), loss (CE/label smoothing/focal); bổ sung CutMix và Mixup.
T00 là baseline trên backbone thắng Bước 2. Có thể thêm cấu hình kết hợp bằng một exp_id mới.
Notebook không tự thay đổi các thí nghiệm đã hoàn tất.
""")
py("""
training_configs = None
if STAGE in ('training', 'inference', 'final'):
    backbone_winner = best_config(backbone_configs)
    recipe_base = dataclasses.replace(BASE, backbone=backbone_winner.backbone)
    training_configs = experiment_plan(recipe_base, 'training')
    display(pd.DataFrame([dataclasses.asdict(c) for c in training_configs])[
        ['exp_id', 'backbone', 'init', 'aug', 'loss', 'mix']])
    if STAGE == 'training':
        training_results = run_experiments(training_configs)
        display(training_results.sort_values('val_macro_f1', ascending=False))
""")
md("""
## 4. So sánh suy luận trên val

I00: một view; I01: lật ngang, gộp xác suất; I02: five-crop (crop 7/8 rồi resize về
kích thước model); I03: lật ngang, gộp logit; I04: temperature scaling khớp trên val.
Tất cả đánh giá FP32. Độ trễ batch 1 đo chính các view và cách gộp trên GPU, warmup 10,
50 lần đo, CUDA synchronize; không gồm đọc/giải mã ảnh.
""")
py("""
if STAGE == 'inference':
    training_winner = best_config(training_configs)
    inference_results = compare_inference(training_winner)
    display(inference_results.sort_values(['val_macro_f1', 'val_ece'], ascending=[False, True]))
    export_results(ROOT)
""")
md("""
## 5. Chung kết ≥3 seed

Sau khi xem kết quả val, đặt `STAGE='final'` và `ALLOW_FINAL_TEST=True`.
`SELECTED_METHOD=None` chọn macro-F1 val cao nhất, hòa thì ECE thấp nhất rồi p95 thấp nhất;
có thể đặt tên phương pháp trong bảng để chốt lựa chọn riêng.
Cấu hình được khóa ở `runs/selection.json`. Baseline T00 và F01 chạy cùng 3 seed.
File test đã có sẽ được giữ nguyên; không chạy lại test để chọn cấu hình.
Temperature được khớp riêng trên val của từng seed và áp dụng sang test.
""")
py("""
if STAGE == 'final':
    if not ALLOW_FINAL_TEST:
        print('Chốt cấu hình trên val rồi bật ALLOW_FINAL_TEST để chạy chung kết')
    else:
        training_winner = best_config(training_configs)
        idir = Path(BASE.out_dir) / 'inference' / training_winner.exp_id / f'seed{training_winner.seed}'
        candidates = [json.loads(p.read_text(encoding='utf-8')) for p in idir.glob('I*.json')]
        if len(candidates) < 5:
            raise RuntimeError('Chạy stage inference trước khi chốt chung kết')
        chosen = sorted(candidates, key=lambda r: (-r['val_macro_f1'], r['val_ece'], r['latency']['p95']))[0]
        method = SELECTED_METHOD or chosen['method']
        selection_path = lock_selection(training_winner, method)
        run_final(selection_path, training_configs[0], seeds=SEEDS)
        export_results(ROOT)
""")
md("""
## 6. Đánh giá chuẩn và sản phẩm

`results.xlsx` chỉ lấy số từ log đã lưu; giai đoạn chưa chạy hiển thị chưa có kết quả.
`Final` dùng mean và std mẫu (`ddof=1`); `PerClass` lấy trực tiếp từ CSV dự đoán test.
Chạy `eval.py` gốc để đối chiếu, rồi viết báo cáo dựa trên số thật và dẫn nguồn tham khảo.
""")
py("""
if STAGE in ('final', 'export'):
    print('Workbook:', export_results(ROOT))
    if all(list((ROOT / 'predictions').glob(f'{tag}_seed*_test.csv')) for tag in ('F01', 'T00')):
        for tag in ('F01', 'T00'):
            subprocess.run([sys.executable, '-X', 'utf8', str(ROOT / 'eval.py'), 'score',
                '--pred', str(ROOT / f'predictions/{tag}_seed*_test.csv'),
                '--test-csv', str(Path(BASE.labels_dir) / 'test_subset0.csv'),
                '--labels', str(Path(BASE.labels_dir) / 'labels.csv'), '--tag', tag,
                '--out', str(ROOT / 'eval_out')], check=True)
        grade = [sys.executable, '-X', 'utf8', str(ROOT / 'eval.py'), 'grade',
                 '--final', str(ROOT / 'predictions/F01_seed*_test.csv'),
                 '--baseline', str(ROOT / 'predictions/T00_seed*_test.csv'),
                 '--test-csv', str(Path(BASE.labels_dir) / 'test_subset0.csv'),
                 '--labels', str(Path(BASE.labels_dir) / 'labels.csv'),
                 '--final-val', str(ROOT / 'predictions/F01_seed*_val.csv'),
                 '--out', str(ROOT / 'eval_out')]
        if list((ROOT / 'predictions').glob('F01uncal_seed*_test.csv')):
            grade += ['--uncal', str(ROOT / 'predictions/F01uncal_seed*_test.csv')]
        subprocess.run(grade, check=True)
    else:
        print('Chưa đủ dự đoán chung kết và baseline để chạy eval.py')
""")
md("""
### Báo cáo và tái lập

Theo dàn ý GUIDE mục 6.3: thiết lập và EDA; bảng backbone; ablation; suy luận/độ trễ;
mean ± std test; phân tích Chinee Apple/Snake Weed; lựa chọn cuối và hạn chế.
Không dùng kết quả SMOKE để kết luận chất lượng model. Nộp `results.xlsx`, báo cáo,
code, notebook, `curves/` và `predictions/`; không commit ảnh, archive hoặc checkpoint.
Xem `code/README.md` để chạy CLI, test và lựa chọn profile.
""")

notebook = nb.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"}})
nb.validate(notebook)
nb.write(notebook, Path(__file__).with_name("lab_day2.ipynb"))
print(f"Created notebook: {len(cells)} cells")
