"""Bundle the maintained lab into one Kaggle-uploadable notebook."""
from pathlib import Path
import copy
import json
import nbformat as nb

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
FILES = ['eval.py', 'README.md', 'GUIDE.md', 'RUBRIC.md'] + [
    'code/' + name for name in ('runtime.py', 'dataset.py', 'model.py', 'losses.py',
                                'train.py', 'inference.py', 'benchmark.py',
                                'lab_workflow.py', 'selftest.py', 'README.md', 'requirements.txt')]
payload = {name: (ROOT / name).read_bytes().decode('utf-8') for name in FILES}

bootstrap = '''
from pathlib import Path
import csv, importlib.util, subprocess, sys

KAGGLE_INPUT_ROOT = Path(globals().get('KAGGLE_INPUT_ROOT', '/kaggle/input'))
LAB_WORK_ROOT = Path(globals().get('LAB_WORK_ROOT', '/kaggle/working/deepweeds-lab'))
LAB_WORK_ROOT.mkdir(parents=True, exist_ok=True)

# Source files are embedded in this notebook; no repo download is needed.
BUNDLED_FILES = __PAYLOAD__
for relative, source in BUNDLED_FILES.items():
    destination = LAB_WORK_ROOT / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.encode('utf-8'))

required_csv = ('labels.csv', 'train_subset0.csv', 'val_subset0.csv', 'test_subset0.csv')
manual_labels = globals().get('MANUAL_LABELS_DIR')
if manual_labels:
    candidates = [Path(manual_labels)]
else:
    candidates = sorted({p.parent for p in KAGGLE_INPUT_ROOT.rglob('labels.csv')
                         if all((p.parent / name).is_file() for name in required_csv)})
if not candidates:
    raise FileNotFoundError('Add Input: cần đủ 4 CSV nhãn/fold 0 trong cùng folder labels')
if len(candidates) != 1:
    raise ValueError(f'Nhiều folder labels: {candidates}. Đặt MANUAL_LABELS_DIR ở cell cấu hình')
KAGGLE_LABELS_DIR = candidates[0]
if not all((KAGGLE_LABELS_DIR / name).is_file() for name in required_csv):
    raise FileNotFoundError('Folder labels thiếu CSV bắt buộc: ' + str(KAGGLE_LABELS_DIR))
with (KAGGLE_LABELS_DIR / 'labels.csv').open(encoding='utf-8-sig', newline='') as stream:
    reference = [row['Filename'] for _, row in zip(range(3), csv.DictReader(stream))]
if not reference:
    raise ValueError('labels.csv không có dữ liệu')
manual_images = globals().get('MANUAL_IMAGES_DIR')
if manual_images:
    image_candidates = [Path(manual_images)]
else:
    image_candidates = sorted({p.parent for p in KAGGLE_INPUT_ROOT.rglob(reference[0])
                               if all((p.parent / name).is_file() for name in reference)})
if not image_candidates:
    raise FileNotFoundError('Add Input: chưa tìm thấy folder images chứa các tên ảnh trong labels.csv')
if len(image_candidates) != 1:
    raise ValueError(f'Nhiều folder ảnh: {image_candidates}. Đặt MANUAL_IMAGES_DIR ở cell cấu hình')
KAGGLE_IMAGES_DIR = image_candidates[0]
if not all((KAGGLE_IMAGES_DIR / name).is_file() for name in reference):
    raise FileNotFoundError('Đường dẫn ảnh không khớp CSV: ' + str(KAGGLE_IMAGES_DIR))

# Preserve Kaggle's installed torch/torchvision CUDA pair. Install other packages only if absent.
packages = {'timm': 'timm', 'numpy': 'numpy', 'pandas': 'pandas', 'PIL': 'Pillow',
            'matplotlib': 'matplotlib', 'sklearn': 'scikit-learn', 'openpyxl': 'openpyxl'}
missing = [package for module, package in packages.items() if importlib.util.find_spec(module) is None]
if missing:
    if not globals().get('INSTALL_MISSING_DEPENDENCIES', True):
        raise ImportError('Missing notebook dependencies: ' + ', '.join(missing))
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *missing], check=True)
for module in ('torch', 'torchvision'):
    if importlib.util.find_spec(module) is None:
        raise ImportError('Kaggle cần có torch/torchvision; chọn môi trường notebook GPU')
print('Ảnh:', KAGGLE_IMAGES_DIR)
print('CSV:', KAGGLE_LABELS_DIR)
print('Code và output:', LAB_WORK_ROOT)
'''.strip().replace('__PAYLOAD__', json.dumps(payload, ensure_ascii=False, indent=2))

original = nb.read(CODE / 'lab_day2.ipynb', as_version=4)
cells = copy.deepcopy(original.cells)
cells[0].source = '''# DeepWeeds — notebook Kaggle độc lập

Chỉ upload **notebook này và dataset**; không cần upload code Python, eval.py hoặc clone repo.
Code đánh giá và huấn luyện được gói sẵn, tự ghi ra thư mục làm việc khi chạy.

Thêm dataset vào Input; ảnh và CSV có thể nằm chung một dataset hoặc ở hai dataset riêng.
Bật GPU và Internet (cần Internet khi tải trọng số ImageNet hoặc cài thư viện thiếu).
Chạy `setup` trước, sau đó `smoke`, `backbones`, `training`, `inference`, `final`, `export`.
Mọi lựa chọn dựa trên val; chỉ bước final dùng test.
'''
cells[1].source = '''## 0. Cấu hình Kaggle

Cell dưới chứa toàn bộ code nguồn. Khi chạy, notebook tự tạo code trong
`/kaggle/working/deepweeds-lab/` và tìm dataset trong `/kaggle/input/`.
Input chỉ được đọc; checkpoint, log và biểu đồ ghi vào working.

Nếu có nhiều bản dataset trùng nhau, điền MANUAL_IMAGES_DIR và MANUAL_LABELS_DIR.
Không cần đổi đường dẫn nếu chỉ gắn một bộ ảnh và một bộ CSV tương ứng.
'''
config = nb.v4.new_code_cell('''# Chỉ chỉnh các lựa chọn chạy tại đây.
STAGE = 'setup'                   # setup | smoke | backbones | training | inference | final | export
ALLOW_FINAL_TEST = False          # bật khi chốt cấu hình và chạy final
MANUAL_IMAGES_DIR = None          # ví dụ: '/kaggle/input/my-dataset/images'
MANUAL_LABELS_DIR = None          # ví dụ: '/kaggle/input/my-dataset/data/labels'
INSTALL_MISSING_DEPENDENCIES = True
''', metadata={'tags': ['configuration']})

setup = cells[2].source
start = setup.index('ROOT = next(')
end = setup.index('sys.path.insert(0, str(ROOT))')
setup = setup[:start] + 'ROOT = LAB_WORK_ROOT\n' + setup[end:]
setup = setup.replace("STAGE = os.environ.get('LAB_STAGE', 'setup')", "STAGE = globals().get('STAGE', 'setup')")
setup = setup.replace("PROFILE = 'local'", "PROFILE = 'kaggle'")
setup = setup.replace('ALLOW_FINAL_TEST = False', "ALLOW_FINAL_TEST = globals().get('ALLOW_FINAL_TEST', False)")
setup = setup.replace('BASE = base_config(ROOT, PROFILE)', '''BASE = base_config(ROOT, PROFILE)
BASE.images_dir = str(KAGGLE_IMAGES_DIR)
BASE.labels_dir = str(KAGGLE_LABELS_DIR)''')
cells[2].source = setup
cells[3].source = '''### Dữ liệu và kiểm tra

Notebook đã tìm đường dẫn images/CSV trong Input ở trên. Cell kế tiếp kiểm tra toàn bộ
17.509 ảnh theo fold 0: số ảnh, giao giữa các tập, union và file thiếu.
Dataset hiện có một nhãn lệch giữa labels.csv và train_subset0.csv cho
`20170714-110407-3.jpg`; giữ CSV nguyên bản theo đề bài, ghi nhận trong báo cáo.
'''
cells[7].source = cells[7].source.replace('smoke(ROOT, PROFILE)',
    'smoke(ROOT, PROFILE, images_dir=BASE.images_dir, labels_dir=BASE.labels_dir)')
cells[18].source += '''

Tải kết quả từ `/kaggle/working/deepweeds-lab/` sau khi hoàn tất. Khi tạo phiên Kaggle
mới, thư mục working của phiên trước có thể không còn. Muốn chạy tiếp stage sau,
hãy giữ kết quả của phiên trước và đưa checkpoint/log về cùng working trước khi chạy.
'''
notebook = nb.v4.new_notebook(cells=[cells[0], cells[1], config,
    nb.v4.new_code_cell(bootstrap, metadata={'tags': ['bootstrap']}), *cells[2:]],
    metadata=copy.deepcopy(original.metadata))
nb.validate(notebook)
output = CODE / 'lab_day2_kaggle.ipynb'
nb.write(notebook, output)
print('Created standalone Kaggle notebook:', output, '| bundled files:', len(FILES))
