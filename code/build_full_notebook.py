"""Build the separate all-stage Kaggle notebook without rewriting existing notebooks."""
from pathlib import Path
import ast
import copy
import json

import nbformat as nb

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent


def build():
    original = nb.read(CODE / 'lab_day2_kaggle.ipynb', as_version=4)
    bootstrap = next(c.source for c in original.cells
                     if c.metadata.get('tags') == ['bootstrap'])
    tree = ast.parse(bootstrap)
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == 'BUNDLED_FILES' for t in node.targets))
    names = list(ast.literal_eval(assignment.value)) + ['code/full_workflow.py']
    payload = {name: (ROOT / name).read_bytes().decode('utf-8') for name in names}
    lines = bootstrap.splitlines(keepends=True)
    lines[assignment.lineno - 1:assignment.end_lineno] = [
        'BUNDLED_FILES = ' + json.dumps(payload, ensure_ascii=False, indent=2) + '\n']
    bootstrap = ''.join(lines)
    setup = next(c.source for c in original.cells if c.cell_type == 'code'
                 and 'ROOT = LAB_WORK_ROOT' in c.source)
    setup = setup.replace("STAGE = globals().get('STAGE', 'setup')", "STAGE = 'setup'")
    setup = setup.replace('SEEDS = (0, 1, 2)', "SEEDS = globals().get('SEEDS', (0, 1, 2))")
    setup += '''
if not torch.cuda.is_available():
    raise RuntimeError('Full run cần GPU; bật Accelerator GPU trước khi Run All')
from full_workflow import restore_outputs, run_all
if PREVIOUS_OUTPUT_DIR:
    restore_outputs(PREVIOUS_OUTPUT_DIR, ROOT)
'''
    eda = [copy.deepcopy(c) for c in original.cells if c.cell_type == 'code' and (
        'split_report = ds.check_split' in c.source or 'fig, axes = plt.subplots(9, 3' in c.source)]
    evaluation = next(c.source for c in original.cells if c.cell_type == 'code'
                      and "if STAGE in ('final', 'export')" in c.source)
    cells = [
        nb.v4.new_markdown_cell('''# DeepWeeds — Kaggle Full Run All

Một notebook độc lập, chạy **setup → 5 backbones → T00–T08 + T09 kết hợp → inference
→ khóa lựa chọn trên val → baseline/final ≥3 seed → export/eval**.
Không phải đổi STAGE, copy cell hoặc tạo notebook mới giữa các bước.

**Bật GPU + Internet, thêm ảnh và CSV vào Input, rồi Save & Run All.**
Mặc định cell cấu hình cho phép chạy test ở cuối; đọc quy tắc dưới đây trước khi chạy.
Mọi lựa chọn tự động chỉ dùng val. Sau khi có kết quả test, không sửa recipe để chọn điểm tốt hơn.

Một full run có thể vượt giới hạn giờ GPU, thời gian phiên hoặc dung lượng Kaggle.
Notebook không vượt qua các giới hạn đó: lưu output phiên trước và khôi phục để tiếp tục.
Báo cáo phân tích vẫn cần viết sau khi có số liệu thật.
'''),
        nb.v4.new_markdown_cell('''## 0. Cấu hình duy nhất

Chạy từ đầu: giữ `PREVIOUS_OUTPUT_DIR = None`.
Dùng lại backbones/training đã chạy: thêm **một output mới nhất** vào Input rồi điền thư mục
`deepweeds-lab` ở biến này. Không cần cell copy riêng. Kết quả chỉ tái sử dụng khi config khớp;
đường dẫn dataset/profile/batch/epoch phải giữ như lần chạy trước.

`ALLOW_FINAL_TEST = True` xác nhận cho phép notebook tự khóa cấu hình bằng val và chạy test cuối.
Nếu không đồng ý, đổi False; runner sẽ dừng trước khi train các thí nghiệm chính thức.
Giữ working root mặc định để dùng lại output cũ. Không chạy song song hai full runner vào cùng output.
'''),
        nb.v4.new_code_cell('''ALLOW_FINAL_TEST = True
PREVIOUS_OUTPUT_DIR = None
# Ví dụ output đã chạy:
# PREVIOUS_OUTPUT_DIR = "/kaggle/input/notebooks/b22dckh063phmvnkin/track4-d2/deepweeds-lab"
MANUAL_IMAGES_DIR = None
MANUAL_LABELS_DIR = None
INSTALL_MISSING_DEPENDENCIES = True
SEEDS = (0, 1, 2)
''', metadata={'tags': ['configuration']}),
        nb.v4.new_code_cell(bootstrap, metadata={'tags': ['bootstrap']}),
        nb.v4.new_code_cell(setup),
        nb.v4.new_markdown_cell('''## 1. EDA và kiểm tra chia dữ liệu

Kiểm tra fold 0, giao rỗng, hợp 17.509 ảnh, file tồn tại và phân bố lớp.
Giữ CSV nguyên bản. Một ảnh có nhãn khác giữa labels.csv và train_subset0.csv đã được
nêu trong tài liệu; không tự sửa nhãn. Biểu đồ EDA lưu ở curves/_eda.
'''),
        *eda,
        nb.v4.new_code_cell('''subprocess.run(
    [sys.executable, '-X', 'utf8', str(ROOT / 'code/selftest.py')],
    check=True, cwd=ROOT
)
'''),
        nb.v4.new_markdown_cell('''## 2. Chạy toàn bộ thí nghiệm

B01–B05 và T00–T08 giữ kế hoạch cũ: 12 epoch, batch 32, seed 0 trên Kaggle.
T09 kết hợp augmentation và loss ngoài baseline tốt nhất theo val; đây là thí nghiệm
**tương tác**, không phải ablation một yếu tố, và không giả định kết hợp sẽ cải thiện.

Runner chọn training theo macro-F1 val. Sau inference, chọn macro-F1 val cao nhất,
hòa thì ECE thấp nhất rồi p95 thấp nhất; ghi selection.json **trước test**.
T00 và F01 chạy cùng seed 0/1/2. Lần chạy hoàn tất được tái sử dụng; checkpoint epoch
chưa hoàn tất được resume khi có last.pt. Resume không đảm bảo khớp từng bit do RNG/augmentation.
Nếu đã có selection.json, không chọn lại backbone/recipe/inference sau khi có test.

Trạng thái/lỗi lưu ở runs/full_status.json. Không bỏ qua lỗi hoặc ghi số liệu giả.
'''),
        nb.v4.new_code_cell('''workbook = run_all(BASE, allow_final_test=ALLOW_FINAL_TEST, seeds=SEEDS)
print('Workbook:', workbook)
STAGE = 'final'
''', metadata={'tags': ['full-run']}),
        nb.v4.new_markdown_cell('## 3. Đánh giá bằng eval.py nguyên bản'),
        nb.v4.new_code_cell(evaluation),
        nb.v4.new_markdown_cell('''## 4. Lấy output và hoàn thiện bài nộp

Output: `/kaggle/working/deepweeds-lab/`.
- results.xlsx: 7 sheet; source trỏ về log thật, mean/std mẫu qua seed.
- runs/: config, history, summary, best.pt, selection.json, combination.json.
- curves/, logits/, predictions/: kết quả backbone/training/final và dự đoán chuẩn.
- eval_out/: số liệu/điểm đề xuất từ eval.py gốc.

Độ trễ inference gồm tensor views + forward + aggregation, không gồm đọc/giải mã ảnh;
workflow hiện đo batch 1, FP32. Không xem là độ trễ end-to-end robot.
Notebook tự động chạy thí nghiệm, **không tự viết kết luận nghiên cứu hoặc đảm bảo đủ mọi
hạng mục báo cáo**. Bổ sung báo cáo, ma trận nhầm lẫn/phân tích lỗi, đo batch lớn nếu cần,
và bằng chứng loss ban đầu/overfit batch nhỏ theo GUIDE trước khi nộp.
Không commit dataset hoặc checkpoint lớn. Sau khi xem test, không thay cấu hình rồi chạy lại.

Nếu phiên bị ngắt: lưu những output còn lấy được, thêm output đó vào phiên mới, đặt
PREVIOUS_OUTPUT_DIR rồi Run All. Không thể khôi phục file đã mất nếu Kaggle chưa lưu output.
''')
    ]
    notebook = nb.v4.new_notebook(cells=cells, metadata=copy.deepcopy(original.metadata))
    nb.validate(notebook)
    for index, cell in enumerate(cells):
        if cell.cell_type == 'code':
            compile(cell.source, f'full-cell-{index}', 'exec')
    output = CODE / 'lab_day2_kaggle_full.ipynb'
    nb.write(notebook, output)
    print('Created:', output, '| cells:', len(cells), '| bundled files:', len(payload))
    return output


if __name__ == '__main__':
    build()
