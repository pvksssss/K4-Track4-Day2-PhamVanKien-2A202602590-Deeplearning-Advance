# Phạm Văn Kiên — 2A202602590 — DeepWeeds Lab Day2

Notebook Kaggle: [https://www.kaggle.com/code/b22dckh063phmvnkin/track4-d2](https://www.kaggle.com/code/b22dckh063phmvnkin/track4-d2) (link do sinh viên cung cấp).

Kết quả final: accuracy98.0325% ± 0.0855 điểm phần trăm, macro-F10.97528 ± 0.00094, mean±sample std qua seed0/1/2. Phân tích và hạn chế nằm trong [report.md](report.md); [results.xlsx](results.xlsx) gồm đủ7sheet. Gói này giữ source và dự đoán đã chạy trong results.zip, không huấn luyện hoặc forward test lại để thay kết quả.

## Nội dung nộp

- `code/`: source thực thi và notebook Kaggle độc lập nhúng source đúng bản đã chạy.
- `predictions/`:26CSVgốc; final/mốc mỗi nhóm3seed test và val.
- `curves/`:20curve huấn luyện, EDA, ma trận nhầm lẫn, F1 lớp, ảnh lỗi và kiểm tra bổ sung.
- `evidence/`:config/history/summary/logits gốc; metrics tính lại; audit dữ liệu; latency và kiểm tra post-hoc local.
- `eval.py`: bản đề bài nguyên gốc; `eval_out/`:kết quả score/grade tính lại từ CSV.
- `manifest.sha256`:checksum mọi file nộp trừ chính manifest.

## Chạy trên Kaggle

Upload **chỉ `code/lab_day2_kaggle_full.ipynb`**, gắn dataset ảnh+CSV ở Input, bật GPU+Internet. Notebook này chạy lần lượt các bước tự động, không cần đổi STAGE. Các notebook kiểu STAGE trong repo gốc là lựa chọn khác; không cần chạy cả hai.

```python
MANUAL_IMAGES_DIR = '/kaggle/input/datasets/b22dckh063phmvnkin/lab02-track4'
MANUAL_LABELS_DIR = '/kaggle/input/datasets/b22dckh063phmvnkin/label-t4'
INSTALL_MISSING_DEPENDENCIES = True
SEEDS = (0, 1, 2)
PREVIOUS_OUTPUT_DIR = None
ALLOW_FINAL_TEST = True
```

Đường dẫn phải đặt trong dấu nháy. Ảnh phải nằm ở thư mục chứa trực tiếp JPG và labels chứa labels.csv/train_subset0.csv/val_subset0.csv/test_subset0.csv. Lựa chọn được khóa bằng val rồi mới chạy test. ALLOW_FINAL_TEST cho phép bước cuối; sau khi xem test không chỉnh recipe chọn điểm cao hơn.

Nếu phiên bị ngắt, thêm Output phiên mới nhất vào Input và đặt `PREVIOUS_OUTPUT_DIR` tới thư mục `deepweeds-lab` có runs/checkpoint/predictions; giữ nguyên cấu hình/dataset. Save & Run tạo phiên mới, working không tự được chuyển sang phiên sau. Không chỉ khôi phục predictions hoặc xlsx vì training/inference cần checkpoint. Các checkpoint20file được giữ trong results.zip gốc/KaggleOutput; gói nộp nhẹ không chứa dataset/checkpoint. Giữ notebook/output version đã tạo results.zip để làm minh chứng vì link Kaggle có thể trỏ phiên bản mới sau này.

Môi trường Kaggle gốc nằm tại `evidence/runs/environment.json`; huấn luyện trên TeslaT4 FP16AMP, eval FP32. Đo bổ sung trên NVIDIA RTX2050 ghi riêng `evidence/supplementary/latency_local.json`, warmup10/n100, không chạy lại test. Số độ trễ bỏ qua decode/resize/normalize ban đầu; images/s=1000/p50 batch1, batch32 đo bổ sung riêng trên RTX2050 trong `evidence/supplementary/latency_batch32.json` (32.000/mean batch ms).

## Đánh giá độc lập CSV

Chạy từ thư mục gói đã giải nén; `data/labels/` dưới đây là vị trí CSV của đề bài, có thể thay bằng đường dẫn thực:

```bash
python -X utf8 eval.py score --pred "predictions/F01_seed*_test.csv" --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --tag F01 --out eval_out
python -X utf8 eval.py score --pred "predictions/T00_seed*_test.csv" --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --tag T00 --out eval_out
python -X utf8 eval.py grade --final "predictions/F01_seed*_test.csv" --baseline "predictions/T00_seed*_test.csv" --final-val "predictions/F01_seed*_val.csv" --latency-p95-ms 22.40820499992 --latency-method proper --test-csv data/labels/test_subset0.csv --val-csv data/labels/val_subset0.csv --labels data/labels/labels.csv --out eval_out
```

Score/grade chỉ đọc dự đoán, không chạy lại mô hình test. Grade phầnI18/19 ý chấm được, tối đa20; F01gốc thiếu test trước/sau hiệu chuẩn; phần bổ sung F01_TS cho19/20, tính hợp lệ hậu nghiệm do giảng viên quyết định. Đây không phải tổng điểm bài. Không coi kiểm tra classifier post-hoc là overfit toàn pipeline trước thực nghiệm.

## Hai kiểm tra bổ sung sau thực nghiệm

F01 gốc và26CSVgốc giữ nguyên. `predictions/posthoc/` có6CSV F01_TS được tạo từ xác suất đã lưu, T khớp riêng val cho mỗi seed. ECEtest giảm0,08477→0,00571; không đổi accuracy/F1/nhãn. `eval_out/posthoc_calibration/grade_I.json` chấm số học19/20; đây là bổ sung sau khi đã xem test, không tự động bảo đảm giảng viên nhận điểmI4a.

Kiểm tra toàn mạng dùng9ảnhtrain cố định và vòngtrain thật; log/config/biểu đồ tại `evidence/supplementary/pipeline_full_posthoc*` và `curves/supplementary/pipeline_full_posthoc.png`. Được thực hiện sau thí nghiệm, không ghi là kiểm tra trước train.

Lệnh đánh giá phần bổ sung, từ thư mục gói giải nén:

```bash
python -X utf8 eval.py score --pred "predictions/posthoc/F01_TS_seed*_test.csv" --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --tag F01_TS --out eval_out/posthoc_calibration
python -X utf8 eval.py grade --final "predictions/posthoc/F01_TS_seed*_test.csv" --baseline "predictions/T00_seed*_test.csv" --uncal "predictions/F01_seed*_test.csv" --final-val "predictions/posthoc/F01_TS_seed*_val.csv" --latency-p95-ms 32.24409999984 --latency-method proper --test-csv data/labels/test_subset0.csv --val-csv data/labels/val_subset0.csv --labels data/labels/labels.csv --out eval_out/posthoc_calibration
```

Tham số độ trễ tronggrade là phép đo trực tiếp F01_TS seed0, p95RTX2050=32.24ms; warmup10/n100/CUDA đồng bộ, FP32/batch1. Xem báo cáo mục6.1–6.2 để phân biệt kết quả gốc và bổ sung.
