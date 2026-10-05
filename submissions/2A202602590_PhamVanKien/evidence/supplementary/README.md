# Bằng chứng đo bổ sung

Các JSON ở đây được tạo sau thực nghiệm Kaggle, ngày 05/10/2026, trên RTX 2050. Không dùng kết quả bổ sung để thay lựa chọn cấu hình đã khóa bằng val.

- `latency_local.json`: batch1, các checkpoint B01–B05 và F01 seed0, warmup10,100 lần đo.
- `latency_batch32.json`: batch32, T09 seed0 với I00–I04 và F01 seed0, warmup10,50 lần đo. Thông lượng=32000/mean batch ms.
- `pipeline_posthoc.json`:18 ảnh train, kiểm tra CE khởi tạo và overfit classifier trên đặc trưng backbone đóng băng. Không phải overfit toàn pipeline trước thực nghiệm gốc.
- Các script Python là bản lưu của phép đo đã thực thi từ `runs/` trong workspace gốc. Muốn chạy lại, đặt script trong `runs/` của workspace, giữ `results.zip`, dataset `images/`, `data/labels/` và thư mục `submissions/2A202602590_PhamVanKien/`. Checkpoint không nằm trong gói nộp nhẹ.

Latency dùng eval/inference_mode, FP32, CUDA đồng bộ; đầu vào tensor ngẫu nhiên, không đọc dữ liệu test. Phạm vi gồm tensor views + forward + aggregation; không gồm đọc/giải mã ảnh hoặc resize/normalize đầu vào. Phiên bản môi trường lưu ở trường `local_versions` của từng JSON.

- `calibration_temperature_lock.json`/`calibration_posthoc.json`: T khớp F01val và kết quả trước/sau, không forward test lại.
- `pipeline_full_posthoc.json`/`pipeline_full_posthoc_history.csv`: overfit toàn mạng9ảnhtrain bằng vòngtrain dự án.
- `complete_posthoc_checks.py`: script hai kiểm tra mới, đặt tại `runs/` workspace gốc để chạy lại.
