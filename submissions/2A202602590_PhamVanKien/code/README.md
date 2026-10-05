# Bài làm DeepWeeds — cách chạy

## Kaggle Full: Run All một lần

Import **`lab_day2_kaggle_full.ipynb`**, thêm dataset ảnh và CSV, bật GPU + Internet,
rồi **Save & Run All**. Không cần đổi STAGE hoặc copy cell giữa các bước.
Bản này giữ riêng, không thay thế notebook chạy từng stage bên dưới.

- Mặc định `ALLOW_FINAL_TEST=True`: xác nhận cho phép tự khóa lựa chọn bằng val rồi
  đánh giá test ở cuối. Đổi False nếu chưa muốn cho phép chung kết; runner sẽ dừng.
- Chạy từ đầu: `PREVIOUS_OUTPUT_DIR=None`.
- Dùng lại kết quả cũ: thêm một output notebook mới nhất vào Input rồi đặt
  `PREVIOUS_OUTPUT_DIR` bằng đường dẫn thư mục `deepweeds-lab` của output đó.
  Tự copy runs/curves/logits/predictions; không ghi đè file xung đột và không dùng lại
  cache kiểm tra split/môi trường của phiên cũ. Chấp nhận output chưa chạy xong backbone.
- Profile Kaggle giữ batch 32, workers 2, 12 epoch, screening seed 0. Muốn tái sử dụng
  kết quả, giữ cùng đường dẫn dataset và Config; recipe đổi thì runner báo lỗi.
- Thứ tự: EDA/selftest → B01–B05 → T00–T08 → T09 kết hợp augmentation/loss chọn trên val
  → inference → selection.json → T00/F01 với seed 0/1/2 → results.xlsx và eval.py.
  T09 là thí nghiệm tương tác; không giả định kết hợp tốt hơn baseline.
- Kết quả hoàn tất được tái sử dụng. Run chưa hoàn tất có last.pt được resume từ epoch
  đã lưu; không đảm bảo tái lập từng bit do RNG/augmentation. Khi đã có selection.json,
  không chọn lại cấu hình dựa trên kết quả test; giữ nguyên các prediction test đã lưu.
- Trạng thái/lỗi: `runs/full_status.json`. Nếu phiên bị Kaggle ngắt, chỉ khôi phục được
  những file output còn lưu/lấy được, không phục hồi dữ liệu đã mất.

Output: `/kaggle/working/deepweeds-lab/`. Một lượt full có thể vượt thời gian phiên,
hạn mức GPU hoặc dung lượng Kaggle; notebook không bỏ qua những giới hạn này.
Vẫn cần viết báo cáo, phân tích lỗi/ma trận nhầm lẫn và các bằng chứng theo GUIDE.
Độ trễ workflow hiện đo FP32, batch 1; không gồm giải mã ảnh.

Tái tạo bản full sau khi sửa nguồn: `python -X utf8 code/build_full_notebook.py`.
Không cần chạy builder khi chỉ upload notebook đã có.

## Kaggle: chỉ upload notebook và dataset

Import **`lab_day2_kaggle.ipynb`**, thêm dataset vào Input, bật GPU và Internet rồi Run All.
Notebook này chứa sẵn toàn bộ module và `eval.py`; không cần upload các file `.py` hoặc clone repo.
Tự tìm ảnh và CSV trong `/kaggle/input/`, kể cả hai folder nằm ở hai Kaggle dataset riêng.
Mặc định chạy `setup`; đổi STAGE ở cell đầu để chạy các bước tiếp theo.
Output ở `/kaggle/working/deepweeds-lab/`. Giữ checkpoint/log khi chuyển sang phiên Kaggle mới.

`build_kaggle_notebook.py` tạo lại bản độc lập sau khi code nguồn hoặc notebook gốc thay đổi.

## Notebook local

Mở `lab_day2.ipynb` trong `code/` hoặc từ thư mục gốc repo. Notebook dùng code đã triển
khai, không import các stub trong `starter/`.

## Môi trường

Cài PyTorch và torchvision phù hợp GPU trước, rồi chạy từ thư mục gốc:

```powershell
python -m pip install -r code/requirements.txt
```

Notebook ghi phiên bản thực tế vào `runs/environment.json`; mỗi lần train ghi Config
và phiên bản Torch trong thư mục riêng. Dùng `PROFILE='local'` cho RTX 2050 4 GB:
batch 4, worker 0, AMP, không cache toàn bộ dataset. `kaggle` dùng batch 32, worker 2.
Đổi profile phải áp dụng cho toàn bộ nhóm so sánh và ghi rõ trong báo cáo.

Ảnh đặt ở `images/` hoặc `data/images/`; nhãn và fold 0 nguyên bản đặt ở `data/labels/`.
Notebook tự tìm ảnh và kiểm tra giao rỗng, số ảnh, file thiếu trước khi chạy.

## Thứ tự chạy notebook

1. `STAGE='setup'`: EDA, kiểm tra split, self-test. Mặc định không huấn luyện.
2. `STAGE='smoke'`: 1 epoch MobileNetV3 scratch, 2 ảnh/lớp train và val, ảnh 64×64.
   Output riêng ở `runs/_smoke/` và `curves/_smoke/`; không vào bảng chính thức.
3. `STAGE='backbones'`: 5 backbone, cùng recipe, 12 epoch, seed 0.
4. `STAGE='training'`: backbone thắng trên val; T00 và ablation khởi tạo, augmentation,
   loss, CutMix/Mixup. Mỗi lần chỉ đổi một yếu tố; so sánh với T00.
5. `STAGE='inference'`: 1 view, hflip/prob, five-crop/prob, hflip/logit, temperature.
   5 phương pháp, FP32, độ trễ batch 1: warmup 10 và 50 lần đo có đồng bộ CUDA.
6. `STAGE='final'`, `ALLOW_FINAL_TEST=True`: khóa cấu hình bằng val, chạy T00 và F01
   với seed 0/1/2, test một lần mỗi seed. File test có sẵn được giữ nguyên.
7. `STAGE='export'`: đọc log thật để xuất `results.xlsx` và chạy `eval.py` khi đã có
   dự đoán chung kết. Sheet thiếu kết quả ghi trạng thái thiếu; không tự điền số giả.

Sau mỗi bước, đổi STAGE rồi restart kernel và Run All. Các nhóm đã chạy xong được tái
sử dụng khi Config không đổi. Với cấu hình mới, dùng exp_id mới; với nghiên cứu mới,
dùng out_dir/pred_dir riêng. Không lấy kết quả test để điều chỉnh lựa chọn.

Lần đầu dùng pretrained cần Internet tải trọng số ImageNet. Notebook không tự tải
ảnh vì workspace hiện đã có dataset. Colab/Kaggle có thể tải theo README gốc rồi mở notebook.

## Chạy CLI và test

```powershell
python -X utf8 code/selftest.py
python -X utf8 -m unittest discover -s tests
python -X utf8 -m unittest discover -s code/tests -v
python -X utf8 code/lab_workflow.py smoke
python -X utf8 code/lab_workflow.py export
```

Một thí nghiệm riêng trên máy local, đầy đủ train/val:

```powershell
python -X utf8 code/train.py --set exp_id=B01 backbone=resnet50 images_dir=images batch_size=4 num_workers=0 cache_images=false pin_memory=false
```

Trên Windows, `runtime.py` chọn pandas Python string storage để tránh native crash
PyArrow sau khi Torch đã nạp. `-X utf8` xử lý tiếng Việt khi `eval.py` ghi JSON; giữ nguyên
`eval.py` theo yêu cầu bài lab.

## Output và giới hạn

- `runs/<exp_id>/seed<k>/`: config.json, history.csv, best.pt, summary.json.
- `logits/<exp_id>/`: logit val và test (chỉ khi được bật).
- `curves/`: đường cong loss/metric/LR; `_eda/` cho EDA, `_smoke/` cho kiểm tra.
- `predictions/`: CSV chuẩn cho T00, F01 và val của F01; bản uncal khi chọn temperature.
- `results.xlsx`: Summary, Backbones, Training, Inference, Final, PerClass, Latency.
  Có cột source truy về log/dự đoán; std dùng ddof=1, không điền 0 khi chỉ có một seed.

Độ trễ của workflow bao gồm transform tensor, forward và gộp view trên device; loại trừ
đọc/giải mã ảnh. Đây chưa phải độ trễ end-to-end của robot. Viết hạn chế này trong báo cáo.
Chạy thử chỉ xác nhận pipeline, không chứng minh accuracy. Bộ test gốc kiểm tra starter
và eval; bộ test bổ sung kiểm tra phần triển khai. Kết quả nghiên cứu chỉ có sau khi chạy
đủ các stage chính thức. Điền link notebook Colab/Kaggle của bạn khi chuẩn bị nộp bài.

`build_notebook.py` tái tạo notebook sạch output khi cần cập nhật cấu trúc cell.
Xem `VALIDATION.md` để biết các kiểm tra đã chạy và giới hạn của kết quả hiện tại.
