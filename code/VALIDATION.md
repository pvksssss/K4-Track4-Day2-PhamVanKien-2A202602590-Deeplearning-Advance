# Kiểm tra triển khai — 2026-10-03

Phạm vi: hoàn thiện notebook dùng các module `code/`, cấu hình máy local, xử lý lỗi
Windows và bổ sung luồng thí nghiệm/xuất bảng. Chưa chạy nghiên cứu chính thức.

## Kết quả đã kiểm tra

- `python -X utf8 -m unittest discover -s tests`: **38/38 đạt**.
- `python -X utf8 code/selftest.py`: **12/12 đạt**, không còn native crash PyArrow.
- `python -X utf8 -m unittest discover -s code/tests -v`: **14/14 đạt** sau khi thêm bản Kaggle độc lập.
- Notebook hợp lệ theo nbformat, các cell code biên dịch được, không còn cell TODO.
- Đã thực thi notebook stage `setup` bằng nbclient; output kiểm tra được lưu tại
  `runs/_smoke/notebook_setup.ipynb`, không nằm trong notebook nguồn nộp bài.
- Đã kiểm tra ảnh đường cong SMOKE: loss, metric và lịch LR hiển thị được.
- Review độc lập đã kiểm tra cấu hình tái sử dụng, val/test, TTA/độ trễ và phục hồi
  sau gián đoạn. Các lỗi quan trọng được sửa và review lại.
- `lab_day2_kaggle.ipynb` chứa sẵn 15 file code/tài liệu, giữ nguyên byte của eval.py;
  test xác nhận ảnh và CSV ở hai dataset riêng vẫn được tìm đúng.
- Đã chạy thành công stage setup của bản độc lập từ thư mục không có repo, dùng dataset
  local làm Input kiểm tra; 0 output lỗi. Bản thực thi nằm ở `runs/_kaggle_check/setup_executed.ipynb`.
  Chưa điều khiển hoặc thực thi trực tiếp trên dịch vụ Kaggle.

## Dữ liệu và GPU

Fold 0: train **10.501**, val **3.501**, test **3.507**, hợp **17.509**;
giao các cặp rỗng và không thiếu ảnh. CSV nguyên bản không thay đổi.
Hợp các split cho Chinee Apple 1.126 và Lantana 1.063. Kiểm tra dữ liệu chi tiết sau đó
phát hiện ảnh `20170714-110407-3.jpg` có Label 0 trong `train_subset0.csv`, nhưng Label 1
(Lantana) trong `labels.csv`. Nhãn master cho số đếm 1.125 và 1.064, khớp bảng README.
Không sửa CSV; ghi nhận sự khác biệt nhãn này trong báo cáo và giữ split theo yêu cầu đề bài.
Audit đầy đủ ở `runs/data_audit.json`: 17.509 ảnh RGB 256×256 giải mã thành công, không ảnh
hỏng; MD5 archive khớp README.

GPU: NVIDIA GeForce RTX 2050 4 GB. Torch: **2.12.0+cu126**.

Lệnh kiểm tra thực tế:

```powershell
python -X utf8 code/lab_workflow.py smoke
```

MobileNetV3 scratch, seed 0, một epoch, batch 2, ảnh 64×64, 18 ảnh train và 18 ảnh val.
Lượt chạy cuối mất khoảng **34,5 giây/epoch**, lưu được best.pt, history.csv, summary.json,
logit và CSV val. **Không chạy test**, không tạo CSV test; không lấy kết quả chạy thử
để đánh giá chất lượng mô hình hoặc ghi vào bảng nghiên cứu.

Output kiểm tra: `runs/_smoke/` và `curves/_smoke/`. Cấu hình có `debug_run=true`.

## Những lỗi đã sửa

- Pandas string storage PyArrow gây access violation trên Windows sau khi Torch nạp:
  chọn Python string storage trong `runtime.py`.
- Cache ảnh nhầm kích thước transform với kích thước ảnh gốc: giữ nguyên pixel gốc.
- Five-crop gọi API center_crop không tồn tại trong torch.nn.functional.
- Tensor đếm MAC nằm trên CPU khi model nằm trên CUDA; bộ đếm Linear bỏ sót số token.
- Checkpoint khôi phục thiếu tag trọng số gốc; scratch bị ghi nhầm tag pretrained.
- AMP bỏ qua optimizer khi gradient không hữu hạn nhưng scheduler/EMA vẫn tiến bước.
- Chọn cấu hình từ kết quả cũ mà không đối chiếu config; hiện báo lỗi khi recipe đổi.
- Dự đoán TTA và đo độ trễ dùng hai backend gộp khác nhau; hiện dùng chung hàm trên device.
- Có file dự đoán test nhưng thiếu summary khi phiên ngắt; hiện phục hồi từ file đã lưu
  và metadata, không chạy lại test. Lưu raw logits để khôi phục bản uncal chính xác.

## Giới hạn

Chưa chạy 5 backbone × 12 epoch, ablation, so sánh suy luận trên toàn bộ val, hoặc chung kết
3 seed. Vì vậy chưa có results.xlsx nghiên cứu hoặc báo cáo chất lượng cuối cùng.
Exporter đã được kiểm tra với fixture riêng: đủ 7 sheet, bỏ debug run và std mẫu ddof=1;
fixture không được dùng như kết quả nghiên cứu. Chạy stage `export` sau khi có log thật.

Đã giữ nguyên `eval.py`, `starter/`, bộ test gốc và tài liệu đề bài.
