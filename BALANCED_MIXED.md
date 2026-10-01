# Balanced Mixed cho bốn phương pháp

Mã nguồn được dựng từ commit `0dd53de08a1163b692c76866d1adfcd0dd12e39e` của `HoangVo-Prog/tbps-sampler`. Mỗi thư mục phương pháp giữ model, loss, optimizer và train loop riêng. DataLoader dùng Balanced Mixed khi truyền `--sampler balanced_mixed`.

## Dataset và batch size

Cả bốn lệnh dùng RSTPReid và 60 epochs. Batch size theo cấu hình launch hoặc mặc định parser tương ứng:

| Phương pháp | Batch | Cấu hình phương pháp được giữ |
|---|---:|---|
| IRRA | 64 | `sdm+mlm+id`, image augmentation, MLM |
| DM-Adapter | 128 | `sdm+aux`, image augmentation, MLM, 6 experts, top-k 2, reduction 8, LR `3e-4` |
| RDE | 64 | TAL/RBS, image và text augmentation, noise rate 0, `tau=0.015`, margin `0.1`, selection ratio `0.3` |
| ITSELF | 256 | `tal+cid`, `only_global`; batch 256 là mặc định parser được script upstream sử dụng |

Sampler đặt 4 cặp positive mỗi batch đầy đủ, 25% hard negative, 25% semi-hard negative và phần còn lại lấy ngẫu nhiên. Các dải cosine toàn cục là hạng 1–10 và 11–64; candidate cùng PID bị loại. Chế độ coverage lấy mỗi dòng train một lần mỗi epoch và giữ batch cuối chưa đủ kích thước. Không bật oversampling PID hiếm.

## Cập nhật hard negative

Ở đầu epoch đầu, sampler mã hóa các ảnh train duy nhất và caption bằng model hiện tại của phương pháp đang chạy, rồi tạo danh sách lân cận toàn cục cho ảnh→caption, caption→ảnh và caption→caption. Danh sách được cập nhật mỗi ba epochs bằng trọng số model đang nằm trong bộ nhớ. Không dùng validation hoặc test split. Với RDE, cùng thứ tự dòng được dùng cho lượt ước lượng loss và lượt train trong một epoch.

Mỗi lần cập nhật cần thêm một lượt trích xuất đặc trưng từ training split và tính cosine toàn cục. Tần suất cấu hình bằng `--sampler-refresh-every`; đặt `1` để cập nhật mỗi epoch. Các tham số này thuộc sampler; cấu hình loss/model/optimizer riêng của bốn phương pháp được giữ lại.

## Kaggle

Mở `kaggle_balanced_mixed.ipynb`, gắn dataset benchmark RSTPReid và bật GPU. Notebook clone repo public này, cài dependency mà không thay PyTorch Kaggle, và dùng dataset root `/kaggle/input/datasets/hoanggv/tbps-benchmark/benchmark`.

Cell IRRA được mở sẵn. Cell DM-Adapter, RDE và ITSELF chứa đầy đủ lệnh nhưng đang comment theo thứ tự yêu cầu. Bỏ comment cell của phương pháp cần chạy. Mỗi phương pháp lưu log/checkpoint riêng và là một thí nghiệm riêng; chạy cả bốn có thể vượt thời lượng một Kaggle session.
