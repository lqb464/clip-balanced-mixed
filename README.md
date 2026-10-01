# Balanced Mixed cho TBPS

Repo giữ cấu trúc mã nguồn IRRA, DM-Adapter, RDE và ITSELF từ [HoangVo-Prog/tbps-sampler](https://github.com/HoangVo-Prog/tbps-sampler), đồng thời tích hợp Balanced Mixed vào DataLoader của từng phương pháp.

## Chạy trên Kaggle

Mở `kaggle_balanced_mixed.ipynb`, gắn dataset benchmark RSTPReid và bật GPU. Notebook clone repo public này, cài các dependency nhưng giữ nguyên PyTorch Kaggle, và đặt dataset root thành `/kaggle/input/datasets/hoanggv/tbps-benchmark/benchmark`.

Notebook có cell theo thứ tự IRRA, DM-Adapter, RDE, ITSELF. Mặc định chỉ mở cell IRRA; ba cell còn lại có đủ lệnh nhưng đang comment. Bỏ comment ở phương pháp muốn chạy. Mỗi phương pháp là một lượt train riêng 60 epochs và giữ model, loss, optimizer, batch size riêng của nó.

## Balanced Mixed

Mỗi batch đầy đủ dành bốn cặp positive cùng PID, lấy từ các ảnh khác nhau. Các vị trí còn lại nhắm tới 25% hard negative, 25% semi-hard negative và 50% random negative. Hard và semi-hard dùng hạng cosine toàn cục 1–10 và 11–64; candidate cùng PID bị loại. Chế độ coverage lấy mỗi bản ghi train một lần và giữ batch cuối chưa đủ kích thước. Không bật oversampling PID hiếm.

Danh sách mining được tính bằng model hiện tại của phương pháp đang chạy ở epoch đầu và cập nhật mỗi ba epochs. Chỉ dùng training split; xét ảnh→caption, caption→ảnh và caption→caption. Đặt `--sampler-refresh-every 1` trong lệnh chạy để cập nhật mỗi epoch. Mỗi lần cập nhật cần mã hóa thêm dữ liệu train và tính cosine toàn cục.

Xem [BALANCED_MIXED.md](BALANCED_MIXED.md) và [THIRD_PARTY.md](THIRD_PARTY.md) để biết cấu hình và nguồn mã. Giữ các file `LICENSE` riêng của từng phương pháp.
