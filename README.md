# CLIP Balanced Mixed

Thí nghiệm fine-tune CLIP ViT-B/16 + SDM trên RSTPReid trong **60 epochs**, so sánh **Random** và **Balanced Mixed**. Repo này chuẩn bị thí nghiệm; chưa có kết quả train 60 epochs.

## Cấu hình chung

| Thành phần | Cấu hình |
|---|---|
| Backbone | OpenAI CLIP ViT-B/16 pretrained, fine-tune cả hai encoder |
| Loss | SDM theo IRRA, temperature cố định 0,02 |
| Đầu phụ | Không dùng MLM, ID head hoặc adapter |
| Kích thước ảnh | 384 × 128, text length 77 |
| Batch sampler và batch loss | **128** cho cả hai sampler |
| Nhóm xử lý encoder | 16 mẫu/lượt, dùng gradient cache và AMP |
| Optimizer | AdamW, LR 1e-5, weight decay 4e-5, betas 0,9 / 0,999 |
| LR schedule | Warmup 5 epochs, sau đó cosine; horizon luôn là 60 |
| Seed ban đầu | 1, dùng chung cho hai lượt train |
| Dữ liệu mỗi epoch | Lấy mỗi bản ghi một lần, giữ batch cuối |
| Chọn checkpoint | R@1 cao nhất trên validation |
| Test cuối cùng | Checkpoint được chọn trên validation, sau đủ 60 epochs |

Gradient cache tính feature cho đủ batch 128, tính **một loss chung trên ma trận 128 × 128**, rồi tính lại encoder theo từng nhóm để truyền gradient. Đây không phải cộng dồn các loss riêng của batch 16. CLIP ViT sử dụng ở đây không có dropout hoặc batch normalization. Test tự động kiểm tra gradient với batch đầy đủ trên model nhỏ; bộ nhớ và tốc độ thực tế cần xác nhận trên T4.

Hai T4 chạy **hai tiến trình độc lập**: GPU 0 chạy Random, GPU 1 chạy Balanced Mixed. Mỗi sampler giữ nguyên cấu trúc batch 128; không dùng DDP. Không thể khẳng định cả 60 epochs sẽ xong trong một session 12 giờ trước khi đo tốc độ thực tế.

## Balanced Mixed

- Mục tiêu 4 cặp cùng PID từ hai ảnh khác nhau: 8 vị trí trong batch.
- 120 vị trí còn lại ưu tiên PID chưa xuất hiện: mục tiêu 30 hard, 30 semi-hard, 60 ngẫu nhiên.
- Hard nằm trong hạng cosine 1–10; semi-hard trong hạng 11–64. Đây là thứ hạng, không phải điều kiện margin của triplet loss.
- Xét ảnh → caption, caption → ảnh, caption → caption. Loại ứng viên cùng PID và ưu tiên mẫu neo chưa được dùng để mining.
- Danh sách negative được tính bằng **CLIP pretrained ban đầu, chỉ trên train split**, rồi giữ cố định. Hai model train đều khởi tạo từ chính pretrained này. Cache từ một encoder đã fine-tune khác sẽ bị từ chối, tránh lẫn điều kiện giữa các lượt chạy.
- Khi không có ứng viên phù hợp, lấy ngẫu nhiên thay thế và ghi diagnostics. Không cho chạy Balanced Mixed thiếu cache rồi âm thầm biến thành phiên bản không mining.
- Cấu hình thí nghiệm dùng coverage: không oversample. Module sampler vẫn giữ chức năng oversample của mã nguồn gốc, nhưng không bật trong thí nghiệm này vì RSTPReid đã cân bằng PID.

Đây là thí nghiệm mới dùng cache từ CLIP pretrained. Các tỷ lệ top-10 trong tài liệu audit trước đó sử dụng checkpoint IRRA khác, nên không phải kết quả đo của repo này.

## Chạy trên Kaggle

Mở **`kaggle_train.ipynb`**, bật Internet và GPU T4 ×2. Gắn bộ benchmark RSTPReid, rồi chạy các cell theo thứ tự.

Repo private tại `https://github.com/lqb464/clip-balanced-mixed`. Trong Kaggle Add-ons → Secrets, thêm `GH_TOKEN` có quyền đọc repo này và bật quyền cho notebook. Token chỉ dùng trong môi trường của tiến trình clone, không ghi vào file hay URL remote.

Notebook mặc định dataset ở:

```text
/kaggle/input/datasets/hoanggv/tbps-benchmark/benchmark
```

Đầu ra nằm tại `/kaggle/working/clip_experiment`, gồm cache và hai thư mục kết quả. Không đưa dataset, cache hoặc checkpoint lên GitHub.

### Tiếp tục sau khi session kết thúc

Mặc định mỗi tiến trình dừng có kiểm soát sau tối đa 9 giờ kể từ lúc khởi chạy train và lưu checkpoint. Thời gian này chưa gồm clone/cài thư viện/tạo cache; vẫn cần theo dõi thời gian session Kaggle. Một batch đang chạy hoặc lần lưu checkpoint có thể làm thời gian dừng vượt ngưỡng một chút.

1. Dùng Save Version để giữ output của notebook trước khi mất session. File trong `/kaggle/working` chỉ giúp phục hồi khi output đã được giữ lại.
2. Ở session mới, gắn output lần trước bằng Add Input.
3. Trong cell cấu hình, đặt `RESUME_FROM` thành đường dẫn đến thư mục **clip_experiment** của output đó. Thư mục phải chứa cả `cache/` và `runs/`; giữ cả `last.pt` lẫn `best.pt`.
4. Chạy lại các cell. Code kiểm tra cache, tự tiếp tục từ `last.pt` và dừng ở **60 epochs tổng cộng**, không cộng thêm 60.

Checkpoint lưu mỗi 50 batch và cuối epoch, gồm optimizer, AMP scaler, trạng thái RNG, epoch và batch tiếp theo. Nếu bị ngắt đột ngột, phần sau checkpoint gần nhất phải chạy lại. Thứ tự sampler và augmentation được tái tạo theo seed/epoch/index. Kết quả số thực trên GPU có thể chênh rất nhỏ do kernel phần cứng.

### Chạy bằng lệnh

```bash
pip install -r requirements.txt
CUDA_VISIBLE_DEVICES=0 python build_cache.py --root /path/to/benchmark
python launch.py --root /path/to/benchmark --epochs 60 --max-hours 9
```

Chỉ train Balanced Mixed trên một GPU:

```bash
python launch.py --root /path/to/benchmark --samplers balanced_mixed --gpus 0 --epochs 60
```

Để giảm bộ nhớ, chọn `--microbatch 8` ngay từ đầu cho cả hai lượt chạy. Batch loss vẫn là 128. Không đổi cấu hình một lượt đang resume.

## Đọc kết quả

- `config.json`: cấu hình, dấu kiểm tra dataset, pretrained và cache.
- `console.log`: tiến trình từng batch.
- `progress.json`: epoch hoàn tất, batch tiếp theo, best validation R@1.
- `metrics.json`: loss, positive coverage, PID diversity, sampler diagnostics và validation từng epoch. Thời gian epoch là phần xử lý trong session hiện tại nếu epoch bị chia qua nhiều session.
- `last.pt`: trạng thái để tiếp tục; `best.pt`: model có validation R@1 cao nhất.
- `result.json`: test R@1/R@5/R@10, mAP, mINP sau khi hoàn tất.
- `comparison_seed1.json`: so sánh hai sampler, chênh lệch tính bằng **điểm phần trăm**.

```bash
python compare.py --runs runs --seed 1
```

Phép so sánh chỉ thực hiện khi cả hai lượt đã hoàn tất và các cấu hình chung khớp nhau. Gallery chứa mỗi ảnh một lần; một query có thể có nhiều ảnh đúng theo PID. Một training seed cho kết quả thử nghiệm ban đầu; để kết luận ổn định, cần lặp nhiều training seed. Năm sampling seed trong audit trước đây không thay thế năm training seed.

Hai lượt train đều dùng SDM và temperature cố định 0,02.

## Kiểm tra

```bash
python -m unittest discover -s tests -v
```

Kiểm tra gradient cache, retrieval có gallery ảnh duy nhất, cache loại cùng PID, coverage và tái tạo thứ tự sampler khi resume. Không thay thế một lượt chạy GPU đầy đủ.

## Nguồn mã

CLIP backbone/tokenizer, từ vựng BPE và Balanced Mixed được tách từ repo `HoangVo-Prog/tbps-sampler` hiện có trên máy. Xem `THIRD_PARTY.md` và `LICENSE`. Đây là CLIP + SDM để đánh giá sampler, không phải tái lập toàn bộ IRRA và không tuyên bố bằng benchmark IRRA công bố.
