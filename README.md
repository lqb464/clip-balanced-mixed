# RDE + Balanced Mixed

Thử **Balanced Mixed trên đúng pipeline RDE của lần chạy old**, để so với **Identity K=4** đã đạt R@1 **63,80%**. Đây không phải CLIP + SDM hay ITSELF. Chưa có kết quả huấn luyện của Balanced Mixed trên pipeline này.

## Cấu hình giữ nguyên

Mốc đối chiếu là `configs.yaml` và code trong `results_training_old/tbps/rde/2024-CVPR-RDE`, lần chạy `20260922_061328_RDE_TAL+sr0.3_tau0.015_margin0.1_n0.0`.

| Thành phần | Cấu hình |
|---|---|
| Dataset | RSTPReid; 37.010 bản ghi train, 3.701 PID |
| Model | RDE, CLIP ViT-B/16, giữ hai nhánh BGE và TSE |
| Loss và cách huấn luyện | TAL + RBS; GMM và consensus division trước mỗi epoch, như code old |
| `loss_names` | `TAL+sr0.3_tau0.015_margin0.1_n0.0` |
| Batch huấn luyện | **64**, không phải 128 hoặc 256 |
| Epochs / horizon scheduler | **60 / 60** |
| `tau` / margin / select ratio | 0,015 / 0,1 / 0,3 |
| Noise nhân tạo | 0,0 |
| Ảnh / text | 384 × 128 / 77 tokens; stride 16 |
| Augmentation | Giữ nguyên cả image augmentation và text augmentation |
| Precision | `convert_weights` FP16 của RDE gốc; không thêm AMP, gradient cache hay gradient clipping |
| Optimizer | Adam; LR backbone 1e-5; betas 0,9 / 0,999; epsilon 1e-3 |
| LR đầu TSE | **0,001**, đúng solver old; bias backbone nhân LR 2 |
| Weight decay | 4e-5; bias 0 |
| Scheduler | Warmup tuyến tính 5 epochs, factor 0,1, rồi cosine về 0 |
| Seed | 1 |
| Workers / batch test cuối | 8 / 512 |
| Chọn checkpoint | BGE+TSE R@1 cao nhất trên **test split**, như old |
| Kết quả cuối | Đánh giá lại cả `best.pth` và `last.pth`, đủ BGE, TSE, BGE+TSE |

Các file model, loss, đầu TSE, tokenizer, augmentation trong dataset và solver được giữ nguyên từ old. Kiểm thử đối chiếu SHA-256 trong `rde/SOURCE_MANIFEST.json`, độc lập với kiểu xuống dòng Windows/Linux. Thay đổi phần tích hợp sampler, lưu kết quả và tiếp tục sau khi session kết thúc; không thay công thức huấn luyện.

**Lưu ý về đánh giá:** chọn checkpoint trên test không phải quy trình test giữ kín độc lập. Repo giữ chính sách này để đối chiếu đúng lần chạy old, và ghi rõ trong báo cáo. Một kết quả tốt hơn ở một seed chưa đủ kết luận sampler tốt hơn một cách ổn định.

## Phần thay đổi: Balanced Mixed

- Thay Identity K=4 bằng Balanced Mixed. Identity K=4 vẫn có thể chạy lại bằng cùng pipeline, không sửa sampler cũ.
- Với batch 64, mục tiêu **4 cặp positive cùng PID từ hai ảnh khác nhau**, chiếm 8 vị trí.
- 56 vị trí còn lại ưu tiên PID chưa có trong batch. Mục tiêu 14 hard negative, 14 semi-hard negative và 28 ngẫu nhiên.
- Hard là hạng cosine 1–10; semi-hard là hạng 11–64, không phải điều kiện margin triplet.
- Xét ảnh → caption, caption → ảnh và caption → caption; loại cùng PID.
- Danh sách negative được tính **chỉ từ train split**, dùng đặc trưng BGE của CLIP pretrained ban đầu trong code RDE, rồi giữ cố định. Không dùng test hay checkpoint old để mining. Cache phải khớp dataset, trọng số pretrained và checksum.
- Dùng chế độ lấy mỗi bản ghi một lần trong mỗi lượt duyệt dữ liệu, giữ batch cuối; không bật oversampling.
- RDE có hai lượt duyệt mỗi epoch: tính loss để fit GMM, rồi train. Balanced Mixed tạo thứ tự riêng, tái lập theo seed cho mỗi lượt; diagnostics ghi riêng `gmm_sampler` và `train_sampler`.
- Ràng buộc có thể nới và mining có thể fallback về ngẫu nhiên khi phần dữ liệu còn lại không đủ điều kiện; số lần được ghi nhận.

**Số mẫu mỗi epoch có thể khác:** Identity cũ chia nhóm K=4 và có thể bỏ phần dư; Balanced Mixed giữ đủ dữ liệu. Giữ cùng 60 epochs không đồng nghĩa cùng số bước cập nhật hay cùng thời gian. Đây là khác biệt của sampler, cần đọc cùng diagnostics khi diễn giải R@1.

## Notebook Kaggle

Dùng **`kaggle_train.ipynb`** trong repo này. Repo public, không cần `GH_TOKEN`. Bật Internet và GPU T4. Mặc định chỉ chạy Balanced Mixed trên GPU 0 để so với kết quả Identity đã có.

1. Cell đầu clone/cập nhật repo, cài dependency và chạy kiểm thử.
2. Cell cấu hình đặt đường dẫn benchmark và output **`/kaggle/working/rde_experiment`**.
3. Tạo hoặc kiểm tra cache negative từ train split.
4. Huấn luyện RDE + Balanced Mixed, đủ **60 epochs tổng cộng**.
5. So với kết quả old, báo chênh lệch R@1/R@5/R@10/mAP/mINP bằng điểm phần trăm.

Muốn chạy lại Identity song song trên T4 thứ hai, đặt `SAMPLERS = ['balanced_mixed', 'identity']`. Có thể thay `identity` bằng `random`. Mỗi GPU chạy một thí nghiệm riêng, không dùng DDP; mọi cấu hình RDE chung giữ như bảng trên.

**Không resume checkpoint CLIP + SDM trước đây, hoặc checkpoint Identity old sang Balanced Mixed.** Cần bắt đầu Balanced Mixed từ pretrained ban đầu. Khác sampler thì là một lượt train riêng.

### Khi session gần hết thời gian

- Mặc định dừng có kiểm soát ở **cuối epoch** khi đã chạy khoảng 9 giờ; không dừng giữa epoch. Thời gian có thể vượt ngưỡng thêm một epoch và thời gian ghi checkpoint.
- Lưu `resume.pth` mỗi epoch, gồm model, optimizer, scheduler, RNG, best score và lịch sử. Giữ cả `best.pth`; `last.pth` chỉ được xuất khi đủ 60 epochs.
- Nếu session bị ngắt giữa epoch, khi resume phải chạy lại epoch chưa hoàn tất. Không resume giữa batch vì RDE còn phụ thuộc nhãn GMM của epoch đó.
- Dùng **Save Version** để giữ output notebook trước khi session mất. Ở session tiếp theo, gắn output bằng Add Input và đặt `RESUME_FROM` tới thư mục `rde_experiment` chứa `cache/` và `runs/`.
- Chạy lại các cell. Code kiểm tra cấu hình, trọng số, annotation và cache; tiếp tục từ epoch kế tiếp, không cộng thêm 60.
- Không thể bảo đảm toàn bộ 60 epochs trong một session 12 giờ. Balanced Mixed giữ nhiều bản ghi hơn Identity old, nên có thể cần nhiều session. Cache/clone/cài thư viện không nằm trong ngân sách train 9 giờ.

## Kết quả old được dùng làm mốc

| Checkpoint | Nhánh | R@1 | R@5 | R@10 | mAP | mINP |
|---|---|---:|---:|---:|---:|---:|
| best | BGE | 61,10 | 80,75 | 86,50 | 49,50 | 27,72 |
| best | TSE | 61,70 | 80,65 | 87,65 | 50,04 | 28,19 |
| **best** | **BGE+TSE** | **63,80** | **81,55** | **87,60** | **51,47** | **29,53** |
| last | BGE+TSE | 63,55 | 81,55 | 87,30 | 51,87 | 30,73 |

Đây là số liệu **đánh giá lại checkpoint** trong log old, không phải số liệu được suy ra từ paper. Best R@1 trong lúc train là 63,75 tại epoch 33, khác lần nạp lại `best.pth` đạt 63,80. `compare.py` dùng 63,80 làm mốc best và 63,55 làm mốc last, không trộn hai checkpoint.

## Lệnh tương đương

```bash
pip install -r requirements.txt
CUDA_VISIBLE_DEVICES=0 python build_cache.py --root /path/to/benchmark
python launch.py --root /path/to/benchmark --samplers balanced_mixed --gpus 0 --max-hours 9
python compare.py --runs runs --seed 1
```

## File đầu ra

- `config.json`, `configs.yaml`: cấu hình thực chạy và dấu kiểm tra.
- `console.log`, `train_log.txt`: log từng epoch, bảng retrieval của RDE.
- `metrics.json`: BGE/TSE loss, retrieval từng epoch, diagnostics của hai lượt sampler.
- `progress.json`: epoch hoàn tất, best score và epoch được chọn.
- `resume.pth`: tiếp tục từ cuối epoch; `best.pth`: checkpoint được chọn; `last.pth`: model ở epoch 60.
- `best_test.json`, `last_test.json`, `result.json`: kết quả sau đủ 60 epochs.
- `comparison_seed1.json`: so Balanced Mixed với Identity old cho từng nhánh và checkpoint.

Không đưa dataset, cache hay checkpoint lên GitHub. Kiểm thử CPU không thay thế một lượt chạy GPU đầy đủ. Xem `VALIDATION.md` và `THIRD_PARTY.md`.
