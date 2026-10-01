# Mã nguồn bên thứ ba

Mã nguồn IRRA, DM-Adapter, RDE và ITSELF cùng license gốc được chép từ [HoangVo-Prog/tbps-sampler](https://github.com/HoangVo-Prog/tbps-sampler), commit `0dd53de08a1163b692c76866d1adfcd0dd12e39e`. Giữ các file `LICENSE` riêng từng phương pháp khi phân phối lại.

Các file root `sampler.py`, `sampler_bridge.py` và `dynamic_mining.py` tích hợp thí nghiệm Balanced Mixed. Model, loss và optimizer theo từng phương pháp giữ từ repo nguồn; phần nạp batch và khởi tạo epoch được mở rộng để dùng Balanced Mixed.
