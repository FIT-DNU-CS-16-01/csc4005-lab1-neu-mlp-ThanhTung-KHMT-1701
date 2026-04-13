[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/LiEKMyvY)
# CSC4005 Lab 1 – NEU MLP Starter

Starter kit này dùng cho **Lab 1 – Training & Regularization** của CSC4005. Repo **không chứa thư mục `data/`**. Sinh viên chạy trực tiếp với đường dẫn tới bộ dữ liệu ngoài repo `NEU-CLS.zip`.

Nội dung lab: so sánh ít nhất 3 cấu hình, xem learning curves, tránh dùng test để chọn mô hình, và kết luận cấu hình tốt nhất dựa trên validation. Case study là NEU Surface Defect Database với 6 lớp lỗi bề mặt thép.

## 1. Cấu trúc repo
```text
csc4005_lab1_neu_mlp_starter/
├── .github/workflows/validate-lab1.yml
├── README.md
├── REPORT_TEMPLATE.md
├── requirements.txt
├── configs/
│   └── baseline.json
├── docs/
│   └── LAB_GUIDE_LAB1.md
│   └── WANDB_GUIDE.md
├── notebooks/
├── outputs/
├── ci/
│   ├── check_structure.py
│   └── smoke_train.py
└── src/
    ├── __init__.py
    ├── dataset.py
    ├── model.py
    ├── train.py
    └── utils.py
```

## 2. Cài đặt
```bash
conda activate csc4005-dl
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Dữ liệu
Repo này **không có thư mục `data/`**. Bạn phải cung cấp đường dẫn dữ liệu khi chạy.

Hỗ trợ 2 kiểu dữ liệu:
- **Kiểu A:** thư mục lớp riêng `Crazing/`, `Inclusion/`, ...
- **Kiểu B:** file ZIP hoặc thư mục phẳng có tên ảnh kiểu `crazing_10.jpg`, `rolled-in_scale_21.jpg`, ...

Ví dụ với đúng bộ dữ liệu do giảng viên cung cấp:
```bash
python -m src.train --data_dir /duong_dan/NEU-CLS.zip --run_name quick_test
```

## 4. Chạy baseline chuẩn của Lab 1
```bash
python -m src.train   --data_dir /duong_dan/NEU-CLS.zip   --project csc4005-lab1-neu-mlp   --run_name baseline_adamw   --optimizer adamw   --lr 0.001   --weight_decay 0.0001   --dropout 0.3   --epochs 20   --batch_size 32   --img_size 64   --patience 5   --augment   --use_wandb
```

## 5. Kết quả đầu ra
Mỗi run sẽ được lưu vào `outputs/<run_name>/` gồm:
- `best_model.pt`
- `history.csv`
- `curves.png`
- `confusion_matrix.png`
- `metrics.json`

## 6. W&B
Tên project thống nhất là:
```text
csc4005-lab1-neu-mlp
```

Xem hướng dẫn chi tiết tại `docs/WANDB_GUIDE.md`.

## 7. Kiểm tra nhanh repo
```bash
python ci/check_structure.py
python ci/smoke_train.py
```
