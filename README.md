# NCS

Repo nghiên cứu **session-based / sequential recommendation**, tổ chức theo nhóm.

## Cấu trúc thư mục

```
NCS_serve/
├── nhom1/          # GNN session — nền tảng
│   ├── SR-GNN/
│   └── GCE-GNN/
├── nhom2/          # GNN session — tự giám sát (SSL)
│   ├── DHCN/
│   ├── COTREC/
│   └── CSGNN/
├── nhom3/          # Sequential / framework tích hợp
│   ├── DuoRec/
│   └── SelfContrastiveLearningRecSys/
├── MyProject/      # Dự án cá nhân
│   ├── thucnghiem/
│   └── thucnghiem2/
├── Data/           # Data tập trung (không commit)
├── Log/            # Log quá trình
├── LogMins/        # Log kết quả cuối
├── ncs_paths.py    # Helper path
└── ncs_logging.py  # Helper log
```

**Lưu ý:** `Data/`, `Log/`, `LogMins/` vẫn dùng tên bài toán trực tiếp (ví dụ `Data/DHCN/`, không phải `Data/nhom2/DHCN/`).

## Chạy nhanh (ví dụ)

```bash
# GCE-GNN test
python3 nhom1/GCE-GNN/main.py --data_path Data/Test/GCE-GNN --dataset test --auto_num_node --epoch 1

# DHCN test
python3 nhom2/DHCN/main.py --dataset test --data_path Data/Test/DHCN --epoch 5

# DuoRec
cd nhom3/DuoRec && python run_seq.py --dataset=ml-100k --model=DuoRec
```
