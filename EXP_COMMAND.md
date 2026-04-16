# Cấu hình 1: AdamW có augment
Cấu hình này dùng làm mốc baseline với AdamW và bật tăng cường dữ liệu.

```bash
python -m src.train 
--data_dir "data/NEU-CLS" 
--project csc4005-lab1-neu-mlp 
--run_name adamw_aug 
--optimizer adamw 
--lr 0.001 
--weight_decay 0.0001 
--dropout 0.3 
--epochs 20 
--batch_size 32 
--img_size 64 
--patience 5 
--augment 
--use_wandb 
--device cuda 
--wandb_mode online 
```

# Cấu hình 2: AdamW không augment
Cấu hình này giúp so sánh trực tiếp tác động của augment.

```bash
python -m src.train 
--data_dir "data/NEU-CLS" 
--project csc4005-lab1-neu-mlp 
--run_name adamw_no_aug 
--optimizer adamw 
--lr 0.001 
--weight_decay 0.0001 
--dropout 0.3 
--epochs 20 
--batch_size 32 
--img_size 64 
--patience 5 
--use_wandb 
--device cuda 
--wandb_mode online 
```

# Cấu hình 3: SGD có augment
Cấu hình này phục vụ so sánh AdamW và SGD trong cùng bối cảnh có augment.

```bash
python -m src.train 
--data_dir "data/NEU-CLS" 
--project csc4005-lab1-neu-mlp 
--run_name sgd_aug 
--optimizer sgd 
--lr 0.01 
--weight_decay 0.0001 
--dropout 0.3 
--epochs 20 
--batch_size 32 
--img_size 64 
--patience 5 
--augment 
--use_wandb 
--device cuda 
--wandb_mode online 
```

# Cấu hình 4: AdamW có scheduler plateau
Cấu hình này kiểm tra ảnh hưởng của giảm learning rate theo validation loss.

```bash
python -m src.train 
--data_dir "data/NEU-CLS" 
--project csc4005-lab1-neu-mlp 
--run_name adamw_plateau 
--optimizer adamw 
--scheduler plateau 
--lr 0.001 
--weight_decay 0.0001 
--dropout 0.3 
--epochs 20 
--batch_size 32 
--img_size 64 
--patience 5 
--augment 
--use_wandb 
--device cuda 
--wandb_mode online 
```
