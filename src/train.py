from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.dataset import create_dataloaders
from src.model import MLPClassifier
from src.utils import (
    EarlyStopping,
    classification_report_dict,
    compute_accuracy,
    ensure_dir,
    plot_curves,
    save_confusion_matrix,
    save_history_csv,
    save_json,
    set_seed,
)

try:
    import wandb
except ImportError:
    wandb = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train MLP for NEU surface defect classification')
    parser.add_argument('--data_dir', type=str, required=True, help='Đường dẫn tới NEU-CLS.zip hoặc thư mục dữ liệu đã giải nén')
    parser.add_argument('--project', type=str, default='csc4005-lab1-neu-mlp')
    parser.add_argument('--run_name', type=str, default='debug_run')
    parser.add_argument('--optimizer', type=str, choices=['adamw', 'sgd'], default='adamw')
    parser.add_argument('--scheduler', type=str, choices=['none', 'plateau'], default='none')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[512, 256, 64])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val_size', type=float, default=0.15)
    parser.add_argument('--test_size', type=float, default=0.15)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_mode', type=str, choices=['online', 'offline'], default='online')
    return parser.parse_args()


def get_optimizer(name: str, model: nn.Module, lr: float, weight_decay: float):
    if name == 'adamw':
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == 'sgd':
        return SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f'Unsupported optimizer: {name}')


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == 'cpu':
        return torch.device('cpu')
    if device_arg == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError(
                'Bạn yêu cầu --device cuda nhưng CUDA không khả dụng trong môi trường hiện tại. '
                'Hãy kiểm tra bản PyTorch CUDA, driver và phần cứng GPU hỗ trợ.'
            )
        return torch.device('cuda')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=(device.type == 'cuda'))
        y = y.to(device, non_blocking=(device.type == 'cuda'))
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
    return running_loss / len(loader.dataset), compute_accuracy(y_true, y_pred)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=(device.type == 'cuda'))
        y = y.to(device, non_blocking=(device.type == 'cuda'))
        logits = model(x)
        loss = criterion(logits, y)
        running_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
    return running_loss / len(loader.dataset), compute_accuracy(y_true, y_pred), y_true, y_pred


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = ensure_dir(Path('outputs') / args.run_name)

    data = create_dataloaders(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.seed,
        augment=args.augment,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )
    print(f'Resolved data directory: {data.resolved_data_dir}')
    print(f'Classes: {data.class_names}')
    if device.type == 'cuda':
        print(f'Using device: {device} ({torch.cuda.get_device_name(0)})')
    else:
        print(f'Using device: {device}')

    model = MLPClassifier(
        input_dim=data.input_dim,
        num_classes=len(data.class_names),
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(args.optimizer, model, args.lr, args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2) if args.scheduler == 'plateau' else None

    use_wandb = args.use_wandb and wandb is not None
    if args.use_wandb and wandb is None:
        print('Cảnh báo: chưa import được wandb. Chạy tiếp ở chế độ không log online.')
    if use_wandb:
        wandb.init(project=args.project, name=args.run_name, config=vars(args), mode=args.wandb_mode)
        wandb.config.update({
            'num_classes': len(data.class_names),
            'class_names': data.class_names,
            'input_dim': data.input_dim,
            'device': str(device),
            'resolved_data_dir': data.resolved_data_dir,
            'gpu_name': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
        })

    history: list[dict[str, float]] = []
    early_stopper = EarlyStopping(patience=args.patience)
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, data.train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, data.val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step(val_loss)
        lr_current = optimizer.param_groups[0]['lr']
        row = {
            'epoch': epoch,
            'train_loss': round(train_loss, 6),
            'train_acc': round(train_acc, 6),
            'val_loss': round(val_loss, 6),
            'val_acc': round(val_acc, 6),
            'lr': lr_current,
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | lr={lr_current:.6f}"
        )
        if use_wandb:
            wandb.log(row)
        if early_stopper.step(val_loss):
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, output_dir / 'best_model.pt')
        if early_stopper.should_stop:
            print(f'Early stopping at epoch {epoch}')
            break

    if best_state is None:
        best_state = model.state_dict()
        torch.save(best_state, output_dir / 'best_model.pt')

    model.load_state_dict(torch.load(output_dir / 'best_model.pt', map_location=device))
    test_loss, test_acc, y_true, y_pred = evaluate(model, data.test_loader, criterion, device)
    report = classification_report_dict(y_true, y_pred, data.class_names)
    cm = save_confusion_matrix(y_true, y_pred, data.class_names, output_dir / 'confusion_matrix.png')
    plot_curves(history, output_dir / 'curves.png')
    save_history_csv(history, output_dir / 'history.csv')
    metrics = {
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'class_names': data.class_names,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'resolved_data_dir': data.resolved_data_dir,
    }
    save_json(metrics, output_dir / 'metrics.json')
    print(f'Best val acc: {best_val_acc:.4f}')
    print(f'Test acc: {test_acc:.4f}')
    print(f'Saved outputs to: {output_dir}')

    if use_wandb:
        wandb.log({
            'best_val_acc': best_val_acc,
            'best_val_loss': best_val_loss,
            'test_acc': test_acc,
            'test_loss': test_loss,
            'confusion_matrix_image': wandb.Image(str(output_dir / 'confusion_matrix.png')),
            'curves_image': wandb.Image(str(output_dir / 'curves.png')),
        })
        wandb.finish()


if __name__ == '__main__':
    main()
