"""CIFAR-10 소규모 교육용 훈련 스크립트.

예:
python -m cv_models.train_cifar10 --epochs 2 --subset-size 5000 --eval-size 2000

정확도는 실행 전에는 알 수 없고, ImageNet 논문 결과와 직접 비교할 수 없습니다.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from cv_models.resnet import TinyResNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='교육용 CIFAR-10 TinyResNet 학습')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--subset-size', type=int, default=5000)
    parser.add_argument('--eval-size', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    args = parser.parse_args()
    if min(args.epochs, args.subset_size, args.eval_size, args.batch_size) <= 0:
        parser.error('epochs, subset-size, eval-size, batch-size는 모두 양수여야 합니다.')
    if args.lr <= 0:
        parser.error('lr는 양수여야 합니다.')
    if args.subset_size > 50000 or args.eval_size > 10000:
        parser.error('CIFAR-10 split 크기를 초과했습니다 (train=50000, test=10000).')
    return args


def reproducible_subset(dataset, n: int, seed: int):
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:n]
    return Subset(dataset, indices.tolist())


def run_epoch(model: nn.Module, loader: DataLoader, device: torch.device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            bs = labels.size(0)
            total_loss += loss.item() * bs
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += bs
    return total_loss / total, 100.0 * correct / total


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise SystemExit('torchvision이 설치되지 않았습니다. https://pytorch.org/get-started/locally/ 를 참조하세요.') from exc

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_ds = datasets.CIFAR10(root=str(args.data_dir), train=True,
                                download=True, transform=preprocess)
    test_ds = datasets.CIFAR10(root=str(args.data_dir), train=False,
                               download=True, transform=preprocess)
    train_ds = reproducible_subset(train_ds, args.subset_size, args.seed)
    test_ds = reproducible_subset(test_ds, args.eval_size, args.seed + 1)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = TinyResNet().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    print(f'환경: {device} / torch={torch.__version__} / seed={args.seed}')
    print(f'data: train={len(train_ds)} / test={len(test_ds)}')
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, device, optimizer)
        te_loss, te_acc = run_epoch(model, test_loader, device)
        print(f'epoch={epoch:02d} train_loss={tr_loss:.4f} '
              f'train_acc={tr_acc:.2f}% test_loss={te_loss:.4f} test_acc={te_acc:.2f}%')
    print('참고: 반복적인 모델 선택에는 공식 test split과 분리된 validation split을 사용하세요.')


if __name__ == '__main__':
    main()
