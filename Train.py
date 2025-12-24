from dataloader_selection import get_dataloaders
from MSR import MSR
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score
from FocalLoss import FocalLoss


focal = FocalLoss(alpha=1, gamma=2, reduction='mean')
ce = nn.CrossEntropyLoss()
alpha = 0.3
def hybrid_loss(outputs, targets):
    return alpha * focal(outputs, targets) + (1 - alpha) * ce(outputs, targets)


def count_parameters(model, trainable_only=False):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())



def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    all_preds, all_targets = [], []

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

        if batch_idx % 100 == 0:
            acc = 100. * correct / total
            print(f'Batch {batch_idx}/{len(train_loader)} | '
                  f'Loss {running_loss/(batch_idx+1):.3f} | '
                  f'Acc {acc:.2f}%')

    acc = 100. * correct / total
    precision = 100. * precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = 100. * recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = 100. * f1_score(all_targets, all_preds, average='macro', zero_division=0)

    return running_loss / len(train_loader), acc, precision, recall, f1


def main(args):
    print(":=========== Training Configuration ===========")
    print(f"| Dataset     : {args.dataset}")
    print(f"| Data path   : {args.data_dir}")
    print(f"| Batch size  : {args.bsz}")
    print(f"| LR          : {args.lr}")
    print(f"| Epochs      : {args.niter}")
    print(":==============================================")


    train_loader, _ = get_dataloaders(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.bsz
    )

    model = MSR(num_classes=args.num_classes)

    print(f"\nTotal params      : {count_parameters(model)}")
    print(f"Trainable params : {count_parameters(model, True)}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = hybrid_loss

    os.makedirs(args.logpath, exist_ok=True)
    best_acc = 0.0

    for epoch in range(args.niter):
        print(f"\nEpoch [{epoch + 1}/{args.niter}]")
        start = time.time()

        tr_loss, tr_acc, tr_p, tr_r, tr_f1 = train(
            model, train_loader, optimizer, criterion, device
        )

        print(f"Train | Loss {tr_loss:.4f} | Acc {tr_acc:.2f}% | F1 {tr_f1:.2f}%")
        print(f"Time  | {time.time() - start:.2f}s")

        if tr_acc > best_acc:
            best_acc = tr_acc
            os.makedirs(args.logpath, exist_ok=True)
            torch.save(
                model.state_dict(),
                os.path.join(args.logpath, "best_model.pth")
            )
            print(f"Best model saved (Train Acc = {best_acc:.2f}%)")



if __name__ == "__main__":
    class Args:
        def __init__(self):
            self.dataset = # write the selected dataset name
            self.num_classes = # write number of classes of the selected dataset
            self.bsz = 32
            self.lr = 1e-4
            self.niter = 60
            self.data_dir = # dataset directory
            self.logpath = #all outputs directory

    args = Args()
    main(args)
