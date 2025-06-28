
from data_loader_kvasir import get_data_loader
from MSR import MSR  
import argparse
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
    else:
        return sum(p.numel() for p in model.parameters())

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

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
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {running_loss/(batch_idx+1):.3f}, Accuracy: {acc:.2f}%')

    acc = 100. * correct / total
    precision = 100. * precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = 100. * recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = 100. * f1_score(all_targets, all_preds, average='macro', zero_division=0)

    train_loss = running_loss / len(train_loader)
    return train_loss, acc, precision, recall, f1

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    acc = 100. * correct / total
    precision = 100. * precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = 100. * recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = 100. * f1_score(all_targets, all_preds, average='macro', zero_division=0)

    val_loss = running_loss / len(val_loader)
    return val_loss, acc, precision, recall, f1

def main(args):
    print(":=========== Kvasir Dataset ===========")
    print(f"|             datapath: {args.data_dir}")
    print(f"|              logpath: {args.logpath}")
    print(f"|                  bsz: {args.bsz}")
    print(f"|                   lr: {args.lr}")
    print(f"|                niter: {args.niter}")
    print(":========================================")

    train_loader, val_loader = get_data_loader(args.data_dir, args.bsz)
    model = MSR(num_classes=6)  

    total_params = count_parameters(model)
    trainable_params = count_parameters(model, trainable_only=True)
    print(f"\nTotal # param.: {total_params}")
    print(f"Trainable # param.: {trainable_params}\n")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = hybrid_loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0

    for epoch in range(args.niter):
        print(f'Epoch {epoch+1}/{args.niter}')
        start_time = time.time()
        
        train_loss, train_acc, train_prec, train_rec, train_f1 = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_prec, val_rec, val_f1 = validate(model, val_loader, criterion, device)

        end_time = time.time()
        epoch_time = end_time - start_time

        print(f"Train — Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, Prec: {train_prec:.2f}%, Rec: {train_rec:.2f}%, F1: {train_f1:.2f}%")
        print(f"Val   — Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, Prec: {val_prec:.2f}%, Rec: {val_rec:.2f}%, F1: {val_f1:.2f}%")
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds\n")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(args.logpath, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.logpath, 'best_model.pth'))
            print('Best model saved with accuracy: {:.2f}%'.format(best_val_acc))

if __name__ == "__main__":
    class Args:
        def __init__(self):
            self.bsz = 32
            self.lr = 1e-4
            self.niter = 60
            self.data_dir = "/kaggle/input/kvasir"  
            self.logpath = "/kaggle/working/log"  

    args = Args()
    main(args)
