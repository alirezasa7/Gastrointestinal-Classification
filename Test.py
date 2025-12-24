from dataloader_selection import get_dataloaders
from MSR import MSR
import os
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def test(model, test_loader, device):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    acc = 100. * accuracy_score(all_targets, all_preds)
    precision = 100. * precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = 100. * recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = 100. * f1_score(all_targets, all_preds, average='macro', zero_division=0)

    return acc, precision, recall, f1


def main(args):
    print(":=========== Test Configuration ===========")
    print(f"| Dataset    : {args.dataset}")
    print(f"| Data path  : {args.data_dir}")
    print(f"| Batch size : {args.bsz}")
    print(f"| Weights    : {args.weights}")
    print(":==========================================")


    _, test_loader = get_dataloaders(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.bsz
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MSR(num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)

    acc, prec, rec, f1= test(model, test_loader, device)

    print("\n========== Test Results ==========")
    print(f"Accuracy  : {acc:.2f}%")
    print(f"Precision : {prec:.2f}%")
    print(f"Recall    : {rec:.2f}%")
    print(f"F1-score  : {f1:.2f}%")
    print("=================================\n")



if __name__ == "__main__":
    class Args:
        def __init__(self):
            self.dataset = # name of the selected dataset
            self.num_classes = # class numbers of the selected dataset
            self.bsz = 32
            self.data_dir = # dataset directory
            self.weights = # directory of the bestmodel.pth(saved weights)

    args = Args()
    main(args)
