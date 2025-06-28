
import torch
from MSR import MSR
from data_loader_kvasir import get_test_loader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MSR(num_classes=6)
model.load_state_dict(torch.load('/kaggle/working/best_model.pth', map_location=device))
model.to(device)
model.eval()

test_loader = get_test_loader('/kaggle/input/kvasir', batch_size=32)
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


accuracy = 100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
print(f'Accuracy of best_model on test set: {accuracy:.2f}%')


cm = confusion_matrix(all_labels, all_preds)
class_names = ['DLP', 'DRM', 'ESO', 'NORM', 'polyp', 'ULC']  

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.xlabel('Predicted Labels', fontweight='bold')
plt.ylabel('True Labels', fontweight='bold')
plt.title('Confusion Matrix')
plt.show()
