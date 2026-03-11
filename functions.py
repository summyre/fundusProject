import torch
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, f1_score, recall_score
import matplotlib.pyplot as plt
import os

# -- early stopping definition -- #
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score > self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0
        
    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)

# -- evaluate model function -- #
def evaluate_model(model, loader, device, class_names, exp_dir, split_name="val"):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # accuracy
    acc = np.mean(all_preds == all_labels)

    # macro metrics
    m_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    m_rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)

    print(f"""
{split_name.capitalize()} Results
accuracy:       {acc:.4f}
macro f1:       {m_f1:.4f}
macro recall:   {m_rec:.4f}
""")
    
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(12,8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(xticks_rotation=45)
    plt.title(f"{split_name.capitalize()} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, f"{split_name}_confusion_matrix.png"))
    plt.show()

    # classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)

    with open(os.path.join(exp_dir, f"{split_name}_classification_report.txt"), "w") as f:
        f.write(report)
    print(report)

    return acc, m_f1, m_rec
