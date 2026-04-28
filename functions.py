import torch
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, f1_score, recall_score, precision_score, roc_auc_score
import matplotlib.pyplot as plt
import os

# -- early stopping definition -- # -- not needed rip
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
    all_probs = []

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())     # not being coloured in??
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # accuracy
    acc = np.mean(all_preds == all_labels)

    # macro metrics
    m_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    m_rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    m_prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')

    print(f"""
{split_name.capitalize()} Results
accuracy:           {acc:.4f}
macro f1:           {m_f1:.4f}
macro recall:       {m_rec:.4f}
macro precision:    {m_prec:.4f}
macro auc:          {auc:.4f}
""")
    
    cm = confusion_matrix(all_labels, all_preds, normalize='true')

    plt.figure(figsize=(12,8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(xticks_rotation=45)
    plt.title(f"{split_name.capitalize()} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, f"{split_name}_confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.show()

    # classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)

    with open(os.path.join(exp_dir, f"{split_name}_classification_report.txt"), "w") as f:
        f.write(report)
    print(report)

    return acc, m_f1, m_rec

def plot_history(history, exp_dir, model_name):
    epochs = range(1, len(history['train_loss']) + 1)

    # -- loss curves -- #
    plt.figure(figsize=(10,6))
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title(f"{model_name} - Training and Validation Loss", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # -- accuracy curves -- #
    plt.figure(figsize=(10,6))
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title(f"{model_name} - Training and Validation Accuracy", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True)

    # add best validation accuracy annotation
    best_val_acc = max(history['val_acc'])
    best_epoch = history['val_acc'].index(best_val_acc) + 1
    plt.annotate(f"Best Val Acc: {best_val_acc:.2f}%\n(Epoch {best_epoch})",
                 xy=(best_epoch, best_val_acc),
                 xytext=(best_epoch + 2, best_val_acc - 5),
                 arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'accuracy_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()