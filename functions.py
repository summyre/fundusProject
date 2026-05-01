import torch
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, classification_report, 
                             f1_score, recall_score, precision_score, roc_auc_score, roc_curve,
                             precision_recall_curve, average_precision_score)
from sklearn.metrics import auc as fn_auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import seaborn as sns
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from collections import defaultdict

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
            all_labels.extend(labels.cpu().numpy()) 
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

    y_true = label_binarize(all_labels, classes=np.arange(len(class_names)))
    y_score = all_probs

    print(f"""
{split_name.capitalize()} Results
accuracy:           {acc:.4f}
macro f1:           {m_f1:.4f}
macro recall:       {m_rec:.4f}
macro precision:    {m_prec:.4f}
macro auc:          {auc:.4f}
""")
    
    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    fig, ax = plt.subplots(figsize=(24,16))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(
        ax=ax,
        xticks_rotation=45,
        cmap='viridis',
        colorbar=True
    )
    
    for text in disp.text_.ravel():
        text.set_fontsize(10)
    
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)

    plt.title(f"{split_name.capitalize()} Confusion Matrix")
    plt.tight_layout(pad=3)
    plt.savefig(os.path.join(exp_dir, f"{split_name}_confusion_matrix.png"), dpi=300, bbox_inches='tight')

    # confusion matrix with sns
    plt.figure(figsize=(24,16))
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f',
        cmap='viridis',
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 10}
    )
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.title(f"{split_name.capitalize()} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, f"{split_name}_confusion_matrix_sns.png"), dpi=300, bbox_inches='tight')

    # per-class roc
    plt.figure(figsize=(10,8))
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true[:,i], y_score[:,i])
        roc_auc = fn_auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.2f})")
    
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{split_name.capitalize()} ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, f"{split_name}_roc_curves.png"))
    plt.close()

    # precision-recall curves
    plt.figure(figsize=(10,8))
    for i, class_name in enumerate(class_names):
        prec, rec, _ = precision_recall_curve(y_true[:,i], y_score[:,i])
        ap = average_precision_score(y_true[:,i], y_score[:,i])

        plt.plot(rec, prec, label=f"{class_name} (AP = {ap:.2f})")
    
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{split_name.capitalize()} Precision-Recall Curves")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, f"{split_name}_pr_curves.png"))
    plt.close()

    # classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)

    with open(os.path.join(exp_dir, f"{split_name}_classification_report.txt"), "w") as f:
        f.write(report)
    print(report)

    # saving predictions
    np.save(os.path.join(exp_dir, "preds.npy"), all_preds)
    np.save(os.path.join(exp_dir, "labels.npy"), all_labels)
    np.save(os.path.join(exp_dir, "probs.npy"), all_probs)

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
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'accuracy_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_gradcam(model, loader, device, exp_dir, class_names, num_images=20):
    class_counter = defaultdict(int)
    max_per_class = num_images // len(class_names)

    model.eval()
    os.makedirs(exp_dir, exist_ok=True)

    target_layers = [model.layer4[-1]] if hasattr(model, "layer4") else [model.fc]
    cam = GradCAM(model=model, target_layers=target_layers)

    for images,labels, paths in loader:
        images = images.to(device)

        for i in range(images.size(0)):
            if sum(class_counter.values()) >= num_images:
                return
            
            input_tensor = images[i].unsqueeze(0)
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).item()
            probs = torch.softmax(output, dim=1)
            confidence = probs[0, pred].item()
            true = labels[i].item()

            if class_counter[true] >= max_per_class:
                continue

            folder = "correct" if pred == true else "incorrect"
            class_name = class_names[true]
            path = os.path.join(exp_dir, folder, class_name)
            os.makedirs(path, exist_ok=True)

            # generate CAM
            grayscale_cam = cam(input_tensor=input_tensor)[0]

            # convert image back to numpy (for overlay)
            img = images[i].cpu().permute(1,2,0).numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)

            cam_img = show_cam_on_image(img, grayscale_cam, use_rgb=True)

            filename = f"{class_counter[true]}_true-{class_names[true]}_pred-{class_names[pred]}_conf-{confidence:.2f}.png"
            cv2.imwrite(os.path.join(path, filename), cam_img)
            class_counter[true] += 1
