# train.py
import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
import joblib

# --------------------------
# RNN Model
# --------------------------
class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super(RNNClassifier, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout, nonlinearity="tanh")
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return self.fc(out)

# --------------------------
# Dataset
# --------------------------
class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --------------------------
# Utils
# --------------------------
def create_sequences(data, labels, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(labels[i+seq_len])
    return np.array(X), np.array(y)

def plot_class_distribution(y_train, y_val, y_test, class_names, save_path):
    train_counts = np.bincount(y_train, minlength=len(class_names))
    val_counts   = np.bincount(y_val, minlength=len(class_names))
    test_counts  = np.bincount(y_test, minlength=len(class_names))

    df_counts = pd.DataFrame({
        "Class": class_names * 3,
        "Count": np.concatenate([train_counts, val_counts, test_counts]),
        "Split": ["Train"] * len(class_names) + ["Val"] * len(class_names) + ["Test"] * len(class_names)
    })

    plt.figure(figsize=(10,6))
    sns.barplot(x="Class", y="Count", hue="Split", data=df_counts)
    plt.title("Class Distribution per Split")
    plt.xticks(rotation=30)
    plt.savefig(save_path)
    plt.close()

# --------------------------
# Training
# --------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, model_dir, class_names):
    best_acc = 0
    train_losses, val_losses, val_accs = [], [], []

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss, preds, trues = 0, [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                preds.extend(torch.argmax(outputs, 1).cpu().numpy())
                trues.extend(y_batch.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = accuracy_score(trues, preds)
        val_f1 = f1_score(trues, preds, average="weighted")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch}/{epochs}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}, val_f1={val_f1:.4f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_size": model.rnn.input_size,
                "hidden_size": model.rnn.hidden_size,
                "num_layers": model.rnn.num_layers,
                "num_classes": len(class_names),
                "dropout": model.rnn.dropout
            }, os.path.join(model_dir, "best_model.pth"))
            print(f"New best model saved (val_acc={val_acc:.4f})")

    # Plot loss curves
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curves")
    plt.savefig(os.path.join(model_dir, "loss_curves.png"))
    plt.close()

    return model

# --------------------------
# Main
# --------------------------
def main(args):
    os.makedirs(args.model_dir, exist_ok=True)

    # Load dataset
    file_path = os.path.join(args.data_dir, "conveyor_fault_dataset.csv")
    df = pd.read_csv(file_path)

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Feature mapping
    feature_map = {
        "Speed": "Speed (rpm)",
        "Load": "Load (kg)",
        "Temperature": "Temperature (℃)",
        "Vibration": "Vibration (m/s²)",
        "Current": "Current (A)"
    }
    features = [feature_map[k] for k in feature_map]
    label_col = "Fault"

    # Encode labels
    le = LabelEncoder()
    df[label_col] = le.fit_transform(df[label_col])
    class_names = le.classes_.tolist()

    # Save class names
    with open(os.path.join(args.model_dir, "classes.txt"), "w") as f:
        f.write("\n".join(class_names))

    # Scale features
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    # Save scaler + features
    joblib.dump(scaler, os.path.join(args.model_dir, "scaler.pkl"))
    with open(os.path.join(args.model_dir, "features.txt"), "w") as f:
        f.write("\n".join(features))

    # Sequence creation
    X, y = create_sequences(df[features].values, df[label_col].values, args.seq_len)

    # Stratified split
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test   = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)

    print(f"Splits: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # Plot class distribution
    plot_class_distribution(y_train, y_val, y_test, class_names,
                            os.path.join(args.model_dir, "class_distribution.png"))

    # Datasets and loaders
    train_loader = DataLoader(SequenceDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(SequenceDataset(X_val, y_val), batch_size=args.batch_size)
    test_loader  = DataLoader(SequenceDataset(X_test, y_test), batch_size=args.batch_size)

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RNNClassifier(input_size=len(features), hidden_size=args.hidden_size,
                          num_layers=args.num_layers, num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train
    model = train_model(model, train_loader, val_loader, criterion, optimizer,
                        device, args.epochs, args.model_dir, class_names)

    # Test
    checkpoint = torch.load(os.path.join(args.model_dir, "best_model.pth"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    preds_all, true_all = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            preds_all.extend(torch.argmax(outputs, 1).cpu().numpy())
            true_all.extend(y_batch.cpu().numpy())

    metrics = {}
    if len(true_all) > 0:
        acc = accuracy_score(true_all, preds_all)
        f1 = f1_score(true_all, preds_all, average="weighted")
        print(f"=== Final Test === Acc={acc:.4f}, F1={f1:.4f}")

        # Confusion Matrix
        cm = confusion_matrix(true_all, preds_all, labels=range(len(class_names)))
        fig, ax = plt.subplots(figsize=(8,6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, cmap="Blues", xticks_rotation=30, values_format="d")
        plt.title("Confusion Matrix (Test)")
        plt.savefig(os.path.join(args.model_dir, "confusion_matrix.png"))
        plt.close()

        # F1 per class
        report = classification_report(true_all, preds_all, target_names=class_names, output_dict=True)
        f1_scores = [report[c]["f1-score"] for c in class_names]
        plt.figure(figsize=(8,5))
        sns.barplot(x=class_names, y=f1_scores)
        plt.title("F1-score per Class (Test)")
        plt.xticks(rotation=30)
        plt.savefig(os.path.join(args.model_dir, "f1_per_class.png"))
        plt.close()

        # Save metrics.json
        metrics = {
            "test_accuracy": acc,
            "test_f1": f1,
            "num_classes": len(class_names),
            "class_names": class_names
        }
        with open(os.path.join(args.model_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
    else:
        print("Test set empty, skipping evaluation plots.")

# --------------------------
# Entry
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--data-dir", type=str, default="/opt/ml/input/data/train")
    parser.add_argument("--model-dir", type=str, default="/opt/ml/model")
    args = parser.parse_args()
    main(args)
