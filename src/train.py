import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from model import get_model
from utils import get_device, save_best_model

train_losses = []
val_losses = []
val_accuracies = []

def train():
    data_dir = "/Users/bimalkumal/Downloads/flower"
    batch_size = 32
    epochs = 30
    lr = 0.0001

    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, class_names = get_dataloaders(data_dir, batch_size)

    model = get_model(num_classes=len(class_names), drop_out=0.4, pretrained=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0

    for epoch in range(epochs):
        # Train
        model.train()
        running_train_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        epoch_train_loss = running_train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)


        # Validation
        model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        epoch_val_loss = running_val_loss / len(val_loader)
        val_accuracy = 100.0 * correct / total

        val_losses.append(epoch_val_loss)
        val_accuracies.append(val_accuracy)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.2f}%"
        )

        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            save_best_model(model, "saved_models/best_model.pth")
            print("Best model saved!")

    print(f"Training complete. Best Val Acc: {best_val_acc:.2f}%")
    metrics = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies
    }
    torch.save(metrics, "saved_models/metrics.pt")
    print(" Metrics saved!")

if __name__ == "__main__":
    train()