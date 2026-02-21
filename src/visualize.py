import torch
import matplotlib.pyplot as plt


def plot_metrics(metrics_path="/Users/bimalkumal/AI Projects/Flower_Classifier/saved_models/metrics.pt"):
    metrics = torch.load(metrics_path, map_location="cpu")

    train_losses = metrics["train_losses"]
    val_losses = metrics["val_losses"]
    val_accuracies = metrics["val_accuracies"]

    epochs = range(1, len(train_losses) + 1)

    # -------- Loss Plot --------
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.show()

    # -------- Accuracy Plot --------
    plt.figure()
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Validation Accuracy")
    plt.show()


if __name__ == "__main__":
    plot_metrics()