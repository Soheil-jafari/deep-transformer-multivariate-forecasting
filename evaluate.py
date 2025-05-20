import matplotlib.pyplot as plt
import os
import torch

def evaluate_model(model, X, Y, scaler):
    model.eval()
    with torch.no_grad():
        pred = model(X).detach().numpy()
        true = Y.detach().numpy()

    pred_flat = pred.reshape(-1, pred.shape[2])
    true_flat = true.reshape(-1, true.shape[2])

    pred_inv = scaler.inverse_transform(pred_flat)
    true_inv = scaler.inverse_transform(true_flat)

    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(true_inv[:, 0], label="True")
    plt.plot(pred_inv[:, 0], label="Predicted")
    plt.legend()
    plt.title("Forecast vs Ground Truth (feature 0)")
    plt.savefig("outputs/forecast_plot.png")
    plt.close()
