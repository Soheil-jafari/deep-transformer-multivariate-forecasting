from dataset import load_data
from train import train_model
from evaluate import evaluate_model

X, Y, scaler = load_data("sample_data.csv", seq_len=24, pred_len=12)
model = train_model(X, Y)
evaluate_model(model, X, Y, scaler)
