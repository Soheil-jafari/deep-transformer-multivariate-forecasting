import torch
import torch.nn as nn
from model import TransformerForecast

def train_model(X, Y):
    model = TransformerForecast(input_dim=X.shape[2])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        output = model(X)
        loss = loss_fn(output, Y)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    return model
