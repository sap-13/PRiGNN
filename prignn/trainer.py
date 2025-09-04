
import torch
import torch.nn.functional as F

def train_and_eval(model, data, task_type, train_mask, val_mask, test_mask, metric_fn, epochs=200, lr=0.005):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        val_acc = metric_fn(out[val_mask], data.y[val_mask])
        test_acc = metric_fn(out[test_mask], data.y[test_mask])
        
    return val_acc, test_acc, loss.item()

