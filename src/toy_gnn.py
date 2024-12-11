import json
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
import subprocess
import torch
from torch.nn import Linear
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling

# mostly from https://github.com/pyg-team/pytorch_geometric/blob/master/examples/link_pred.py

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tfs = T.RandomLinkSplit(num_val=0.05, num_test=0.05, is_undirected=False,
                      add_negative_train_samples=False)

with open("coo.json", "r") as coo_file:
    edge_index = torch.tensor(json.load(coo_file), dtype=torch.long).to(device)
with open("embeddings.npy", "rb") as embeddings_file:
    x = torch.tensor(np.load(embeddings_file), dtype=torch.float).to(device)
dataset = Data(x=x, edge_index=edge_index)


print(dataset.num_nodes, dataset.num_edges, dataset.has_isolated_nodes(), dataset.has_self_loops(), dataset.is_directed())

train_data, val_data, test_data = tfs(dataset)

class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.fc = Linear(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        return self.fc(x)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

model = Net(dataset.num_features, 128, 64).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
criterion = torch.nn.BCEWithLogitsLoss()

print(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")

def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)

    # We perform a new round of negative sampling for every training epoch:
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

    edge_label_index = torch.cat(
        [train_data.edge_label_index, neg_edge_index],
        dim=-1,
    )
    edge_label = torch.cat([
        train_data.edge_label,
        train_data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)

    out = model.decode(z, edge_label_index).view(-1)
    loss = criterion(out, edge_label)
    loss.backward()
    optimizer.step()
    return loss

@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())

@torch.no_grad()
def metrics(data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    out = model.decode(z, data.edge_label_index).view(-1)
    actual = data.edge_label.cpu().numpy()
    preds = (out.cpu().numpy() > 0.5)
    accuracy = accuracy_score(actual, preds)
    precision = precision_score(actual, preds)
    recall = recall_score(actual, preds)
    return accuracy, precision, recall

best_val_auc = final_test_auc = 0

for epoch in range(0, 1001):
    loss = train()
    val_auc = test(val_data)
    test_auc = test(test_data)
    if val_auc > best_val_auc:
        best_val = val_auc
        final_test_auc = test_auc
    test_acc, test_prec, test_recall = metrics(test_data)
    train_acc, train_prec, train_recall = metrics(train_data)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, Test: {test_auc:.4f},',
          f'Train A/P/R: {train_acc:.4f}/{train_prec:.4f}/{train_recall:.4f},',
          f'Test A/P/R: {test_acc:.4f}/{test_prec:.4f}/{test_recall:.4f}')
    if epoch % 100 == 0:
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, f"checkpoints/checkpoint_{epoch:04}.tar")

print(f'Final Test: {final_test_auc:.4f}')