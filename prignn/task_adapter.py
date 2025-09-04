from dataclasses import dataclass
from typing import Callable, Any
import torch_geometric.datasets as datasets
from torch_geometric.data import Dataset

@dataclass
class Task:
    name: str
    dataset: Dataset
    task_type: str
    metric_fn: Callable[[Any, Any], float]

def get_task(name: str) -> Task:
    if name.lower() == 'cora':
        return Task(
            name='Cora',
            dataset=datasets.Planetoid(root='/tmp/Cora', name='Cora'),
            task_type='NodeClassification',
            metric_fn=lambda y_pred, y_true: (y_pred.argmax(dim=1) == y_true).float().mean().item()
        )
    elif name.lower() == 'citeseer':
        return Task(
            name='CiteSeer',
            dataset=datasets.Planetoid(root='/tmp/CiteSeer', name='CiteSeer'),
            task_type='NodeClassification',
            metric_fn=lambda y_pred, y_true: (y_pred.argmax(dim=1) == y_true).float().mean().item()
        )
    elif name.lower() == 'pubmed':
        return Task(
            name='PubMed',
            dataset=datasets.Planetoid(root='/tmp/PubMed', name='PubMed'),
            task_type='NodeClassification',
            metric_fn=lambda y_pred, y_true: (y_pred.argmax(dim=1) == y_true).float().mean().item()
        )
    elif name.lower() == 'ogbn-arxiv':
        from ogb.nodeproppred import PygNodePropPredDataset
        return Task(
            name='ogbn-arxiv',
            dataset=PygNodePropPredDataset(name='ogbn-arxiv', root='/tmp/ogbn-arxiv'),
            task_type='NodeClassification',
            metric_fn=lambda y_pred, y_true: (y_pred.argmax(dim=1) == y_true).float().mean().item()
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")