from dataclasses import dataclass
import yaml

@dataclass
class Profile:
    dataset_name: str
    task_type: str
    num_nodes: int
    num_edges: int
    num_features: int
    num_classes: int

    def to_prompt_string(self) -> str:
        return f"""
- Dataset: {self.dataset_name}
- Task Type: {self.task_type}
- Number of Nodes: {self.num_nodes}
- Number of Edges: {self.num_edges}
- Number of Features: {self.num_features}
- Number of Classes: {self.num_classes}
"""

def profile_data(dataset_name, data, task_type):
    """
    Profiles the dataset to extract key statistics.
    The number of classes is derived directly from the data.y tensor.
    """
    return Profile(
        dataset_name=dataset_name,
        task_type=task_type,
        num_nodes=data.num_nodes,
        num_edges=data.num_edges,
        num_features=data.num_features,
        num_classes=int(data.y.max().item()) + 1,
    )

def load_search_space(path: str = "prignn/search_space.yaml"):
    with open(path, 'r') as f:
        space = yaml.safe_load(f)
    return space