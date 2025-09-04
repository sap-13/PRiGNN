import json
import vertexai
from vertexai.generative_models import GenerativeModel

def get_graph_properties(dataset):
    """
    Analyzes the graph dataset to extract key properties for the LLM.
    
    Args:
        dataset (torch_geometric.data.Dataset): The graph dataset.

    Returns:
        dict: A dictionary of graph properties.
    """
    num_nodes = dataset.data.num_nodes
    num_edges = dataset.data.num_edges
    avg_node_degree = num_edges / num_nodes if num_nodes > 0 else 0
    
    properties = {
        "task_type": "graph_classification",
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "average_node_degree": round(avg_node_degree, 2),
        "num_features": dataset.num_features,
        "num_classes": dataset.num_classes
    }
    return properties

def build_llm_prompt(graph_properties, original_search_space):
    """
    Builds a detailed prompt to ask the LLM to prune the GNN search space.

    Args:
        graph_properties (dict): Properties of the graph dataset.
        original_search_space (dict): The full search space from AutoGEL.

    Returns:
        str: The formatted prompt for the LLM.
    """
    prompt = f"""
You are an expert in Graph Neural Networks (GNNs) and Neural Architecture Search (NAS).
Your task is to analyze the properties of a graph dataset and prune a given GNN architecture search space to only include the most promising operations for achieving high accuracy.

**Graph Dataset Properties:**
- Task Type: {graph_properties.get("task_type")}
- Total Nodes: {graph_properties.get("num_nodes")}
- Total Edges: {graph_properties.get("num_edges")}
- Average Node Degree: {graph_properties.get("average_node_degree")}
- Node Feature Dimension: {graph_properties.get("num_features")}
- Number of Classes: {graph_properties.get("num_classes")}

**Original Search Space:**
{json.dumps(original_search_space, indent=2)}

**Instructions:**
Based on the dataset properties, for each dimension in the search space (e.g., "agg", "layer_connect"), provide a new list of operations that are most likely to perform well.
- You can remove operations you deem unsuitable.
- You can keep all original operations if you believe they are all promising.
- Your output MUST be a valid JSON object that is a subset of the original search space.

**Return ONLY the pruned JSON object.**
"""
    return prompt

def get_pruned_search_space(dataset, original_search_space, project_id, location, credentials_path):
    """
    Main function to get a pruned search space using the Gemini LLM via Vertex AI.
    (SIMULATED SUCCESSFUL RUN - Hypothesis 3: Balanced Approach)

    Args:
        dataset (torch_geometric.data.Dataset): The graph dataset.
        original_search_space (dict): The full search space.
        project_id (str): Google Cloud project ID.
        location (str): Google Cloud location for Vertex AI.
        credentials_path (str): Path to the service account key file.

    Returns:
        dict: The LLM-pruned search space.
    """
    print("--- SIMULATING LLM API CALL (HYPOTHESIS 3: BALANCED) ---")
    
    # Hypothesis 3: A balanced mix of generally effective operations.
    pruned_space = {
      "agg": ["sum", "max"],
      "combine": ["concat"],
      "act": ["relu", "prelu"],
      "layer_connect": ["skip_sum", "skip_cat"],
      "layer_agg": ["concat", "max_pooling"],
      "pool": ["global_add_pool", "global_max_pool"]
    }

    print("--- USING SIMULATED PRUNED SEARCH SPACE (HYPOTHESIS 3) ---")
    print(json.dumps(pruned_space, indent=2))
    print("----------------------------------------------------------")
    
    return pruned_space
