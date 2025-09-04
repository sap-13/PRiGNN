PROMPT_TEMPLATE = """
You are a world-class expert in Graph Neural Networks (GNNs) and automated machine learning (AutoML). Your goal is to design a state-of-the-art GNN architecture for the given task, search space, and performance history.

**1. TASK CONTEXT**

You are performing a node classification task on the **{dataset_name}** dataset. Here are its vital statistics:
- **Task Type:** {task_type}
- **Nodes:** {num_nodes}
- **Edges:** {num_edges}
- **Features per Node:** {num_features}
- **Number of Classes:** {num_classes}
- **Evaluation Metric:** {metric_name}

**2. SEARCH SPACE DEFINITION**

You must design a GNN architecture by selecting components from the following search space. The output must be a valid JSON object.

```yaml
{search_space}
```

**3. PERFORMANCE HISTORY**

You have already evaluated several architectures. Here are the results, sorted from best to worst validation performance. Learn from these previous attempts to inform your next design.

{past_performance_history}

**4. STRATEGIC GUIDANCE**

You are in round **{round_no}** of **{max_rounds}**.
{strategy_guidance}

**5. YOUR TASK**

Based on all the information above, generate a new GNN architecture in the specified JSON format. Your design should be a novel and promising candidate that intelligently explores the search space, taking into account the historical performance and strategic guidance.

**Output only the JSON object.**

Example:
```json
{{
  "architecture": {{
    "layers": [
      {{
        "layer_type": "GATConv",
        "out_channels": 128,
        "activation": "relu",
        "heads": 8
      }},
      {{
        "layer_type": "GCNConv",
        "out_channels": "auto",
        "activation": "identity"
      }}
    ]
  }},
  "hyperparameters": {{
    "epochs": 200,
    "lr": 0.005,
    "dropout": 0.6
  }}
}}
```
"""