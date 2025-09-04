```json
{
  "architecture": {
    "layers": [
      {
        "layer_type": "GATConv",
        "out_channels": 128,
        "activation": "leaky_relu"
      },
      {
        "layer_type": "SAGEConv",
        "out_channels": 64,
        "activation": "prelu"
      },
      {
        "layer_type": "GCNConv",
        "out_channels": "auto",
        "activation": "identity"
      }
    ]
  },
  "hyperparameters": {
    "epochs": 200,
    "lr": 0.001,
    "dropout": 0.5
  }
}
```