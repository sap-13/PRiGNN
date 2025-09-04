```json
{
  "architecture": {
    "layers": [
      {
        "layer_type": "GCNConv",
        "out_channels": 64,
        "activation": "prelu"
      },
      {
        "layer_type": "GATConv",
        "out_channels": 128,
        "activation": "relu"
      },
      {
        "layer_type": "GCNConv",
        "out_channels": "auto",
        "activation": "identity"
      }
    ]
  },
  "hyperparameters": {
    "epochs": 100,
    "lr": 0.005,
    "dropout": 0.5
  }
}
```