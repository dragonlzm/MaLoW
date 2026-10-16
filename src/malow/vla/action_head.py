"""MLP action decoder extracted from Hippocampus; see SOURCE_NOTICE.md."""
import torch
from torch import nn

class ActionSimpleMLPModel(nn.Module):
    """
    Simple MLP-based action prediction head.

    Directly maps condition features to actions per-timestep using a lightweight MLP.
    This is simpler than ActionMLPModel as it processes each timestep independently
    without flattening all condition tokens together.

    Input: [B, T, D] where T = future_action_window_size + 1, D = action_hidden_dim (e.g., 2048)
    Output: [B, T, action_dim] where action_dim is typically 7
    """

    def __init__(self, action_hidden_dim, action_dim, future_action_window_size, hidden_dim=512, num_layers=2, dropout=0.1):
        """
        Initialize simple MLP action prediction model.

        Args:
            action_hidden_dim: Input feature dimension per timestep (e.g., 2048).
            action_dim: Output action dimensionality per timestep (e.g., 7).
            future_action_window_size: Number of future steps to predict.
            hidden_dim: Hidden dimension of MLP layers.
            num_layers: Number of MLP layers (including output layer).
            dropout: Dropout probability.
        """
        super().__init__()
        self.in_channels = action_dim
        self.action_dim = action_dim
        self.action_hidden_dim = action_hidden_dim
        self.future_action_window_size = future_action_window_size
        layers = []
        current_dim = action_hidden_dim
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(current_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout)])
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, action_dim))
        self.mlp = nn.Sequential(*layers)

    def predict(self, condition):
        """
        Inference: predict actions from conditioning features.

        Args:
            condition: Conditioning features [B, T, D].

        Returns:
            torch.Tensor: Predicted action sequence [B, T, action_dim].
        """
        import time
        time_start = time.time()
        action_pred = self.mlp(condition)
        return action_pred

def get_action_model(config):
    c = config.framework.action_model
    if c.action_model_type != "mlp":
        raise ValueError("This release supports the Qwen3.5 MLP action head only.")
    return ActionSimpleMLPModel(c.action_hidden_dim, c.action_dim,
        c.future_action_window_size, getattr(c, "mlp_hidden_dim", 512),
        getattr(c, "mlp_num_layers", 2), getattr(c, "mlp_dropout", 0.1))
