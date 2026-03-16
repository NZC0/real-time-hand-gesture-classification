"""
MLP architecture for rude-gesture classification.

Input  : 63 features  (21 landmarks × 3 coords, normalised)
Hidden : 256 → 128 → 64 → 32  (ReLU + BatchNorm + Dropout 0.3)
Output : 7 classes  (raw logits — use CrossEntropyLoss during training)

Loss   : CrossEntropyLoss (LogSoftmax + NLLLoss; expects raw logits)
"""

import torch
import torch.nn as nn

# Class index ↔ name mapping (must stay in sync with data collection & JS)
CLASS_NAMES: list[str] = [
    "middle_finger",    # 0
    "reversed_v",       # 1
    "thumbs_up",        # 2
    "corna",            # 3
    "crossed_fingers",  # 4
    "ok_sign",          # 5
    "neutral",          # 6
]

NUM_FEATURES = 63   # 21 landmarks × (x, y, z)
NUM_CLASSES  = len(CLASS_NAMES)


class GestureMLP(nn.Module):
    """Multi-layer perceptron for hand-gesture classification."""

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()

        def block(in_features: int, out_features: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )

        self.net = nn.Sequential(
            block(NUM_FEATURES, 256),
            block(256, 128),
            block(128, 64),
            block(64, 32),
            nn.Linear(32, NUM_CLASSES),   # raw logits — no final softmax
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities (useful for inference)."""
        return torch.softmax(self.forward(x), dim=-1)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (class_indices, confidences)."""
        proba = self.predict_proba(x)
        confidences, indices = proba.max(dim=-1)
        return indices, confidences


def load_model(path: str, device: str = "cpu") -> GestureMLP:
    """Load a saved GestureMLP checkpoint."""
    model = GestureMLP()
    state = torch.load(path, map_location=device)
    # Support both raw state_dict and {'model_state_dict': ...} checkpoints
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model
