import torch
from src.model import get_model

def test_model_output_shape():
    model = get_model()
    model.eval()
    dummy_input = torch.randn(1, 3, 32, 32)
    output = model(dummy_input)
    assert output.shape == (1, 10)