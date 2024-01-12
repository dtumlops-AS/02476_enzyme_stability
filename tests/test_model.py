import torch

def test_model():
    from mlops_enzyme_stability.models.MLP import MyNeuralNet
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("config.yaml")
    model = MyNeuralNet(cfg)
    random_input = torch.rand(1024)
    out = model(random_input)
    assert out.shape == (1,), "Model output shape should be (1,)"
    print(model.criterion)
    assert str(model.criterion) == "MSELoss()", "Model criterion should be torch.nn.MseLoss()"
    