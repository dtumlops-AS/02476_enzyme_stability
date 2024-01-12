from tests import _PATH_DATA
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from mlops_enzyme_stability.data.make_dataset import load_data, save_tensor, preprocessing
import os

def test_load_data():
    train_df, test_df, test_labels = load_data()
    for df in [train_df, test_df, test_labels]:
        assert isinstance(df, pd.DataFrame)
        assert not df.empty, "Raw data could not be loaded"

def test_preprocessing():
    # Mock data simulating the scenarios in your preprocessing function
    mock_train_data = {
        'seq_id': [1, 2, 3, 4],
        'pH': [7.0, 8.0, 7.5, 6.5],
        'tm': [50, 55, 60, 45],
    }
    mock_train_updates_data = {
        'seq_id': [1, 2, 3],
        'pH': [None, 7.8, None],
        'tm': [None, 62, None],
    }
    mock_solution = {
        'seq_id': [2, 4],
        'pH': [7.8, 6.5],
        'tm': [62, 45],
    }

    df_train = pd.DataFrame(mock_train_data).set_index('seq_id')
    df_train_updates = pd.DataFrame(mock_train_updates_data).set_index('seq_id')
    df_solution = pd.DataFrame(mock_solution).set_index('seq_id')

    # Call preprocessing
    processed_df = preprocessing(df_train, df_train_updates)

    # Check that rows with all features missing are removed
    assert processed_df.equals(df_solution), "Preprocessing failed"


def test_save_tensor(tmp_path):
    tensor_dummy = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
    file_path = tmp_path / "test_tensor.pt"
    save_tensor(tensor_dummy, file_path)
    assert file_path.exists()





def test_dataloader():
    processed_dir = os.path.join(_PATH_DATA, 'processed')
    train_tensors = torch.load(os.path.join(processed_dir, "train_tensors.pt"))
    train_labels = torch.load(os.path.join(processed_dir, "train_target.pt"))
    test_tensors = torch.load(os.path.join(processed_dir, "test_tensors.pt"))
    test_labels = torch.load(os.path.join(processed_dir, "test_target.pt"))
    
    trainset = TensorDataset(train_tensors, train_labels)
    trainloader = DataLoader(trainset, batch_size=16, shuffle=True)
    assert len(trainloader) == 1812, "Train dataloader should have 1812 batches"
    
    testset = TensorDataset(test_tensors, test_labels)
    testloader = DataLoader(testset, batch_size=16, shuffle=True)
    print(len(testloader))
    assert len(testloader) == 151, "Test dataloader should have 151 batches"
