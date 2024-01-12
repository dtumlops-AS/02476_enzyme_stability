import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
import re
import os

if __name__ == "__main__":
    # Paths to raw and processed data
    raw_path = "data/raw/"
    save_path = "data/processed/"

    def add_spaces(x):
        return " ".join(list(x))

    # Data
    train_df = pd.read_csv(raw_path + "train_fixed.csv")[:3]
    test_df = pd.read_csv(raw_path + "test.csv")[:3]
    test_labels = pd.read_csv(raw_path + "test_labels.csv")[:3]
    dataloader_train = torch.utils.data.DataLoader(train_df["protein_sequence"], batch_size=1, shuffle=False, num_workers=0)
    dataloader_test = torch.utils.data.DataLoader(test_df["protein_sequence"], batch_size=1, shuffle=False, num_workers=0)

    # Model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    model = BertModel.from_pretrained("Rostlab/prot_bert")
    model = model.to(device)
    model.eval()

    # Encode train data
    Xtrain, Xtest = [], []
    for i, x in enumerate(tqdm(dataloader_train, desc="Encoding training data")):
        token = tokenizer(add_spaces(x), return_tensors='pt')
        output = model(**token.to(device))
        Xtrain.append(output[1].detach().cpu())
            
    # Convert to tensors and save
    Xtrain = torch.stack(Xtrain)
    torch.save(Xtrain, save_path + "train_tensors.pt")
    del Xtrain # Remove variable for memory reasons
        
    # Encode test data
    for i, x in enumerate(tqdm(dataloader_test, desc="Encoding testing data")):
        token = tokenizer(add_spaces(x), return_tensors='pt')
        output = model(**token.to(device))
        Xtest.append(output[1].detach().cpu())
        
    # Convert to tensors and save
    Xtest = torch.stack(Xtest)
    torch.save(Xtest, save_path + "test_tensors.pt")

    # Save labels
    ytrain = torch.Tensor(train_df["tm"])
    ytest = torch.Tensor(test_labels["tm"])
    torch.save(ytrain, save_path + "train_target.pt")
    torch.save(ytest, save_path + "test_target.pt")

    print("\nFinished embedding")