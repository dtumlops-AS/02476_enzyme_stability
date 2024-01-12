import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
import os

# define directories and create them if not existing
save_path = "mlops_enzyme_stability/data/processed/"
if not os.path.exists(save_path):
        os.makedirs(save_path)
raw_path = "mlops_enzyme_stability/data/raw/"
if not os.path.exists(raw_path):
        os.makedirs(raw_path)

def preprocessing(df_train,df_train_updates):
    # Remove rows with all features missing
    all_features_nan = df_train_updates.isnull().all("columns")
    drop_indices = df_train_updates[all_features_nan].index
    df_train = df_train.drop(index=drop_indices)

    # Correct transposed pH and tm values
    swap_ph_tm_indices = df_train_updates[~all_features_nan].index
    df_train.loc[swap_ph_tm_indices, ["pH", "tm"]] = df_train_updates.loc[swap_ph_tm_indices, ["pH", "tm"]]
    
    # for pytest purposes
    return df_train

def load_data():
    train_df = pd.read_csv(raw_path + "train_fixed.csv")[:3]
    test_df = pd.read_csv(raw_path + "test.csv")[:3]
    test_labels = pd.read_csv(raw_path + "test_labels.csv")[:3]
    return train_df, test_df, test_labels

def add_spaces(x):
        return " ".join(list(x))

def tokenize_and_encode_sequence(tokenizer, model, device, sequence):
    tokenized = tokenizer(sequence, return_tensors='pt')
    output = model(**tokenized.to(device))
    return output[1].detach().cpu()

def save_tensor(tensor_list, file_path):
    tensor = torch.stack(tensor_list)
    torch.save(tensor, file_path)
    del tensor  # To free memory

def main():
    # load data for preprocessing
    train_raw = pd.read_csv(f"{raw_path}train.csv", index_col="seq_id")
    train_updates_raw = pd.read_csv(f"{raw_path}train_updates_20220929.csv", index_col="seq_id")
    df_train = preprocessing(train_raw, train_updates_raw)
    # Save the updated training data
    df_train.to_csv(f"{raw_path}train_fixed.csv")

    # load data
    train_df, test_df, test_labels = load_data()

    dataloader_train = torch.utils.data.DataLoader(train_df["protein_sequence"], batch_size=1, shuffle=False, num_workers=0)
    dataloader_test = torch.utils.data.DataLoader(test_df["protein_sequence"], batch_size=1, shuffle=False, num_workers=0)

    # Model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    model = BertModel.from_pretrained("Rostlab/prot_bert")
    model = model.to(device)
    model.eval()

    # Process and encode data
    Xtrain = [tokenize_and_encode_sequence(tokenizer, model, device, seq) for seq in tqdm(dataloader_train, desc="Encoding training data")]
    Xtest = [tokenize_and_encode_sequence(tokenizer, model, device, seq) for seq in tqdm(dataloader_test, desc="Encoding testing data")]
            
    # Save encoded data
    save_tensor(Xtrain, save_path + "train_tensors.pt")
    save_tensor(Xtest, save_path + "test_tensors.pt")

    # Save labels
    ytrain = torch.Tensor(train_df["tm"])
    ytest = torch.Tensor(test_labels["tm"])
    torch.save(ytrain, save_path + "train_target.pt")
    torch.save(ytest, save_path + "test_target.pt")

    print("\nFinished embedding")

if __name__ == "__main__":
    main()