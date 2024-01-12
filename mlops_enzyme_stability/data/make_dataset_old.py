if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    import torch
    from tqdm import tqdm
    from transformers import BertModel, BertTokenizer
    import re

    raw_path = "data/raw/"
    save_path = "data/processed/"

    # Define encoder class using pretrained model
    class BertEncoder:
        def __init__(self):
            
            self.tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
            self.model = BertModel.from_pretrained("Rostlab/prot_bert")

        def tokenize(self, sequence, only_encoding=True):

            sequence_subbed = re.sub(r"[UZOB]", "X", sequence)
            encoded_input = self.tokenizer(sequence_subbed, return_tensors='pt')
            last_layer, pooling = self.model(**encoded_input)
            
            if only_encoding:
                return pooling

            return pooling, last_layer

    # Load in data, preprocess it and apply the BERT encoder to make a dataset of encoded amino acid sequences.
    
    train_df = pd.read_csv(raw_path + "train.csv")
    test_df = pd.read_csv(raw_path + "test.csv")
    test_labels = pd.read_csv(raw_path + "test_labels.csv")

    def add_spaces(x):
        return " ".join(list(x))

    encoder = BertEncoder()

    Xtrain = torch.Tensor([encoder.tokenize(add_spaces(x)) for x in tqdm(train_df["protein_sequence"], desc="Encoding training data")])
    ytrain = torch.Tensor(train_df["tm"])
    Xtest = torch.Tensor([encoder.tokenize(add_spaces(x)) for x in tqdm(test_df["protein_sequence"], desc="Encoding test data")])
    ytest = test_labels["tm"]
    
    save_path = "data/processed/"

    torch.save(Xtrain, save_path + "train_tensors.pt")
    torch.save(ytrain, save_path + "train_target.pt")
    torch.save(Xtest, save_path + "test_tensors.pt")
    torch.save(ytest, save_path + "test_target.pt")
    