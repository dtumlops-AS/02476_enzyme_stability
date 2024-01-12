from transformers import BertModel, BertTokenizer
import re
# Download the pretrained transformed model and initialize the tokenizer (encoder) and the decoder

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

if __name__ == "__main__":
    sequence_Example = "A E T C Z A O"
    print(f"encoding sequence: {sequence_Example}")
    tokenizer = BertEncoder()
    last, pool = tokenizer.tokenize(sequence_Example)

    print(last.shape)
    print(pool.shape)
        

