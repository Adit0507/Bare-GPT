import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids =[]
        self.target_ids = []

        token_ids =tokenizer.encode(txt)    #tokenizin entire text
        for i in range(0, len(token_ids)- max_length, stride): # sliding window to chunk book into overlapping sequences of max length
            input_chunk = token_ids[i: i +max_length]
            target_chunk =token_ids[i +1: i +max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    def __len__(self):  #returns total no. of rows in dataset
        return len(self.input_ids) 
    def __getitem__(self, index): #returns single row in dataset
        return self.input_ids[index], self.target_ids[index]
def create_dataloader_v1(txt, batch_size=24, max_length=256, stride= 128, shuffle=True, drop_last=True, num_workers= 0):
    tokenizer= tiktoken.get_encoding("gpt2")
    dataset =GPTDatasetV1(txt, tokenizer, max_length, stride)
                                                                            #drops the last batch if its shorter tahn specified batch size to prevetn loss spikes during training
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    return dataloader

# testing the dataloader
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text =f.read()

dataloader = create_dataloader_v1(raw_text,batch_size=1, max_length=4,stride=1, shuffle=False)
data_iter =iter(dataloader) #converts dataloader into py iterator to fetch next entry
first_batch =next(data_iter)
print(first_batch)