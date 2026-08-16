import torch
import torch.nn as nn
import tiktoken
import sys
from attention import MultiHeadAttention

sys.path.append("..")
tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch)

GPT_CONFIG_124M= {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768, #embedding dimension
    "n_heads": 12,   # no of heads
    "n_layers": 12, #no of layers
    "drop_rate": 0.1,
    "qkv_bias": False   
}

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return 0.5*x* (1 + torch.tanh(torch.sqrt(torch.tensor(2.0/ torch.pi)) *(x +0.044715* torch.pow(x, 3))))

# layer normalization class
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale =nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var= x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x =(x -mean)/ torch.sqrt(var + self.eps)
        return self.scale* norm_x + self.shift


# feed forward neural network
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),GELU(),nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),)

    def forward(self, x):
        return self.layers(x)
ffn = FeedForward(GPT_CONFIG_124M)
x =torch.rand(2, 3,768)
out =ffn(x)
print(out.shape)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att =MultiHeadAttention(d_in=cfg["emb_dim"],d_out=cfg["emb_dim"],context_length=cfg["context_length"],num_heads=cfg["n_heads"],dropout=cfg["drop_rate"],qkv_bias=cfg["qkv_bias"])
        self.ff =FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # shortcut conn. for attention block
        shortcut= x     
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut    #addin original input back

        shortcut =x
        x = self.norm2(x)   #shortcut conn for feed forward block
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut    #addin original input back
        return x
torch.manual_seed(123)
x = torch.rand(2, 4, 768)
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)
print("Input shape:", x.shape)
print("Output shape:", output.shape)

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])    
        self.trf_blocks =nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm= LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len= in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

torch.manual_seed(123)
model =GPTModel(GPT_CONFIG_124M)
out= model(batch)
print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)
print(out)

# generating text
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print("encoded: ", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded_tensor.shape:", encoded_tensor.shape)

model.eval()
out =generate_text_simple(model=model, idx=encoded_tensor, max_new_tokens=6, context_size=GPT_CONFIG_124M["context_length"])
print("Output: ", out)
print("Output length: ", len(out[0]))

# convertin ids back to text
decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text) #untrained output, its great!!!