from importlib.metadata import version
print("torch version", version("torch"))
import torch
inputs= torch.tensor(
    [[0.43, 0.15, 0.89], 
[0.55, 0.87, 0.66], 
[0.57, 0.85, 0.64], 
[0.22, 0.58, 0.33], 
[0.77, 0.25, 0.10], 
[0.05, 0.80, 0.55]] 
)
query= inputs[1]
attn_scores_2= torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i]= torch.dot(x_i, query)
print(attn_scores_2)
# normalzing previously computed attention 
attn_weights_2_tmp= attn_scores_2/ attn_scores_2.sum()
print("Attention weights: ", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())
# softmax for normalizing attention scores
def softmax_naive(x):
    return torch.exp(x)/ torch.exp(x).sum(dim=0)

attn_weights_2_naive= softmax_naive(attn_scores_2)
print("Attention weights: ", attn_weights_2_naive)
print("Sum: ", attn_weights_2_naive.sum())
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights: ", attn_weights_2)
print("Sum: ", attn_weights_2.sum())
query= inputs[1]
context_vec_2 =torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]* x_i
print(context_vec_2)
attn_scores= torch.empty(6,6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j]= torch.dot(x_i, x_j)
print(attn_scores)
attn_scores= inputs@inputs.T
print(attn_scores)
attn_weights = torch.softmax(attn_scores, dim=1)
print(attn_weights)
row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("Row 2 sum:", row_2_sum)
print("All row sums:", attn_weights.sum(dim=-1))
all_context_vecs= attn_weights@inputs
print(all_context_vecs)
x_2 =inputs[1]
d_in =inputs.shape[1]
d_out= 2

torch.manual_seed(123)
W_query =torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key =torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value =torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
query_2 =x_2 @ W_query
key_2 =x_2 @ W_key
value_2 =x_2 @ W_value

keys = inputs @ W_key
values = inputs @ W_value
print("keys.shape:", keys.shape)
print("values.shape:", values.shape)
keys_2 =keys[1]
attn_scores_22= query_2.dot(keys_2)
print(attn_scores_22)
attn_scores_2 =query_2@ keys.T
print(attn_scores_2)
d_k =keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print(attn_weights_2)
context_vec_2 =attn_weights_2@ values
print(context_vec_2)
# self attention class
import torch.nn as nn
class SelfAttention_V1(nn.Module): 
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query= nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys= x @self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec
torch.manual_seed(123)
sa_v1 = SelfAttention_V1(d_in, d_out)
print(sa_v1(inputs))
# self attenton class using pytorch linear layers
class SelfAttention_V2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    def forward(self, x):
        keys =self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
        attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec
torch.manual_seed(789)
sa_v2 = SelfAttention_V2(d_in, d_out)
print(sa_v2(inputs))
queries =sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores =queries@keys.T
attn_weights = torch.softmax(attn_scores/ keys.shape[-1]**0.5, dim=-1)
print(attn_weights)

context_length= attn_scores.shape[0]
mask_simple =torch.tril(torch.ones(context_length, context_length))
print(mask_simple)
masked_simple =attn_weights *mask_simple
print(masked_simple)
# renormalizin attention weights to sum up to 1 again in each row
row_sum =masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm =masked_simple/ row_sum
print(masked_simple_norm)
mask =torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked =attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)
attn_weights =torch.softmax(masked/ keys.shape[-1]** 0.5, dim =1)
print(attn_weights)
torch.manual_seed(123)
dropout =torch.nn.Dropout(0.5)
ex =torch.ones(6, 6)
print(dropout(ex))
torch.manual_seed(123)
print(dropout(attn_weights))
batch =torch.stack((inputs, inputs), dim=0)
print(batch.shape)
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out= d_out
        self.W_query= nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout =nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in =x.shape
        keys =self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores= queries@keys.transpose(1, 2)
        attn_scores.masked_fill(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores/ keys.shape[-1]**0.5, dim =-1)
        attn_weights =self.dropout(attn_weights)
        context_vec =attn_weights@ values
        return context_vec
torch.manual_seed(123)
context_length =batch.shape[1]
ca= CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs= ca(batch)
print("context_vecs.shape:", context_vecs.shape)
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads= nn.ModuleList([CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
        for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
torch.manual_seed(123)
context_length =batch.shape[1]
d_in, d_out =3, 2
mha= MultiHeadAttentionWrapper(d_in, d_out, context_length, 0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out,context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
        "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask",torch.triu(torch.ones(context_length, context_length),diagonal=1))
    def forward(self, x):
        b, num_tokens,d_in = x.shape
        keys =self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys =keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries =queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1,2)
        values =values.transpose(1, 2)

        # computes dot product for each head
        attn_scores =queries@keys.transpose(2, 3)
        mask_bool =self.mask.bool()[:num_tokens, :num_tokens]   #masks truncateed to no of tokens
        # uses mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights =torch.softmax(attn_scores/ keys.shape[-1]**0.5, dim= -1)
        attn_weights =self.dropout(attn_weights)

        context_vec =(attn_weights@ values).transpose(1,2)
        context_vec= context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec =self.out_proj(context_vec)

        return context_vec
torch.manual_seed(123)
batch_size, context_length, d_in =batch.shape
d_out =2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs= mha(batch)
print(context_vecs)