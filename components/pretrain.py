# import sys
# sys.path.append(".")
# import torch
# from components.model   import GPTModel

# GPT_CONFIG_124M = {
# "vocab_size": 50257,
# "context_length": 256,
# "emb_dim": 768,
# "n_heads": 12,
# "n_layers": 12,
# "drop_rate": 0.1,
# "qkv_bias": False
# }

# # torch.manual_seed(123)
# # model= GPTModel(GPT_CONFIG_124M)
# # model.eval()

# import tiktoken
# from components.model import generate_text_simple

# def text_to_token_ids(text, tokenizer):
#     encoded= tokenizer.encode(text, allowed_special={'<|endoftext|>'})
#     encoded_tensor =torch.tensor(encoded).unsqueeze(0)  #ads batch dimension
#     return encoded_tensor
# def token_ids_to_text(token_ids, tokenizer):
#     flat = token_ids.squeeze(0)
#     return tokenizer.decode(flat.tolist()) #removes batch dimension

# # start_context ="Attention is all you need"
# # tokenizer= tiktoken.get_encoding("gpt2")
# # token_ids= generate_text_simple(model=model, idx=text_to_token_ids(start_context, tokenizer), max_new_tokens=10, context_size=GPT_CONFIG_124M["context_length"])

# # print("Output text: \n", token_ids_to_text(token_ids, tokenizer))
# # # text gen loss
# # inputs = torch.tensor([[16833, 3626, 6100], # ["every effort moves",
# # [40, 1107, 588]]) # "I really like"]
# # targets = torch.tensor([[3626, 6100, 345 ], # [" effort moves you",
# # [1107, 588, 11311]]) # " really like chocolate"]
# # with torch.no_grad():
# #     logits= model(inputs)
# # probas= torch.softmax(logits, dim=-1)
# # print(probas.shape)     
# # token_ids= torch.argmax(probas, dim=-1, keepdim=True)
# # print("Token IDs: ", token_ids)
# # print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
# # print(f"Outputs batch 1:"f" {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")

# from pathlib import Path
# file_path= Path(__file__).resolve().parent / "the-verdict.txt"
# with open(file_path, "r", encoding="utf-8") as file:
#     text_data=file.read()

# # training and validation dataset
# # train_ratio =0.9 #90% of data fr trraining and rest for validation
# # split_idx =int(train_ratio*len(text_data))
# # train_data =text_data[:split_idx]
# # val_data =text_data[split_idx:]

# # from utils.dataloader import create_dataloader_v1
# # torch.manual_seed(123)
 
# # train_loader= create_dataloader_v1(train_data, batch_size=2, max_length=GPT_CONFIG_124M["context_length"], stride=GPT_CONFIG_124M["context_length"], drop_last=True, shuffle=True, num_workers=0)
# # val_loader= create_dataloader_v1(val_data, batch_size=2, max_length=GPT_CONFIG_124M["context_length"], stride=GPT_CONFIG_124M["context_length"], drop_last=False, shuffle=False, num_workers=0)
# # print("Train Loader: ")
# # for x, y in train_loader:
# #     print(x.shape, y.shape)
# # print("Validation Loader: ")
# # for x, y in val_loader:
# #     print(x.shape, y.shape)

# # calculatin cross entropy loss of given batch
# def calc_loss_batch(input_batch, target_batch, model, device):
#     input_batch= input_batch.to(device)
#     target_batch =target_batch.to(device)
#     logits =model(input_batch)
#     loss =torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
#     return loss

# # computing training and validation loss
# def calc_loss_loader(data_loader,model, device, num_batches=None):
#     total_loss =0
#     if len(data_loader)== 0:
#         return float("nan")
#     elif num_batches is None:
#         num_batches =len(data_loader)
#     else:
#         num_batches =min(num_batches, len(data_loader))

#     # reduces no of batches to match total no of batches in data loader if num_batches exceeds no of batches in data loader
#     for i, (input_batch, target_batch) in enumerate(data_loader):
#         if i <num_batches:
#             loss = calc_loss_batch(input_batch, target_batch, model, device)
#             total_loss +=loss.item()
#         else:
#             break
#     return total_loss/ num_batches

# # device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # model.to(device)
# # with torch.no_grad():
# #     train_loss= calc_loss_loader(train_loader, model, device)
# #     val_loss =calc_loss_loader(val_loader, model, device)

# # print("Training loss: ", train_loss)
# # print("Validation loss: ", val_loss)    

# # pretraining llm
# def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq,eval_iter,start_context, tokenizer):
#     #lists to track token losses and tokens seen
#     train_losses, val_losses, track_token_seen= [],[], []
#     tokens_seen, global_step= 0, -1

#     for epoch in range(num_epochs):
#         model.train()
#         for input_batch, target_batch in train_loader:
#             optimizer.zero_grad()   #restes loss gradients from prevbatch iteration
#             loss =calc_loss_batch(input_batch, target_batch, model, device)
#             loss.backward() #calculatin loss gradients
#             optimizer.step() #updates model weights using loss gradients
#             tokens_seen +=input_batch.numel()
#             global_step += 1

#             # optimal evaluation step
#             if global_step%eval_freq== 0:
#                 train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
#                 train_losses.append(train_loss)
#                 val_losses.append(val_loss)
#                 track_token_seen.append(tokens_seen)
#                 print(f"Ep {epoch+1} (Step {global_step:06d}): "
#                 f"Train loss {train_loss:.3f}, "
#                 f"Val loss {val_loss:.3f}"
#                 )
#         generate_and_print_sample(model, tokenizer, device, start_context)
#     return train_losses, val_losses,track_token_seen

# def generate_and_print_sample(model, tokenizer, device, start_context):
#     model.eval()
#     context_size =model.pos_emb.weight.shape[0]
#     encoded= text_to_token_ids(start_context, tokenizer).to(device)
#     with torch.no_grad():
#         token_ids= generate_text_simple(model=model,idx=encoded, max_new_tokens=50, context_size=context_size)

#     decoded_text =token_ids_to_text(token_ids,tokenizer)
#     print(decoded_text.replace("\n", " "))
#     model.train()

# # calculates loss oveer training and validation set while ensuring model is in eval mode with gradient tracking annd droput disabled
# def evaluate_model(model, train_loader, val_loader,device, eval_iter):
#     model.eval()
#     with torch.no_grad():
#         train_loss= calc_loss_loader(train_loader, model, device,num_batches=eval_iter)
#         val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
#         model.train()

#     return train_loss, val_loss

# # torch.manual_seed(123)
# # model= GPTModel(GPT_CONFIG_124M)
# # model.to(device)
# # optimizer=torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
# # num_epochs= 10
# # train_losses, val_losses, tokens_seen = train_model_simple(model,train_loader, val_loader,optimizer,device, num_epochs=num_epochs, eval_freq=5, eval_iter=5,start_context="Every effort moves you",tokenizer=tokenizer)
# # plot for training and validation set losses
# # import matplotlib.pyplot as plt
# # from matplotlib.ticker import MaxNLocator
# # def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
# #     fig,ax1 =plt.subplots(figsize=(5,3))
# #     ax1.plot(epochs_seen, train_losses, label="Training loss")
# #     ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
# #     ax1.set_xlabel("Epochs")
# #     ax1.set_ylabel("Loss")
# #     ax1.legend(loc="upper right")
# #     ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
# #     ax2 = ax1.twiny()
# #     ax2.plot(tokens_seen, train_losses, alpha=0)
# #     ax2.set_xlabel("Tokens seen")
# #     fig.tight_layout()
# #     plt.show()

# # epochs_tensor= torch.linspace(0, num_epochs,len(train_losses))
# # plot_losses(epochs_tensor, tokens_seen,train_losses,val_losses)

import torch

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(
        text,
        allowed_special={"<|endoftext|>"}
    )
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())