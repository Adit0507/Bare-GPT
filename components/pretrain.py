import torch
from components.model import generate_text_simple

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

def generate(model, idx, max_new_tokens, context_size,temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        if top_k is not None:
            top_logits, _= torch.topk(logits, top_k)
            min_val =top_logits[:, -1]
            logits =torch.where(logits <min_val,torch.tensor(float('-inf')).to(logits.device),logits)
        if temperature >0.0:
            logits =logits/ temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next =torch.argmax(logits, dim=-1, keepdim=True)
        if idx_next== eos_id:
            break
        idx =torch.cat((idx,idx_next), dim=1)
    return idx

# calculatin cross entropy loss of given batch
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch= input_batch.to(device)
    target_batch =target_batch.to(device)
    logits =model(input_batch)
    loss =torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

# computing training and validation loss
def calc_loss_loader(data_loader,model, device, num_batches=None):
    total_loss =0
    if len(data_loader)== 0:
        return float("nan")
    elif num_batches is None:
        num_batches =len(data_loader)
    else:
        num_batches =min(num_batches, len(data_loader))

    # reduces no of batches to match total no of batches in data loader if num_batches exceeds no of batches in data loader
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i <num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss +=loss.item()
        else:
            break
    return total_loss/ num_batches

# pretraining llm
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq,eval_iter,start_context, tokenizer):
    #lists to track token losses and tokens seen
    train_losses, val_losses, track_token_seen= [],[], []
    tokens_seen, global_step= 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()   #restes loss gradients from prevbatch iteration
            loss =calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() #calculatin loss gradients
            optimizer.step() #updates model weights using loss gradients
            tokens_seen +=input_batch.numel()
            global_step += 1

            # optimal evaluation step
            if global_step%eval_freq== 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_token_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                f"Train loss {train_loss:.3f}, "
                f"Val loss {val_loss:.3f}"
                )
        generate_and_print_sample(model, tokenizer, device, start_context)
    return train_losses, val_losses,track_token_seen

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size =model.pos_emb.weight.shape[0]
    encoded= text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids= generate_text_simple(model=model,idx=encoded, max_new_tokens=50, context_size=context_size)

    decoded_text =token_ids_to_text(token_ids,tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()

# calculates loss oveer training and validation set while ensuring model is in eval mode with gradient tracking annd droput disabled
def evaluate_model(model, train_loader, val_loader,device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss= calc_loss_loader(train_loader, model, device,num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
        model.train()

    return train_loss, val_loss