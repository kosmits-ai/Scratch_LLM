import torch
import torch.nn as nn
import tiktoken # type: ignore
from torch.utils.data import Dataset, DataLoader
import urllib.request
import os
import time
from src.gpt_download3 import download_and_load_gpt2
import numpy as np

GPT_CONFIG_124M = {
   'vocab_size': 50257,
   'context_length': 256,
   'emb_dim': 768,
   'n_heads': 12,
   'n_layers': 12,
   'drop_rate': 0.1,
   'qkv_bias': False
}


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads):
        super().__init__()
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.Q_query = nn.Linear(d_in, d_out)  #Linear offers better initialization than random values
        self.W_key = nn.Linear(d_in, d_out)
        self.W_value = nn.Linear(d_in, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))
        self.out_proj = nn.Linear(d_out, d_out)

    def forward(self, x):
        b , num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.Q_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2) #we do this in order to declare keys based on num_heads so we can split after for each head.
        queries = queries.transpose(1,2)
        values = values.transpose(1,2)


        attention_scores = queries @ keys.transpose(2,3) 

        mask_bool = self.mask.bool() [:num_tokens, :num_tokens]
        attention_scores.masked_fill_(mask_bool, -torch.inf)

        attention_weights = torch.softmax(attention_scores / (keys.shape[-1] ** 0.5), dim=-1)
        attention_weights = self.dropout(attention_weights)

        context_vector = (attention_weights @ values).transpose(1,2) #shape (b, num_tokens, num_heads, head_dim)

        context_vector = context_vector.contiguous().view(b, num_tokens, self.d_out)
        context_vector = self.out_proj(context_vector)

        return context_vector
    
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim = True)
        var = x.var(dim=-1, keepdim = True, unbiased = False)
        norm_x = (x - mean) / torch.sqrt(var+self.eps)
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (1+ torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3)))) #activation function of gpt-2

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
        nn.Linear(cfg['emb_dim'] , 4 * cfg['emb_dim']), #expansion
        GELU(), #activation
        nn.Linear(4 * cfg['emb_dim'], cfg['emb_dim'] #contraction
        ))
    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in = cfg['emb_dim'],
            d_out = cfg['emb_dim'],
            context_length = cfg['context_length'],
            num_heads = cfg['n_heads'],
            dropout = cfg['drop_rate']
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg['emb_dim'])
        self.norm2 = LayerNorm(cfg['emb_dim'])
        self.drop_shortcut = nn.Dropout(cfg['drop_rate'])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x) #shape [batch_Size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = shortcut + x

        return x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])

        self.trf_blocks = nn.Sequential(* [TransformerBlock(cfg) for _ in range(cfg['n_layers'])])

        self.final_norm = LayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(
            cfg['emb_dim'], cfg['vocab_size'], bias=False
        )
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits


class DeepNeuralNetworkExample(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU()),
        ])
    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and x.shape ==layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x

def generate_text(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        
        with torch.no_grad():
            logits = model(idx_cond)    #batch, num_tokens, vocab_size
            logits = logits[:, -1, :]
            probas = torch.softmax(logits, dim=-1)
            idx_next = torch.argmax(probas, dim=-1, keepdim=True)
            idx = torch.cat([idx, idx_next], dim=1)
    
    return idx

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={'<|endoftext|>'})

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding('gpt2')
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    drop_last=drop_last,
    num_workers=num_workers
    )

    return dataloader

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), target_batch.flatten())
    return loss 

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float('nan')
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss
        else:
            break
    return total_loss / num_batches


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [] , []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() #reset loss grads fro previous iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() #calculate loss grads
            optimizer.step()    #update model weights
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss: .3f}, Val loss {val_loss: .3f}")
                
        generate_and_print_example(model, tokenizer, device, start_context)
    return train_losses, val_losses, track_tokens_seen

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches = eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def generate_and_print_example(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text(
            model = model, idx = encoded, max_new_tokens = 50, context_size = context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace('\n', ' '))
        model.train()

# Logits -> Top-K logits -> Logits / temp -> softmax -> multinomial sampling


def scaled_temperature(logits, temperature):        #the higher the temperature the more creative the model.
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)

def print_sampled_tokens(probas, tokenizer):
    torch.manual_seed(123) # Manual seed for reproducibility
    sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample))
    output = token_ids_to_text(sampled_ids, tokenizer)
    return output

def  generate(model, idx, max_new_tokens, context_size, temperature =0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

def _copy_param_(param: torch.nn.Parameter, array_like):
    """Copy numpy/torch data into an existing Parameter preserving dtype/device."""
    with torch.no_grad():
        src = torch.as_tensor(array_like, dtype=param.dtype, device=param.device)
        if param.shape != src.shape:
            raise ValueError(f"Shape mismatch. Param {tuple(param.shape)} vs source {tuple(src.shape)}")
        param.copy_(src)


def load_weights_into_gpt(gpt, params):
    # embeddings
    _copy_param_(gpt.pos_emb.weight, params['wpe'])
    _copy_param_(gpt.tok_emb.weight, params['wte'])

    # transformer blocks
    for b in range(len(params['blocks'])):
        pb = params['blocks'][b]

        # ---- attention qkv (fused) ----
        c_attn_w = pb['attn']['c_attn']['w']  # (embed, 3*embed)
        c_attn_b = pb['attn']['c_attn']['b']  # (3*embed,)
        q_w, k_w, v_w = np.split(c_attn_w, 3, axis=-1)
        q_b, k_b, v_b = np.split(c_attn_b, 3, axis=-1)

        # Linear expects (out_features, in_features)
        _copy_param_(gpt.trf_blocks[b].att.Q_query.weight, q_w.T)
        _copy_param_(gpt.trf_blocks[b].att.W_key.weight,   k_w.T)
        _copy_param_(gpt.trf_blocks[b].att.W_value.weight, v_w.T)
        _copy_param_(gpt.trf_blocks[b].att.Q_query.bias, q_b)
        _copy_param_(gpt.trf_blocks[b].att.W_key.bias,   k_b)
        _copy_param_(gpt.trf_blocks[b].att.W_value.bias, v_b)

        # ---- attention output projection (c_proj) ----
        _copy_param_(gpt.trf_blocks[b].att.out_proj.weight, pb['attn']['c_proj']['w'].T)
        _copy_param_(gpt.trf_blocks[b].att.out_proj.bias,   pb['attn']['c_proj']['b'])

        # ---- MLP ----
        _copy_param_(gpt.trf_blocks[b].ff.layers[0].weight, pb['mlp']['c_fc']['w'].T)
        _copy_param_(gpt.trf_blocks[b].ff.layers[0].bias,   pb['mlp']['c_fc']['b'])
        _copy_param_(gpt.trf_blocks[b].ff.layers[2].weight, pb['mlp']['c_proj']['w'].T)
        _copy_param_(gpt.trf_blocks[b].ff.layers[2].bias,   pb['mlp']['c_proj']['b'])

        # ---- layer norms ----
        _copy_param_(gpt.trf_blocks[b].norm1.scale, pb['ln_1']['g'])
        _copy_param_(gpt.trf_blocks[b].norm1.shift, pb['ln_1']['b'])
        _copy_param_(gpt.trf_blocks[b].norm2.scale, pb['ln_2']['g'])
        _copy_param_(gpt.trf_blocks[b].norm2.shift, pb['ln_2']['b'])

    # final layer norm (ln_f)
    _copy_param_(gpt.final_norm.scale, params['g'])
    _copy_param_(gpt.final_norm.shift, params['b'])

    # tied output head
    _copy_param_(gpt.out_head.weight, params['wte'])


url = 'https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt'
file_path = 'the-verdict.txt'

if not os.path.exists(file_path):
    with urllib.request.urlopen(url) as response:
        text_data = response.read().decode('utf-8')
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text_data)

else:
    with open(file_path, 'r', encoding='utf-8') as file:
        text_data = file.read()

tokenizer = tiktoken.get_encoding('gpt2')

total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))

print('Total characters of text_data:', total_characters)
print('Vocabulary size:', total_tokens)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
settings, params = download_and_load_gpt2(model_size='124M', models_dir='gpt2')

print('Settings:', settings)
print('Parameters dictionary keys:', params.keys())

print(params['wte'])
print('Token embeddings weight tensor dimensions:', params['wte'].shape)

model_config = {
    'gpt2-small (124M)' : {'emb_dim': 768, 'n_layers': 12, 'n_heads': 12}
}

model_name = 'gpt2-small (124M)'
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_config[model_name])
NEW_CONFIG.update({'context_length':1024, 'qkv_bias': True})
gpt = GPTModel(NEW_CONFIG)
gpt.eval()

load_weights_into_gpt(gpt, params)
gpt.to(device)

torch.manual_seed(123)
token_ids = generate(
    model=gpt,
    idx=text_to_token_ids('Every effort moves you', tokenizer).to(device),
    max_new_tokens=25,
    context_size=NEW_CONFIG['context_length'],
    top_k=50,
    temperature=1.5
)

print('Output text:\n', token_ids_to_text(token_ids, tokenizer))
