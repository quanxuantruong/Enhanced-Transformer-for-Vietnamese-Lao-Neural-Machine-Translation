import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# Bỏ import lr_scheduler
# import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import sentencepiece as spm
import sacrebleu

import math
import os
import time
import random
from tqdm import tqdm

# --- Configuration ---
# Paths
DATA_DIR = '/data2/cmdir/home/ioit107/thviet/MTVL/data'
TRAIN_VI = os.path.join(DATA_DIR, 'train2023.vi')
TRAIN_LO = os.path.join(DATA_DIR, 'train2023.lo')
DEV_VI = os.path.join(DATA_DIR, 'dev2023.vi')
DEV_LO = os.path.join(DATA_DIR, 'dev2023.lo')
TEST_VI = os.path.join(DATA_DIR, 'test2023.vi')
TEST_LO = os.path.join(DATA_DIR, 'test2023.lo')
MODEL_PREFIX_VI = '/data2/cmdir/home/ioit107/thviet/MTVL/source/spm_vi'
MODEL_PREFIX_LO = '/data2/cmdir/home/ioit107/thviet/MTVL/source/spm_lo'
MODEL_SAVE_PATH = '/data2/cmdir/home/ioit107/thviet/MTVL/source/transformer_vi_lo_final_best.pt'

# Model Hyperparameters
SRC_VOCAB_SIZE_SPM = 8000
TGT_VOCAB_SIZE_SPM = 8000
EMB_SIZE = 256
NHEAD = 4
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
FFN_HID_DIM_GEGLU = int(EMB_SIZE * 8 / 3)
if FFN_HID_DIM_GEGLU % NHEAD !=0: FFN_HID_DIM_GEGLU = (FFN_HID_DIM_GEGLU // NHEAD +1) * NHEAD
DROPOUT = 0.1
MAX_SEQ_LEN_ROPE = 512

# Training Hyperparameters
FIXED_LEARNING_RATE = 1e-4 # Learning rate cố định
BATCH_SIZE = 128
NUM_EPOCHS = 20
CLIP = 1.0
LABEL_SMOOTHING = 0.1
SEED = 42
# WARMUP_STEPS đã bị loại bỏ

# Decoding Hyperparameters (Greedy)
MAX_LEN_EVAL = 100

# --- Setup ---
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- SentencePiece Training & Loading ---
def train_sentencepiece(input_file, model_prefix, vocab_size):
    if not os.path.exists(model_prefix + '.model'):
        print(f"Training SentencePiece model for {model_prefix}...")
        spm.SentencePieceTrainer.train(
            f'--input={input_file} --model_prefix={model_prefix} '
            f'--vocab_size={vocab_size} --model_type=bpe '
            f'--character_coverage=1.0 --shuffle_input_sentence=true '
            f'--bos_id=0 --eos_id=1 --unk_id=2 --pad_id=3 '
        )
        print(f"SentencePiece model saved to {model_prefix}.model")
    else:
        print(f"SentencePiece model {model_prefix}.model already exists.")

train_sentencepiece(TRAIN_VI, MODEL_PREFIX_VI, SRC_VOCAB_SIZE_SPM)
train_sentencepiece(TRAIN_LO, MODEL_PREFIX_LO, TGT_VOCAB_SIZE_SPM)

sp_vi = spm.SentencePieceProcessor()
sp_vi.load(MODEL_PREFIX_VI + '.model')
sp_lo = spm.SentencePieceProcessor()
sp_lo.load(MODEL_PREFIX_LO + '.model')

SRC_VOCAB_SIZE_ACTUAL = sp_vi.get_piece_size()
TGT_VOCAB_SIZE_ACTUAL = sp_lo.get_piece_size()
PAD_IDX = sp_vi.pad_id()
BOS_IDX = sp_vi.bos_id()
EOS_IDX = sp_vi.eos_id()
UNK_IDX = sp_vi.unk_id()

print(f"Actual Vocab Sizes: VI={SRC_VOCAB_SIZE_ACTUAL}, LO={TGT_VOCAB_SIZE_ACTUAL}")
print(f"Special IDs: PAD={PAD_IDX}, BOS={BOS_IDX}, EOS={EOS_IDX}, UNK={UNK_IDX}")
assert sp_vi.pad_id() == sp_lo.pad_id() and \
       sp_vi.bos_id() == sp_lo.bos_id() and \
       sp_vi.eos_id() == sp_lo.eos_id(), "Special token IDs mismatch!"

# --- Dataset and DataLoader ---
class TranslationDataset(Dataset):
    def __init__(self, src_file, trg_file, sp_src, sp_trg):
        self.sp_src = sp_src
        self.sp_trg = sp_trg
        print(f"Loading data from {src_file} and {trg_file}...")
        with open(src_file, 'r', encoding='utf-8') as f_src, \
             open(trg_file, 'r', encoding='utf-8') as f_trg:
            self.src_sents = [line.strip() for line in f_src]
            self.trg_sents = [line.strip() for line in f_trg]
        assert len(self.src_sents) == len(self.trg_sents), "Mismatch in number of sentences."
        print(f"Loaded {len(self.src_sents)} sentence pairs.")

    def __len__(self):
        return len(self.src_sents)

    def __getitem__(self, idx):
        src_text = self.src_sents[idx]
        trg_text = self.trg_sents[idx]
        src_tokens = [BOS_IDX] + self.sp_src.encode_as_ids(src_text) + [EOS_IDX]
        trg_tokens = [BOS_IDX] + self.sp_trg.encode_as_ids(trg_text) + [EOS_IDX]
        return torch.tensor(src_tokens, dtype=torch.long), torch.tensor(trg_tokens, dtype=torch.long)

def generate_batch(data_batch):
    src_batch, trg_batch = [], []
    for (src_item, trg_item) in data_batch:
        src_batch.append(src_item)
        trg_batch.append(trg_item)
    src_batch_padded = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    trg_batch_padded = pad_sequence(trg_batch, batch_first=True, padding_value=PAD_IDX)
    return src_batch_padded, trg_batch_padded

def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_padding_masks(src, tgt, pad_idx):
    src_padding_mask = (src == pad_idx)
    tgt_padding_mask = (tgt == pad_idx)
    return src_padding_mask, tgt_padding_mask

train_dataset = TranslationDataset(TRAIN_VI, TRAIN_LO, sp_vi, sp_lo)
dev_dataset = TranslationDataset(DEV_VI, DEV_LO, sp_vi, sp_lo)
test_dataset = TranslationDataset(TEST_VI, TEST_LO, sp_vi, sp_lo)

train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch, num_workers=2, pin_memory=True)
valid_iterator = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=generate_batch, num_workers=2, pin_memory=True)
test_iterator = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=generate_batch, num_workers=2, pin_memory=True)

# --- Model Architecture Components (Giữ nguyên như phiên bản trước đó) ---

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_seq_len, device='cpu', dtype=torch.float32)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len = seq_len
        t = torch.arange(self.max_seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(dtype)
        self.register_buffer("cos_cached", emb.cos()[:, None, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[:, None, :], persistent=False)

    def _rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q, k):
        seq_len_q = q.shape[-2]
        seq_len_k = k.shape[-2]
        if seq_len_q > self.max_seq_len or seq_len_k > self.max_seq_len or \
           self.cos_cached.device != q.device or self.cos_cached.dtype != q.dtype:
             self._set_cos_sin_cache(max(self.max_seq_len, seq_len_q, seq_len_k), device=q.device, dtype=q.dtype)
        cos_q = self.cos_cached[:seq_len_q, ...]
        sin_q = self.sin_cached[:seq_len_q, ...]
        cos_k = self.cos_cached[:seq_len_k, ...]
        sin_k = self.sin_cached[:seq_len_k, ...]
        if q.ndim == 4:
            cos_q = cos_q.transpose(0,1).unsqueeze(0)
            sin_q = sin_q.transpose(0,1).unsqueeze(0)
            cos_k = cos_k.transpose(0,1).unsqueeze(0)
            sin_k = sin_k.transpose(0,1).unsqueeze(0)
        q_embed = (q * cos_q) + (self._rotate_half(q) * sin_q)
        k_embed = (k * cos_k) + (self._rotate_half(k) * sin_k)
        return q_embed, k_embed

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=PAD_IDX)
        self.emb_size = emb_size
    def forward(self, tokens: torch.Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class RotaryMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, rotary_emb, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.rotary_emb = rotary_emb
        if self.rotary_emb.dim != self.head_dim:
            print(f"Warning: RoPE dim ({self.rotary_emb.dim}) != head_dim ({self.head_dim}). Re-initializing RoPE.")
            self.rotary_emb = RotaryEmbedding(dim=self.head_dim, max_seq_len=self.rotary_emb.max_seq_len)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        batch_size, tgt_len, _ = query.shape
        _, src_len, _ = key.shape
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = self.rotary_emb(q, k)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if attn_mask is not None:
            attn_mask = attn_mask.to(dtype=attention_scores.dtype, device=attention_scores.device)
            attention_scores = attention_scores + attn_mask
        if key_padding_mask is not None:
             key_padding_mask = key_padding_mask.to(device=attention_scores.device)
             attention_scores = attention_scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        output = torch.matmul(attention_probs, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.embed_dim)
        output = self.out_proj(output)
        return output

class GeGLU_FFN(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.w_gate = nn.Linear(d_model, dim_feedforward, bias=False)
        self.w_up = nn.Linear(d_model, dim_feedforward, bias=False)
        self.w_down = nn.Linear(dim_feedforward, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        gate = self.w_gate(x)
        up = self.w_up(x)
        fused_output = F.gelu(gate) * up
        fused_output = self.dropout(fused_output)
        output = self.w_down(fused_output)
        return output

class CustomTransformerEncoderLayerWithGeGLU_PreLN(nn.Module):
    def __init__(self, d_model, nhead, rotary_emb_instance, dim_feedforward, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = RotaryMultiHeadAttention(d_model, nhead, rotary_emb_instance, dropout=dropout)
        self.dropout_attn = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = GeGLU_FFN(d_model, dim_feedforward, dropout)
        self.dropout_ff = nn.Dropout(dropout)
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src_norm = self.norm1(src)
        attn_output = self.self_attn(src_norm, src_norm, src_norm, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout_attn(attn_output)
        src_norm = self.norm2(src)
        ff_output = self.feed_forward(src_norm)
        src = src + self.dropout_ff(ff_output)
        return src

class CustomTransformerDecoderLayerWithGeGLU_PreLN(nn.Module):
    def __init__(self, d_model, nhead, rotary_emb_instance, dim_feedforward, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = RotaryMultiHeadAttention(d_model, nhead, rotary_emb_instance, dropout=dropout)
        self.dropout_self_attn = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.multihead_attn = RotaryMultiHeadAttention(d_model, nhead, rotary_emb_instance, dropout=dropout)
        self.dropout_cross_attn = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.feed_forward = GeGLU_FFN(d_model, dim_feedforward, dropout)
        self.dropout_ff = nn.Dropout(dropout)
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt_norm = self.norm1(tgt)
        self_attn_output = self.self_attn(tgt_norm, tgt_norm, tgt_norm, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout_self_attn(self_attn_output)
        tgt_norm = self.norm2(tgt)
        cross_attn_output = self.multihead_attn(tgt_norm, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout_cross_attn(cross_attn_output)
        tgt_norm = self.norm3(tgt)
        ff_output = self.feed_forward(tgt_norm)
        tgt = tgt + self.dropout_ff(ff_output)
        return tgt

class CustomTransformerEncoder(nn.Module):
    def __init__(self, layer_creator_fn, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([layer_creator_fn() for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output

class CustomTransformerDecoder(nn.Module):
    def __init__(self, layer_creator_fn, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([layer_creator_fn() for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt
        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output

class Seq2SeqTransformerRoPEGeGLU_PreLN(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int,
                 dropout: float = 0.1,
                 max_seq_len_rope: int = 2048,
                 tie_weights: bool = True):
        super().__init__()
        self.emb_size = emb_size
        head_dim = emb_size // nhead
        if emb_size % nhead != 0:
            raise ValueError(f"emb_size ({emb_size}) must be divisible by nhead ({nhead})")
        self.rotary_emb = RotaryEmbedding(dim=head_dim, max_seq_len=max_seq_len_rope)
        encoder_layer_creator = lambda: CustomTransformerEncoderLayerWithGeGLU_PreLN(
            d_model=emb_size, nhead=nhead, rotary_emb_instance=self.rotary_emb,
            dim_feedforward=dim_feedforward, dropout=dropout
        )
        decoder_layer_creator = lambda: CustomTransformerDecoderLayerWithGeGLU_PreLN(
            d_model=emb_size, nhead=nhead, rotary_emb_instance=self.rotary_emb,
            dim_feedforward=dim_feedforward, dropout=dropout
        )
        encoder_norm = nn.LayerNorm(emb_size)
        self.encoder = CustomTransformerEncoder(encoder_layer_creator, num_encoder_layers, encoder_norm)
        decoder_norm = nn.LayerNorm(emb_size)
        self.decoder = CustomTransformerDecoder(decoder_layer_creator, num_decoder_layers, decoder_norm)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.dropout_emb = nn.Dropout(dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        if tie_weights:
            print("Applying weight tying between target embedding and generator.")
            self.generator.weight = self.tgt_tok_emb.embedding.weight
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
             if p.dim() > 1 and 'embedding' not in name and \
                not (hasattr(self.generator, 'weight') and p is self.generator.weight and \
                     hasattr(self.tgt_tok_emb, 'embedding') and self.generator.weight is self.tgt_tok_emb.embedding.weight):
                nn.init.xavier_uniform_(p)
             elif 'embedding.weight' in name and p.dim() > 1:
                 nn.init.normal_(p, mean=0, std=self.emb_size ** -0.5)

    def forward(self, src, trg, causal_tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        src_emb = self.dropout_emb(self.src_tok_emb(src))
        tgt_emb = self.dropout_emb(self.tgt_tok_emb(trg))
        memory = self.encoder(src_emb, mask=None, src_key_padding_mask=src_padding_mask)
        outs = self.decoder(tgt_emb, memory,
                            tgt_mask=causal_tgt_mask, memory_mask=None,
                            tgt_key_padding_mask=tgt_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: torch.Tensor, src_padding_mask: torch.Tensor):
        src_emb = self.dropout_emb(self.src_tok_emb(src))
        return self.encoder(src_emb, mask=None, src_key_padding_mask=src_padding_mask)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor,
               causal_tgt_mask: torch.Tensor,
               tgt_padding_mask: torch.Tensor,
               memory_key_padding_mask: torch.Tensor):
        tgt_emb = self.dropout_emb(self.tgt_tok_emb(tgt))
        return self.decoder(tgt_emb, memory,
                            tgt_mask=causal_tgt_mask, memory_mask=None,
                            tgt_key_padding_mask=tgt_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask)

# --- Training & Evaluation Components ---
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=LABEL_SMOOTHING)
model = Seq2SeqTransformerRoPEGeGLU_PreLN(
    NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD,
    SRC_VOCAB_SIZE_ACTUAL, TGT_VOCAB_SIZE_ACTUAL, FFN_HID_DIM_GEGLU, DROPOUT,
    max_seq_len_rope=MAX_SEQ_LEN_ROPE, tie_weights=True
)
model = model.to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'The model has {count_parameters(model):,} trainable parameters')

# Sử dụng learning rate cố định
optimizer = optim.AdamW(model.parameters(), lr=FIXED_LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01)
# Không cần scheduler nữa
# global_step_count = 0 # Vẫn có thể giữ để theo dõi tổng số bước

def train_epoch(model, iterator, optimizer, criterion, clip, current_epoch): # Bỏ scheduler khỏi tham số
    # global global_step_count # Không cần thiết nếu không dùng scheduler
    model.train()
    epoch_loss = 0
    pbar = tqdm(iterator, desc=f"Training Epoch {current_epoch:02d}", leave=False)

    for i, (src, tgt) in enumerate(pbar): # Thêm enumerate để lấy index
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_input = tgt[:, :-1]

        src_padding_mask, tgt_padding_mask_for_input = create_padding_masks(src, tgt_input, PAD_IDX)
        tgt_seq_len = tgt_input.shape[1]
        causal_tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)

        logits = model(src, tgt_input,
                       causal_tgt_mask=causal_tgt_mask,
                       src_padding_mask=src_padding_mask,
                       tgt_padding_mask=tgt_padding_mask_for_input,
                       memory_key_padding_mask=src_padding_mask)

        optimizer.zero_grad()
        tgt_out = tgt[:, 1:]
        loss = criterion(logits.reshape(-1, TGT_VOCAB_SIZE_ACTUAL), tgt_out.reshape(-1))

        if torch.isnan(loss) or torch.isinf(loss):
             print(f"Warning: NaN or Inf loss detected at epoch {current_epoch}, batch {i}! Skipping update.")
             if torch.cuda.is_available(): torch.cuda.empty_cache()
             continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        # Không còn scheduler.step()
        # global_step_count += 1

        epoch_loss += loss.item()
        current_lr = optimizer.param_groups[0]['lr'] # LR sẽ không đổi
        pbar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{current_lr:.3e}") # Bỏ step

    return epoch_loss / max(1, len(iterator))


# --- Greedy Decoding (Giữ nguyên như phiên bản trước) ---
@torch.no_grad()
def greedy_decode_batch(model, src, src_padding_mask, max_len,
                        bos_idx, eos_idx, pad_idx, device):
    model.eval()
    batch_size = src.size(0)
    memory = model.encode(src, src_padding_mask)
    memory_key_padding_mask = src_padding_mask
    ys = torch.ones(batch_size, 1).fill_(bos_idx).type(torch.long).to(device)
    finished_sents = torch.zeros(batch_size, dtype=torch.bool, device=device)
    for _ in range(max_len - 1):
        tgt_seq_len = ys.size(1)
        causal_tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
        tgt_padding_mask = (ys == pad_idx)
        out = model.decode(ys, memory, causal_tgt_mask, tgt_padding_mask, memory_key_padding_mask)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        just_finished = (next_word == eos_idx)
        finished_sents = finished_sents | just_finished
        next_word = next_word.masked_fill(finished_sents & (ys[:, -1] != pad_idx), eos_idx)
        next_word = next_word.masked_fill(finished_sents & (ys[:, -1] == eos_idx), pad_idx)
        ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)
        if finished_sents.all():
            break
    return ys.cpu().tolist()


# --- Evaluation Function (Giữ nguyên như phiên bản trước) ---
@torch.no_grad()
def evaluate(model, iterator, criterion, sp_trg_tokenizer, max_len_eval):
    model.eval()
    epoch_loss = 0
    all_refs_text_for_bleu = []
    all_hyps_text_for_bleu = []
    pbar = tqdm(iterator, desc="Evaluating", leave=False)
    for src, tgt in pbar:
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_input = tgt[:, :-1]
        src_padding_mask_loss, tgt_padding_mask_for_input = create_padding_masks(src, tgt_input, PAD_IDX)
        tgt_seq_len = tgt_input.shape[1]
        causal_tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
        try:
            logits = model(src, tgt_input,
                           causal_tgt_mask=causal_tgt_mask,
                           src_padding_mask=src_padding_mask_loss,
                           tgt_padding_mask=tgt_padding_mask_for_input,
                           memory_key_padding_mask=src_padding_mask_loss)
            tgt_out_for_loss = tgt[:, 1:]
            loss = criterion(logits.reshape(-1, TGT_VOCAB_SIZE_ACTUAL), tgt_out_for_loss.reshape(-1))
            epoch_loss += loss.item()
        except Exception as e:
            print(f"Error during loss calculation in eval: {e}")
        src_padding_mask_decode = (src == PAD_IDX)
        hyps_ids_batch = greedy_decode_batch(
            model, src, src_padding_mask_decode,
            max_len=max_len_eval,
            bos_idx=BOS_IDX, eos_idx=EOS_IDX, pad_idx=PAD_IDX,
            device=device
        )
        target_sents_ids_for_decode = tgt[:, 1:].cpu().tolist()
        for hyp_ids in hyps_ids_batch:
            cleaned_hyp_ids = [id_ for id_ in hyp_ids if id_ not in [BOS_IDX, EOS_IDX, PAD_IDX]]
            hyp_text = sp_trg_tokenizer.decode_ids(cleaned_hyp_ids)
            all_hyps_text_for_bleu.append(hyp_text)
        for ref_ids_list in target_sents_ids_for_decode:
            cleaned_ref_ids = [id_ for id_ in ref_ids_list if id_ not in [BOS_IDX, EOS_IDX, PAD_IDX]]
            ref_text = sp_trg_tokenizer.decode_ids(cleaned_ref_ids)
            all_refs_text_for_bleu.append([ref_text])
    avg_loss = epoch_loss / max(1, len(iterator))
    bleu = sacrebleu.corpus_bleu(all_hyps_text_for_bleu, list(zip(*all_refs_text_for_bleu)))
    return avg_loss, bleu.score


# --- Inference Function (Giữ nguyên như phiên bản trước) ---
def translate_sentence_greedy(sentence: str,
                              model_final: Seq2SeqTransformerRoPEGeGLU_PreLN,
                              sp_src: spm.SentencePieceProcessor,
                              sp_trg: spm.SentencePieceProcessor,
                              device: torch.device,
                              max_len: int = 50):
    model_final.eval()
    src_tokens = [BOS_IDX] + sp_src.encode_as_ids(sentence) + [EOS_IDX]
    src_tensor = torch.LongTensor(src_tokens).unsqueeze(0).to(device)
    src_padding_mask = (src_tensor == PAD_IDX)
    output_ids_batch = greedy_decode_batch(
        model_final, src_tensor, src_padding_mask,
        max_len=max_len,
        bos_idx=BOS_IDX, eos_idx=EOS_IDX, pad_idx=PAD_IDX,
        device=device
    )
    output_ids = output_ids_batch[0]
    cleaned_tgt_ids = [i for i in output_ids if i not in [BOS_IDX, EOS_IDX, PAD_IDX]]
    return sp_trg.decode_ids(cleaned_tgt_ids)

# --- Main Training Loop ---
best_valid_loss = float('inf')
train_losses = []
valid_losses = []
bleu_scores = []

print(f"Starting Training ({NUM_EPOCHS} epochs)...")
print(f"Device: {device}, Batch Size: {BATCH_SIZE}, Fixed LR: {FIXED_LEARNING_RATE}, Label Smoothing: {LABEL_SMOOTHING}")

for epoch in range(1, NUM_EPOCHS + 1):
    start_time = time.time()

    train_loss = train_epoch(model, train_iterator, optimizer, criterion, CLIP, epoch) # Bỏ scheduler
    valid_loss, bleu_score_val = evaluate(model, valid_iterator, criterion, sp_lo, MAX_LEN_EVAL)

    end_time = time.time()
    epoch_mins = int((end_time - start_time) / 60)
    epoch_secs = int((end_time - start_time) % 60)
    current_lr = optimizer.param_groups[0]['lr'] # Sẽ luôn là FIXED_LEARNING_RATE

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    bleu_scores.append(bleu_score_val)

    print(f'Epoch: {epoch:02}/{NUM_EPOCHS} | Time: {epoch_mins}m {epoch_secs}s | LR: {current_lr:.3e}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(min(train_loss, 700)):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(min(valid_loss, 700)):7.3f}')
    print(f'\t Val. BLEU: {bleu_score_val:.2f}')

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"\tBest model saved to {MODEL_SAVE_PATH} (Val Loss: {best_valid_loss:.3f})")

print("Training finished.")

# --- Final Evaluation on Test Set (Giữ nguyên) ---
try:
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    print(f"Loaded best model from {MODEL_SAVE_PATH} for final evaluation.")
    test_loss, test_bleu = evaluate(model, test_iterator, criterion, sp_lo, MAX_LEN_EVAL)
    print('-' * 30)
    print('|      FINAL TEST SET RESULTS      |')
    print('-' * 30)
    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(min(test_loss, 700)):7.3f} |')
    print(f'| Test BLEU: {test_bleu:.2f} |')
    print('-' * 30)
except FileNotFoundError:
    print(f"ERROR: Best model file not found at {MODEL_SAVE_PATH}. Skipping final evaluation.")
except Exception as e:
     print(f"An error occurred during final evaluation: {e}")

