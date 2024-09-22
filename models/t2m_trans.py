import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical
import models.pos_encoding as pos_encoding
from exit.utils import cosine_schedule, uniform, top_k, gumbel_sample, top_p, top_k_top_p_filtering, gumbel_noise, generate_src_mask, show
from tqdm import tqdm
from einops import rearrange, repeat
from exit import utils
from math import ceil
import random


def sample_next_token_categorical(logits, top_k=1, temperature=1.0):
    """
    top_k=1, means greedy sampling
    top_k=0, means the tokens will be predicted from all indices based on their probabilities
    """

    logits /= max(temperature, 1e-10)
    
    if top_k > 0:
        top_k_values, top_k_indices = torch.topk(logits, top_k)
        logits = torch.full_like(logits, float('-inf'))
        logits.scatter_(-1, top_k_indices, top_k_values)

    dist = Categorical(logits=logits)
    next_token = dist.sample()

    return next_token


def create_causal_mask(seq_len, bs):
    """
    Create a causal mask for autoregressive training.
    
    :param seq_len: Length of the sequence
    :return: A (seq_len, seq_len) mask tensor with 0s in the upper triangular part and 1s in the lower triangular part.
    """
    mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool)).unsqueeze(0).repeat(bs, 1, 1)
    return mask


def random_value(p=0.01, value_1=[0, 0.5], value_2=[0.5, 1]):
    # Generate a random number between 0 and 1
    random_num = torch.rand(1).item()
    # Return value_1 if random number is less than the prob, otherwise return value_2
    return value_1 if random_num < p else value_2


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb



class Attention(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1, args=None):
        super().__init__()
        assert embed_dim % 8 == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.n_head = n_head

    def forward(self, x, src_mask):
        B, T, C = x.size() 

        ### calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        ### Original Attention: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if src_mask is not None:
            att[~src_mask] = float('-inf')
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # ## Flash Attention
        # y = F.scaled_dot_product_attention(q, k, v, src_mask)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        ### output projection
        y = self.resid_drop(self.proj(y))
        return y


class RelativeSelfAttention(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1, args=None):
        super().__init__()
        assert embed_dim % 8 == 0
        # key, query, value projections for all heads
        self.head_dim = embed_dim // n_head

        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.max_relative_position = args.block_size + args.time_cond

        self.relative_position_k = RelativePosition(self.head_dim, self.max_relative_position)
        self.relative_position_v = RelativePosition(self.head_dim, self.max_relative_position)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.n_head = n_head

    def forward(self, x, src_mask=None):
        B, T, C = x.size() 

        ### calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x  )
        q = self.query(x)
        v = self.value(x)

        r_q1 = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)        
        r_k1 = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        attn1 = torch.matmul(r_q1, r_k1.transpose(3, 2))

        r_q2 = q.permute(1, 0, 2).contiguous().view(T, B*self.n_head, self.head_dim)
        r_k2 = self.relative_position_k(T, T)
        attn2 = torch.matmul(r_q2, r_k2.transpose(1, 2)).transpose(0, 1)
        attn2 = attn2.contiguous().view(B, self.n_head, T, T)
        attn = (attn1 + attn2) * (1.0 / math.sqrt(k.size(-1)))

        if src_mask is not None:
            attn[~src_mask] = float('-inf')

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        #attn = [batch size, n heads, query len, key len]
        r_v1 = v.view(B, -1, self.n_head, self.head_dim).transpose(1, 2)
        weight1 = torch.matmul(attn, r_v1)
        r_v2 = self.relative_position_v(T, T)
        weight2 = attn.permute(2, 0, 1, 3).contiguous().view(T, B*self.n_head, T)
        weight2 = torch.matmul(weight2, r_v2)
        weight2 = weight2.transpose(0, 1).contiguous().view(B, self.n_head, T, self.head_dim)

        x = weight1 + weight2
        x = x.transpose(1, 2).contiguous()
        x = x.view(B, -1, C)

        ### output projection
        x = self.resid_drop(self.proj(x))

        return x


class RelativeCrossAttention(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1, args=None):
        super().__init__()
        assert embed_dim % 8 == 0
        self.head_dim = embed_dim // n_head
        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.max_relative_position = 77

        self.relative_position_k = RelativePosition(self.head_dim, self.max_relative_position)
        self.relative_position_v = RelativePosition(self.head_dim, self.max_relative_position)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(embed_dim, embed_dim)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.n_head = n_head

    def forward(self, x, cond, text_mask=None):
        B, len_q, C = x.size()
        B, len_k, D = cond.size()
        len_v = len_k

        ### calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(cond)
        q = self.query(x)
        v = self.value(cond)

        r_q1 = q.view(B, len_q, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)        
        r_k1 = k.view(B, len_k, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        attn1 = torch.matmul(r_q1, r_k1.transpose(3, 2))

        r_q2 = q.permute(1, 0, 2).contiguous().view(len_q, B*self.n_head, self.head_dim)
        r_k2 = self.relative_position_k(len_q, len_k)
        attn2 = torch.matmul(r_q2, r_k2.transpose(1, 2)).transpose(0, 1)
        attn2 = attn2.contiguous().view(B, self.n_head, len_q, len_k)
        attn = (attn1 + attn2) * (1.0 / math.sqrt(k.size(-1)))

        if text_mask is not None:
            attn[~text_mask] = float('-inf')
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        #attn = [batch size, n heads, query len, key len]
        r_v1 = v.view(B, -1, self.n_head, self.head_dim).transpose(1, 2)
        weight1 = torch.matmul(attn, r_v1)
        r_v2 = self.relative_position_v(len_q, len_v)
        weight2 = attn.permute(2, 0, 1, 3).contiguous().view(len_q, B*self.n_head, len_k)
        weight2 = torch.matmul(weight2, r_v2)
        weight2 = weight2.transpose(0, 1).contiguous().view(B, self.n_head, len_q, self.head_dim)

        x = weight1 + weight2
        
        #x = [batch size, n heads, query len, head dim]
        x = x.transpose(1, 2).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        x = x.view(B, -1, C)
        
        #x = [batch size, query len, hid dim]
        x = self.resid_drop(self.proj(x))
        
        return x



class RelativePosition(nn.Module):
    """ https://github.com/evelinehong/Transformer_Relative_Position_PyTorch """
    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat).cuda()
        embeddings = self.embeddings_table[final_mat].cuda()

        return embeddings

    

class Block(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1, fc_rate=4, args=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = RelativeSelfAttention(embed_dim, block_size, n_head, drop_out_rate, args=args) \
            if args.use_relative_position else Attention(embed_dim, block_size, n_head, drop_out_rate)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, fc_rate * embed_dim),
            nn.GELU(),
            nn.Linear(fc_rate * embed_dim, embed_dim),
            nn.Dropout(drop_out_rate),
        )

    def forward(self, x, src_mask=None):
        x = x + self.attn(self.ln1(x), src_mask)
        x = x + self.mlp(self.ln2(x))
        return x
    
class CrossAttention(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1):
        super().__init__()
        assert embed_dim % 8 == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(embed_dim, embed_dim)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(block_size, 77)).view(1, 1, block_size, 77))
        self.n_head = n_head

    def forward(self, x, word_emb, text_mask=None):
        B, T, C = x.size()
        B, N, D = word_emb.size()

        ### calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(word_emb).view(B, N, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(word_emb).view(B, N, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        ## Original Attention: (B, nh, T, hs) x (B, nh, hs, N) -> (B, nh, T, N)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if text_mask is not None:
            att[~text_mask] = float('-inf')
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, N) x (B, nh, N, hs) -> (B, nh, T, hs)

        # ### Flash Attention
        # y = F.scaled_dot_product_attention(q, k, v, text_mask)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        ### output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block_crossatt(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1, fc_rate=4, args=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ln3 = nn.LayerNorm(embed_dim)
        self.attn = RelativeCrossAttention(embed_dim, block_size, n_head, drop_out_rate, args=args) \
            if args.use_relative_position else CrossAttention(embed_dim, block_size, n_head, drop_out_rate)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, fc_rate * embed_dim),
            nn.GELU(),
            nn.Linear(fc_rate * embed_dim, embed_dim),
            nn.Dropout(drop_out_rate),
        )

    def forward(self, x, word_emb, text_mask=None):
        x = x + self.attn(self.ln1(x), self.ln3(word_emb), text_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class CrossCondTransBase(nn.Module):
    def __init__(self, vqvae, num_vq=1024, embed_dim=512, clip_dim=512, num_layers=2,
                 num_local_layer = 1,n_head=8, drop_out_rate=0.1, fc_rate=4, args=None):
        super().__init__()
        self.vqvae = vqvae
        self.args = args

        n_embed_base = 2 + 50
        self.learn_tok_emb = nn.Embedding(n_embed_base, self.vqvae.vqvae.code_dim) # Maskbook
        self.to_emb = nn.Linear(self.vqvae.vqvae.code_dim, embed_dim)

        if args.time_cond:
            self.t_embedder = TimestepEmbedder(args.embed_dim_gpt)
            self.l_embedder = TimestepEmbedder(args.embed_dim_gpt)
        
        self.cond_emb = nn.Linear(clip_dim, embed_dim)

        if not args.use_relative_position:
            self.pos_embedding = nn.Embedding(args.block_size, embed_dim)
            self.pos_embed = pos_encoding.PositionEmbedding(args.block_size, embed_dim, 0.0, False)
        
        self.drop = nn.Dropout(drop_out_rate)
        # transformer block
        
        self.blocks = nn.Sequential(*[Block(embed_dim, args.block_size, n_head, drop_out_rate, fc_rate, args=args) for _ in range(num_layers-num_local_layer)])
        self.num_local_layer = num_local_layer
        if num_local_layer > 0:
            self.word_emb = nn.Linear(clip_dim, embed_dim)
            self.cross_att = nn.Sequential(*[Block_crossatt(embed_dim, args.block_size, n_head, drop_out_rate, fc_rate, args=args) for _ in range(num_local_layer)])

        self.args = args
        self.block_size = args.block_size
        self.num_cond = args.num_cond
        self.end_id = self.vqvae.end_id
        self.pad_id = self.vqvae.pad_id
        self.mask_id = self.vqvae.mask_id

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_token_embeddings(self, idx, *args, **kwargs):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        # forward the Trans model
        not_learn_idx = idx < self.vqvae.vqvae.num_code
        learn_idx = ~not_learn_idx
        
        token_embeddings = torch.empty((*idx.shape, self.vqvae.vqvae.code_dim), device=idx.device)
        token_embeddings[not_learn_idx] = self.vqvae.vqvae.quantizer.dequantize(idx[not_learn_idx]).requires_grad_(False) 
        token_embeddings[learn_idx] = self.learn_tok_emb(idx[learn_idx]-self.vqvae.vqvae.num_code)
        token_embeddings = self.to_emb(token_embeddings)
        if not self.args.use_relative_position:
            token_embeddings = self.pos_embed(token_embeddings)

        return token_embeddings

    def get_corruption_rate(self, idx):
        num_mask = (idx >= self.mask_id).sum(-1)
        lengthes = (idx==self.end_id).int().argmax(-1)
        r = self.t_embedder(num_mask) + self.l_embedder(lengthes)
        return r.unsqueeze(1)

    def forward(self, idx, clip_feature, src_mask, word_emb, text_mask=None,
                                time=None):
        token_embeddings = self.get_token_embeddings(idx, clip_feature)

        if self.num_local_layer > 0:
            word_emb = self.word_emb(word_emb)

            for module in self.cross_att:
                token_embeddings = module(token_embeddings, word_emb, text_mask)
        
        if self.args.time_cond:
            r = self.get_corruption_rate(idx)
            token_embeddings = torch.cat([r, token_embeddings], dim=1)

        clip_feature = self.cond_emb(clip_feature).unsqueeze(1)
        token_embeddings = torch.cat([clip_feature, token_embeddings], dim=1)

        if not self.args.use_relative_position:
            token_embeddings = self.pos_embed(token_embeddings)

        for block in self.blocks:
            token_embeddings = block(token_embeddings, src_mask)

        return token_embeddings

class CrossCondTransHead(nn.Module):

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4,
                args=None):
        super().__init__()

        self.blocks = nn.Sequential(*[Block(embed_dim, args.block_size, n_head, drop_out_rate, fc_rate, args=args) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_vq, bias=False)
        self.block_size = args.block_size

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, src_mask):
        for block in self.blocks:
            x = block(x, src_mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits



class Text2Motion_Transformer(nn.Module):
    def __init__(self, vqvae,num_vq=1024, embed_dim=512, clip_dim=512,
                 num_layers=2, num_local_layer=0,
                 n_head=8, drop_out_rate=0.1, fc_rate=4, args=None):
        
        super().__init__()
        args.block_size = args.block_size + args.time_cond # 50 + 1 (for clip_feature always) + 1 (for time optional)
        args.num_cond = 1 + args.time_cond # 1 (for clip_feature) + 1 (for time)

        self.trans_base = CrossCondTransBase(vqvae, num_vq, embed_dim, clip_dim, num_layers, num_local_layer, n_head, drop_out_rate, fc_rate, args)
        self.trans_head = CrossCondTransHead(num_vq, embed_dim, num_layers, n_head, drop_out_rate, fc_rate, args)

        self.args = args
        self.n_head = n_head
    
        self.block_size = args.block_size
        self.num_vq = num_vq
        self.time_cond = args.time_cond
        self.num_cond = args.num_cond

        self.mask_id = vqvae.mask_id
        self.pad_id  = vqvae.pad_id
        self.end_id  = vqvae.end_id

        self.max_length = self.args.max_length

        if   self.args.sampling_type == "CBS":  self.sample = self.Confidence_Based_Sampling
        elif self.args.sampling_type == "OAAS": self.sample = self.Order_Agnostic_Autoregressive_Sampling
        elif self.args.sampling_type == "RS":   self.sample = self.Random_Sampling
        
        self.mask_scheduler = utils.__dict__[args.mask_scheduler + '_schedule']
        if args.max_steps_generation == 49:
            self.mask_scheduler = utils.__dict__['linear_schedule']
        
    def get_block_size(self):
        return self.block_size

    def forward(self, *args, type='forward', **kwargs):
        '''type=[forward, sample]'''
        if   type=='forward':  return self.forward_function(*args, **kwargs)
        elif type=='sample':   return self.sample(*args, **kwargs)
        else: raise ValueError(f'Unknown "{type}" type')
        
    def get_attn_mask(self, src_mask, att_txt=None):
        if src_mask.ndim >= 4: return src_mask

        if self.time_cond:
            att_time = torch.tensor([[True]]*src_mask.shape[0]).to(src_mask.device)
            src_mask = torch.cat([att_time, src_mask],  dim=1)

        if att_txt is None:
            att_txt = torch.tensor([[True]]*src_mask.shape[0]).to(src_mask.device)
            src_mask = torch.cat([att_txt, src_mask],  dim=1)

        B, T = src_mask.shape
        src_mask = src_mask.view(B, 1, 1, T).repeat(1, self.n_head, T, 1)
        return src_mask

    def forward_function(self, idxs, clip_feature, src_mask=None,
                         att_txt=None, word_emb=None, text_mask=None,):
        src_mask  = self.get_attn_mask(src_mask, att_txt)
        
        x = self.trans_base(idxs, clip_feature, src_mask, word_emb, text_mask=text_mask)
        x = self.trans_head(x, src_mask)

        return x[:, self.num_cond:]


    def get_text_mask(self, text_mask, src_mask):
        if text_mask is None: return text_mask
        T = src_mask.shape[-1] - self.num_cond
        B, N = text_mask.shape
        text_mask = text_mask.view(B, 1, 1, N).repeat(1, self.n_head, T, 1)
        return text_mask

    def get_randperm(self, bs, T, device, seq_mask_with_end, seq_mask_no_end, args):
        rand = torch.rand((bs, T), device=device)
        rand -= seq_mask_no_end.int()
        randperm = rand.argsort(dim = -1)

        if not self.training and args.start_ids_in_sampling == 'ARANGED':
            randperm = torch.arange(T).unsqueeze(0).repeat(bs, 1).to(device)

        return randperm

    def permuted_corruption(self, target=None, T=None, m_tokens_len=None, args=None, eval_edit=False, *argsv, **kwargs):
        # target = torch.tensor([[45, 80, 10, 3, 50, 92, 11, 15, 13, self.end_id], [10, 32, 25, 9, 15, 25, 3, 53, self.end_id, self.pad_id]]).to(args.device)
        # m_tokens_len = torch.tensor([9, 8]).to(args.device)
        # m_tokens, T = target, 10

        device = args.device
        bs = m_tokens_len.shape[0]

        if self.training: p_a, p_b = random_value(p=0.1, value_1=[0, 0.5], value_2=[0.5, 1])
        else:             p_a, p_b = 1, 1

        seq_mask_no_end   = generate_src_mask(T, m_tokens_len).to(device)
        seq_mask_with_end = generate_src_mask(T, m_tokens_len+1).to(device)

        if self.training:
            # Random replacement
            pkeep = torch.zeros((bs)).float().uniform_(0.6, 1).to(device).unsqueeze(-1)
            mask = torch.bernoulli(pkeep * torch.ones(target.shape, device=device))
            mask = torch.logical_or(mask, ~seq_mask_no_end).int()
            r_indices = torch.randint_like(target, args.nb_code)
            input_indices = mask * target + (1-mask) * r_indices

        # Masking target (Building input)
        rand_mask_probs = torch.zeros(bs, device=device).float().uniform_(p_a, p_b)
        num_token_masked = (m_tokens_len * rand_mask_probs).round().clamp(min = 1)

        randperm_mask = self.get_randperm(bs, T, device, seq_mask_with_end, seq_mask_no_end, args)
        mask_token = randperm_mask < rearrange(num_token_masked, 'b -> b 1')

        if eval_edit:
            mask_token = torch.full_like(target, fill_value=False).bool().to(device)
            mask_token[target==-1] = True
            randperm = self.get_randperm(bs, T, device, seq_mask_with_end, mask_token, args).argsort(-1)
        else:
            randperm = self.get_randperm(bs, T, device, seq_mask_with_end, seq_mask_no_end, args)

        # if self.training and args.use_aranged_perm: # This should be after building mask_token
        #     all_indices = torch.randperm(bs).to(device)
        #     selected_indices = all_indices[:round(args.aranged_perm_rate*bs)]
        #     randperm[selected_indices] = torch.arange(T).to(device)
        
        # Build Attention Mask
        if args.z_0_attend_to_all:
            permuted_causal_mask = randperm[:, None, :] >= randperm[..., None]
        else:
            permuted_causal_mask = randperm[:, None, :] <= randperm[..., None] # the mask 
        # show([permuted_causal_mask], randperm=randperm, b=0)

        bidirectional_mask_c = ~mask_token.unsqueeze(1).repeat(1, T, 1)
        bidirectional_mask = bidirectional_mask_c | bidirectional_mask_c.transpose(1, 2)
        # show([permuted_causal_mask, bidirectional_mask], randperm=randperm, b=0)

        hybrid_mask = permuted_causal_mask | bidirectional_mask
        # b = 0
        # show([hybrid_mask, bidirectional_mask], randperm=randperm, b=b)
        
        nz = torch.nonzero(mask_token == False)
        if args.unmasked_tokens_not_attend_to_mask_tokens:
            hybrid_mask[nz[:, 0], nz[:, 1]] = ~mask_token[nz[:, 0]]
        # show([hybrid_mask, bidirectional_mask], randperm=randperm, b=0)

        if self.training:
            masked_input_indices = torch.where(mask_token, randperm + self.mask_id, input_indices) # MASK_ID=START_ID
        else:
            masked_input_indices = randperm + self.mask_id # self.mask_id=start_mask_id
            masked_input_indices[~seq_mask_with_end] = self.pad_id
            masked_input_indices.scatter_(-1, m_tokens_len[..., None].long(), self.end_id)
        
        seq_mask_with_end_with_conds = generate_src_mask(T+self.num_cond, m_tokens_len+1+self.num_cond).to(device)
        seq_mask_with_end_with_conds_expanded = seq_mask_with_end_with_conds.unsqueeze(1).repeat(1, T+self.num_cond, 1)
        # show([seq_mask_with_end_with_conds_expanded])
        hybrid_mask = F.pad(hybrid_mask, (self.num_cond, 0, self.num_cond, 0), value=True) # Padding 1 True for clip_feature - another 1 for time (if used)
        # show([hybrid_mask])
        if args.unmasked_tokens_not_attend_to_mask_tokens:
            hybrid_mask[:, 0, self.num_cond:] = ~mask_token
        
        hybrid_mask[~seq_mask_with_end_with_conds_expanded] = False
        # show([hybrid_mask], b=0)
        # bidirectional_mask[~seq_mask_with_end_with_conds_expanded[:, self.num_cond:, self.num_cond:]] = False
        # b = 0
        # show([hybrid_mask[:, self.num_cond:, self.num_cond:], bidirectional_mask], randperm=randperm, b=b)
        hybrid_mask = hybrid_mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)
        # show([hybrid_mask[:, 0]], b=0)
        
        return masked_input_indices, hybrid_mask, seq_mask_no_end, seq_mask_with_end, mask_token, randperm, seq_mask_with_end_with_conds_expanded


    def Order_Agnostic_Autoregressive_Sampling(self, clip_feature, word_emb, m_length=None, if_test=False, rand_pos=True, CFG=-1,
                                     token_cond=None, max_steps = 10, *args, **kwargs):
        max_length = 49
        # max_length, self.block_size, max_steps = 9, 11, 5
        T = max_length + 1
        batch_size = clip_feature.shape[0]
        
        m_tokens_len = torch.ceil((m_length)/4).long() # m_tokens_len = torch.full_like(m_tokens_len, 9)
        src_token_mask = generate_src_mask(self.block_size-self.num_cond, m_tokens_len+1)
        src_token_mask_noend = generate_src_mask(self.block_size-self.num_cond, m_tokens_len)

        start_mask_id = self.mask_id

        if token_cond is not None:
            ids = token_cond.clone()
            ids[~src_token_mask_noend] = self.pad_id
            ids[~src_token_mask] = self.pad_id # [INFO] replace with pad id
            ids.scatter_(-1, m_tokens_len[..., None].long(), self.end_id) # [INFO] replace with end id
            mask_id, src_mask, *_, randperm, seq_mask_with_end_with_conds_expanded = \
                self.permuted_corruption(ids, T, m_tokens_len, args=self.args, eval_edit=True)
            token_cond[token_cond==-1] = mask_id[token_cond==-1]
            ids = token_cond.clone()
            num_token_cond = (ids>start_mask_id).sum(-1)
            mask_id = ids.clone()
        else:
            mask_id, src_mask, *_, randperm, seq_mask_with_end_with_conds_expanded = \
                self.permuted_corruption(T=T, m_tokens_len=m_tokens_len, args=self.args)
            ids = mask_id.clone()
        src_mask[torch.arange(batch_size), :, :, m_tokens_len+1] = True # Making end_token True

        rand_pos = self.args.rand_pos
        if rand_pos:
            top_k = self.args.top_k
            top_p = self.args.top_p
            temperature = self.args.temperature
        else:
            top_k = 1 # Because we have multinomial, if we do not want any randomness in sampling, top_k sould be 1
            temperature = 0
            top_p = 0

        INVALID = 1e5
        times = mask_id - start_mask_id
        times[times < 0] = INVALID

        sample_max_steps = torch.round(max_steps/max_length*m_tokens_len) + 1e-8

        if token_cond is not None:
            m_tokens_len_task = num_token_cond
            times[token_cond!=mask_id] = INVALID
        else:
            m_tokens_len_task = m_tokens_len

        for step in range(max_steps):

            timestep = torch.clip(step/(sample_max_steps), max=1)
            if len(m_tokens_len)==1 and step > 0 and torch.clip(step-1/(sample_max_steps), max=1).cpu().item() == timestep:
                break
            
            rand_mask_prob = self.mask_scheduler(timestep) # timestep #
            num_token_masked = (rand_mask_prob * m_tokens_len).long().clip(min=1)

            if token_cond is not None:
                num_token_masked = (rand_mask_prob * num_token_cond).long().clip(min=1)
                times[token_cond!=mask_id] = INVALID

            if self.args.z_0_attend_to_all:
                is_not_mask = (times < (m_tokens_len_task-num_token_masked).unsqueeze(-1))
            else:
                is_not_mask = (times >= num_token_masked.unsqueeze(-1))
            ids = torch.where(is_not_mask, ids, mask_id)

            if src_mask.ndim == 4: # We have to update src_mask to attend to new predicted tokens
                m = F.pad(ids < self.end_id, (self.num_cond, 0), value=False).unsqueeze(1).repeat(1, T+self.num_cond, 1)
                # m |= m.transpose(1, 2)
                m[~seq_mask_with_end_with_conds_expanded] = False
                # show([m], b=5)
                # show([m], b=5, randperm=randperm)
                m = m.unsqueeze(1).repeat(1, self.n_head, 1, 1)
                src_mask |= m
            # show([m[:, 0], src_mask[:, 0]], b=0)
            # show([m[:, 0, self.num_cond:, self.num_cond:], src_mask[:, 0, self.num_cond:, self.num_cond:]], randperm=randperm, b=0)
            
            logits = self.forward(ids, clip_feature, src_mask, word_emb=word_emb)

            if logits.shape[-1] > self.end_id: logits = logits[..., :self.end_id] # Here we sample with length, so no pred_end should be predicted

            filtered_logits = top_k_top_p_filtering(logits, top_k, top_p)
            probs = F.softmax(filtered_logits, dim=-1)
            pred_ids = torch.multinomial(probs.view(-1, logits.shape[-1]), 1).view(batch_size, T, 1).squeeze(-1)

            is_mask = ids >= start_mask_id
            ids = torch.where(is_mask, pred_ids, ids)

        return ids


    def Confidence_Based_Sampling(self, clip_feature, word_emb, m_length=None, if_test=False, CFG=-1,
                            token_cond=None, max_steps = 10, text_mask=None):
        max_length = 49
        T = max_length + 1
        batch_size = clip_feature.shape[0]

        shape = (batch_size, self.block_size - self.num_cond)
        topk_filter_thres = .9
        starting_temperature = 1.0
        # scores = torch.ones(shape, dtype = torch.float32, device = clip_feature.device)
        
        m_tokens_len = torch.ceil((m_length)/4).long()
        src_token_mask = generate_src_mask(self.block_size-self.num_cond, m_tokens_len+1)
        src_token_mask_noend = generate_src_mask(self.block_size-self.num_cond, m_tokens_len)
        scores = torch.where(~src_token_mask_noend, 1e5, 0.)

        start_mask_id = self.mask_id

        if token_cond is not None:
            ids = token_cond.clone()
            ids[~src_token_mask_noend] = self.pad_id
            ids[~src_token_mask] = self.pad_id # [INFO] replace with pad id
            ids.scatter_(-1, m_tokens_len[..., None].long(), self.end_id) # [INFO] replace with end id
            mask_id, src_mask, *_, randperm, seq_mask_with_end_with_conds_expanded = \
                self.permuted_corruption(ids, T, m_tokens_len, args=self.args, eval_edit=True)
            token_cond[token_cond==-1] = mask_id[token_cond==-1]
            ids = token_cond.clone()
            num_token_cond = (ids>start_mask_id).sum(-1)
        else:
            mask_id, src_mask, *_, randperm, seq_mask_with_end_with_conds_expanded = \
                self.permuted_corruption(T=T, m_tokens_len=m_tokens_len, args=self.args)
            ids = mask_id.clone()
        src_mask[torch.arange(batch_size), :, :, m_tokens_len+1] = True # Making end_token True
        # show([src_mask[:, 0]], b=0) # show([src_mask[:, 0, 1:, 1:]], randperm=randperm, b=2)

        sample_max_steps = torch.round(max_steps/max_length*m_tokens_len) + 1e-8

        if self.args.rand_pos: temperature = self.args.temperature
        else:                  temperature = 0

        for step in range(max_steps):

            timestep = torch.clip(step/(sample_max_steps), max=1)
            if len(m_tokens_len)==1 and step > 0 and torch.clip(step-1/(sample_max_steps), max=1).cpu().item() == timestep:
                break
            
            rand_mask_prob = self.mask_scheduler(timestep) # timestep #
            num_token_masked = (rand_mask_prob * m_tokens_len).long().clip(min=1)
            
            if token_cond is not None:
                num_token_masked = (rand_mask_prob * num_token_cond).long().clip(min=1)
                scores[token_cond!=mask_id] = 1e5

            sorted_indices = scores.argsort(dim=1)  # (b, k), sorted_indices[i, j] = the index of j-th lowest element in scores on dim=1
            ranks = sorted_indices.argsort(dim=1)  # (b, k), rank[i, j] = the rank (0: lowest) of scores[i, j] on dim=1
            is_mask = (ranks < num_token_masked.unsqueeze(-1))
            ids = torch.where(is_mask, mask_id, ids)

            if src_mask.ndim == 4: # We have to update src_mask to attend to new predicted tokens
                m = F.pad(ids < self.end_id, (self.num_cond, 0), value=False).unsqueeze(1).repeat(1, T+self.num_cond, 1)
                # m |= m.transpose(1, 2)
                m[~seq_mask_with_end_with_conds_expanded] = False
                # show([m], b=5)
                # show([m], b=5, randperm=randperm)
                m = m.unsqueeze(1).repeat(1, self.n_head, 1, 1)
                src_mask |= m
            # show([m[:, 0], src_mask[:, 0]], b=0) # with conds
            # show([m[:, 0, 1:, 1:], src_mask[:, 0, 1:, 1:]], randperm=randperm, b=0) # without conds
            logits = self.forward(ids, clip_feature, src_mask, word_emb=word_emb, text_mask=text_mask)

            if logits.shape[-1] > self.end_id: logits = logits[..., :self.end_id] # Here we sample with length, so no pred_end should be predicted

            filtered_logits = logits
            
            if self.args.rand_pos:
                pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)
            else:
                pred_ids = logits.argmax(-1)

            is_mask = ids >= start_mask_id
            ids = torch.where(is_mask, pred_ids, ids)

            probs_without_temperature = logits.softmax(dim=-1)  # (b, seqlen, ntoken)
            scores = probs_without_temperature.gather(2, pred_ids.unsqueeze(dim=-1))  # (b, seqlen, 1)
            scores = scores.squeeze(-1)  # (b, seqlen)

            # We do not want to re-mask the previously kept tokens, or pad tokens
            scores = scores.masked_fill(~is_mask, 1e5)

        return ids


    def Random_Sampling(self, clip_feature, word_emb, m_length=None, if_test=False, CFG=-1,
                            token_cond=None, max_steps = 10, text_mask=None):
        max_length = 49
        T = max_length + 1
        batch_size = clip_feature.shape[0]

        shape = (batch_size, self.block_size - self.num_cond)
        topk_filter_thres = .9
        starting_temperature = 1.0
        # scores = torch.ones(shape, dtype = torch.float32, device = clip_feature.device)
        
        m_tokens_len = torch.ceil((m_length)/4).long()
        src_token_mask = generate_src_mask(self.block_size-self.num_cond, m_tokens_len+1)
        src_token_mask_noend = generate_src_mask(self.block_size-self.num_cond, m_tokens_len)
        scores = torch.where(~src_token_mask_noend, 1e5, 0.)

        start_mask_id = self.mask_id

        if token_cond is not None:
            ids = token_cond.clone()
            ids[~src_token_mask_noend] = self.pad_id
            ids[~src_token_mask] = self.pad_id # [INFO] replace with pad id
            ids.scatter_(-1, m_tokens_len[..., None].long(), self.end_id) # [INFO] replace with end id
            mask_id, src_mask, *_, randperm, seq_mask_with_end_with_conds_expanded = \
                self.permuted_corruption(ids, T, m_tokens_len, args=self.args, eval_edit=True)
            token_cond[token_cond==-1] = mask_id[token_cond==-1]
            ids = token_cond.clone()
            num_token_cond = (ids>start_mask_id).sum(-1)
        else:
            mask_id, src_mask, *_, randperm, seq_mask_with_end_with_conds_expanded = \
                self.permuted_corruption(T=T, m_tokens_len=m_tokens_len, args=self.args)
            ids = mask_id.clone()
        src_mask[torch.arange(batch_size), :, :, m_tokens_len+1] = True # Making end_token True

        sample_max_steps = torch.round(max_steps/max_length*m_tokens_len) + 1e-8

        for step in range(max_steps):

            timestep = torch.clip(step/(sample_max_steps), max=1)
            if len(m_tokens_len)==1 and step > 0 and torch.clip(step-1/(sample_max_steps), max=1).cpu().item() == timestep:
                break
            
            rand_mask_prob = self.mask_scheduler(timestep) # timestep #
            num_token_masked = (rand_mask_prob * m_tokens_len).long().clip(min=1)

            if token_cond is not None:
                num_token_masked = (rand_mask_prob * num_token_cond).long().clip(min=1)
                scores[token_cond!=mask_id] = 1e5

            for i in range(batch_size):
                tokens = torch.nonzero(scores[i] != 1e5).squeeze().tolist()
                if not isinstance(tokens, list): tokens = [tokens]
                ids_to_mask = random.sample(tokens, k=len(tokens) - num_token_masked[i])
                scores[i][ids_to_mask] = 1e5

            sorted_indices = scores.argsort(dim=1)  # (b, k), sorted_indices[i, j] = the index of j-th lowest element in scores on dim=1
            ranks = sorted_indices.argsort(dim=1)  # (b, k), rank[i, j] = the rank (0: lowest) of scores[i, j] on dim=1
            is_mask = (ranks < num_token_masked.unsqueeze(-1))
            ids = torch.where(is_mask, mask_id, ids)

            if src_mask.ndim == 4: # We have to update src_mask to attend to new predicted tokens
                m = F.pad(ids < self.end_id, (self.num_cond, 0), value=False).unsqueeze(1).repeat(1, T+self.num_cond, 1)
                # m |= m.transpose(1, 2)
                m[~seq_mask_with_end_with_conds_expanded] = False
                # show([m], b=5)
                # show([m], b=5, randperm=randperm)
                m = m.unsqueeze(1).repeat(1, self.n_head, 1, 1)
                src_mask |= m
            # show([m[:, 0], src_mask[:, 0]], b=0) # with conds
            # show([m[:, 0, 1:, 1:], src_mask[:, 0, 1:, 1:]], randperm=randperm, b=0) # without conds
            logits = self.forward(ids, clip_feature, src_mask, word_emb=word_emb, text_mask=text_mask)

            if logits.shape[-1] > self.end_id: logits = logits[..., :self.end_id] # Here we sample with length, so no pred_end should be predicted

            filtered_logits = logits
            
            if self.args.rand_pos: temperature = 1
            else:                  temperature = 0

            pred_ids = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)

            is_mask = ids >= start_mask_id
            ids = torch.where(is_mask, pred_ids, ids)

            probs_without_temperature = logits.softmax(dim=-1)  # (b, seqlen, ntoken)
            scores = probs_without_temperature.gather(2, pred_ids.unsqueeze(dim=-1))  # (b, seqlen, 1)
            scores = scores.squeeze(-1)  # (b, seqlen)

            # We do not want to re-mask the previously kept tokens, or pad tokens
            scores = scores.masked_fill(~is_mask, 1e5)

        return ids


    def sampling_AR_with_length_batch(self, clip_feature, word_emb, m_length=None, if_test=False, CFG=-1,
                                token_cond=None, max_steps = 10, text_mask=None):
        device = clip_feature.device
        max_length = 49
        T = max_length + 1
        batch_size = clip_feature.shape[0]
        shape = (batch_size, self.block_size - self.num_cond)
        m_tokens_len = torch.ceil((m_length)/4).long()
        src_token_mask = generate_src_mask(self.block_size-self.num_cond, m_tokens_len+1)
        src_token_mask_noend = generate_src_mask(self.block_size-self.num_cond, m_tokens_len)

        start_mask_id = self.mask_id

        if token_cond is not None:
            ids = token_cond.clone()
            ids[~src_token_mask_noend] = self.pad_id
            ids[~src_token_mask] = self.pad_id # [INFO] replace with pad id
            ids.scatter_(-1, m_tokens_len[..., None].long(), self.end_id) # [INFO] replace with end id
            mask_id, src_mask, *_, randperm, seq_mask_with_end_with_conds_expanded = \
                self.permuted_corruption(ids, T, m_tokens_len, args=self.args, eval_edit=True)
            token_cond[token_cond==-1] = mask_id[token_cond==-1]
            ids = token_cond.clone()
            num_token_cond = (ids>start_mask_id).sum(-1)
        else:
            mask_id, src_mask, *_, randperm, seq_mask_with_end_with_conds_expanded = \
                self.permuted_corruption(T=T, m_tokens_len=m_tokens_len, args=self.args)
            ids = mask_id.clone()
        src_mask[torch.arange(batch_size), :, :, m_tokens_len+1] = True # Making end_token True
        
        if self.args.rand_pos:
            top_k = self.args.top_k
            top_p = self.args.top_p
            temperature = self.args.temperature

        times = ids - self.mask_id
        neg_times = times < 0
        times[neg_times] = 1e5
        times_sorted = times.argsort(-1)
        # times_sorted[neg_times] = 1e5
        times_sorted = torch.split(times_sorted[~neg_times], m_tokens_len.tolist())

        if not self.args.z_0_attend_to_all: # Start sampling from T
            times_sorted_ = []
            for t in times_sorted:
                times_sorted_.append(reversed(t))
            times_sorted = times_sorted_

        times_sorted_ = []
        for t in times_sorted:
            times_sorted_.append(F.pad(t, (0, T-len(t)), value=1e5).unsqueeze(0))
        times_sorted = torch.cat(times_sorted_, dim=0)

        bs_index = torch.arange(batch_size)

        for i in range(T):
            t = times_sorted[:, i]

            # id_input = id[:, :t+1] if self.args.start_ids_in_sampling == "ARANGED" else id
            # src_mask_input = src_mask[..., :t+1+self.num_cond, :t+1+self.num_cond] if self.args.start_ids_in_sampling == "ARANGED" else src_mask[bs:bs+1]

            if src_mask.ndim == 4: # We have to update src_mask to attend to new predicted tokens
                m = F.pad(ids < self.end_id, (self.num_cond, 0), value=False).unsqueeze(1).repeat(1, T+self.num_cond, 1)
                # m |= m.transpose(1, 2)
                m[~seq_mask_with_end_with_conds_expanded] = False
                # show([m], b=5)
                # show([m], b=5, randperm=randperm)
                m = m.unsqueeze(1).repeat(1, self.n_head, 1, 1)
                src_mask |= m
            # show([m[:, 0], src_mask[:, 0]], b=0)
            # show([m[:, 0, 1:, 1:], src_mask[:, 0, 1:, 1:]], randperm=randperm, b=0)

            logits = self.forward(ids, clip_feature, src_mask, word_emb=word_emb)
            if logits.shape[-1] > self.end_id: logits = logits[..., :self.end_id]

            mask_valid_t = t < 1e5
            t = torch.where(mask_valid_t, t, 0)

            if self.args.rand_pos:
                filtered_logits = top_k_top_p_filtering(logits[bs_index, t], top_k, top_p)
                pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)
                # pred_ids = gumbel_sample(logits[bs_index, t], temperature = temperature, dim = -1)
            else:
                pred_ids = logits[bs_index, t].argmax(-1)

            ids[mask_valid_t, t[mask_valid_t]] = pred_ids[mask_valid_t]


        assert (ids[torch.arange(batch_size), m_tokens_len] == self.end_id).all(), "Some thing is wrong, end id is replaced"
        assert (ids > self.pad_id).sum() == 0, "Mask tokens found in ids"

        return ids
