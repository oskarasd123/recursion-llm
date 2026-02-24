import torch
from torch import nn, Tensor
import torch.nn.functional as F
from flash_attn import flash_attn_func, flash_attn_varlen_func
import math
from dataclasses import dataclass


def norm(x):
    return F.rms_norm(x, (x.size(-1),))

class ShortEmbedding(nn.Module):
    def __init__(self, num_embeddings, dim, divisor, dtype = torch.bfloat16):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings, dim//divisor, dtype=dtype)
        self.embed_linear = CastedLinear(dim//divisor, dim)
    
    def forward(self, ids):
        return self.embed_linear(self.embed(ids))

class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up_lin = CastedLinear(dim, dim * 4, dtype=torch.bfloat16)
        self.down_lin = CastedLinear(dim * 4, dim, dtype=torch.bfloat16)
        self.down_lin.weight.data.zero_()
    
    def forward(self, x):
        return self.down_lin(F.gelu(self.up_lin(x)))


def rotary(x_THD: Tensor, cos: Tensor, sin: Tensor):
    cos = cos[:x_THD.size(-3), None, :]
    sin = sin[:x_THD.size(-3), None, :]
    x1, x2 = x_THD.chunk(2, dim=-1)
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat((y1, y2), -1)

class ROPE(nn.Module):
    def __init__(self, dim : int, base : int, max_seq_len : int):
        super().__init__()
        freqs = (1/base) ** torch.linspace(0, 1, dim//2)
        t = torch.arange(max_seq_len)
        theta = torch.outer(t, freqs)
        self.register_buffer("sin", theta.sin())
        self.register_buffer("cos", theta.cos())
    
    def forward(self, x):
        return rotary(x, self.cos.type_as(x), self.sin.type_as(x))

class HalfROPE(nn.Module):
    def __init__(self, dim : int, base : int, max_seq_len : int):
        super().__init__()
        freqs = (1/base) ** torch.linspace(0, 1, dim//4)
        t = torch.arange(max_seq_len)
        theta = torch.outer(t, freqs)
        self.register_buffer("sin", theta.sin())
        self.register_buffer("cos", theta.cos())
    
    def forward(self, x):
        x1, x2 = x.chunk(2, -1)
        return torch.cat([rotary(x1, self.cos.type_as(x), self.sin.type_as(x)), x2], -1)


class CastedLinear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.type_as(x), self.bias.type_as(x) if self.bias is not None else None)

class SliceLinear(nn.Linear):
    def __init__(self, in_features, out_features, slice_start=0, bias = True, device = None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.slice_start = slice_start
    def forward(self, x): # can take n_features larger than weight
        return F.linear(x[..., self.slice_start:self.weight.size(-1)+self.slice_start], self.weight.type_as(x), self.bias.type_as(x) if self.bias is not None else None)


@dataclass
class AttnArgs:
    rope : ROPE
    cu_seqlens : Tensor
    max_seqlen : Tensor

#torch.compiler.allow_in_graph(flash_attn_varlen_func)
flash_attn_varlen_func = torch.compiler.disable(flash_attn_varlen_func)

class CausalAttn(nn.Module):
    def __init__(self, dim, num_heads, window_size = -1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.q_lin = CastedLinear(dim, dim, dtype=torch.bfloat16)
        self.k_lin = CastedLinear(dim, dim, dtype=torch.bfloat16)
        self.v_lin = CastedLinear(dim, dim, dtype=torch.bfloat16)
        self.o_lin = CastedLinear(dim, dim, dtype=torch.bfloat16)
        self.o_lin.weight.data.zero_()
        self.attn_gate = SliceLinear(20, num_heads)
        
    def forward(self, x, args : AttnArgs):
        T, D = x.shape
        q = self.q_lin(x).view(T, self.num_heads, self.head_dim)
        k = self.k_lin(x).view(T, self.num_heads, self.head_dim)
        v = self.v_lin(x).view(T, self.num_heads, self.head_dim)

        rope = args.rope
        cu_seqlens, max_seqlen = args.cu_seqlens, args.max_seqlen
        
        q, k = rope(q), rope(k)

        #q = q.transpose(1, 2)
        #k = k.transpose(1, 2)
        #v = v.transpose(1, 2)
        
        #o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = flash_attn_varlen_func(q, k, v, 
                                   cu_seqlens, cu_seqlens, 
                                   max_seqlen, max_seqlen, 
                                   causal=True, window_size=(self.window_size, 0))
        #o = flash_attn_func(q, k, v, causal=True, window_size=(self.window_size, 0))
        o = o * torch.sigmoid(self.attn_gate(x).view(T, self.num_heads, 1))
        #o = o.transpose(1, 2).contiguous()
        o = o.view(T, D)
        return self.o_lin(o)

class Block(nn.Module):
    def __init__(self, dim, num_heads, window_size=-1):
        super().__init__()
        self.attn = CausalAttn(dim, num_heads, window_size)
        self.mlp = MLP(dim)
        self.xs_gate = nn.Parameter(torch.zeros(dim))
        self.x0_gate = nn.Parameter(torch.zeros(dim))
    
    def forward(self, x, args, xs = None, x0 = None):
        if xs is not None:
            x = x + xs * self.xs_gate.type_as(xs)
        if x0 is not None:
            x = x + x0 * self.x0_gate.type_as(x0)
        x = x + self.attn(norm(x), args)
        x = x + self.mlp(norm(x))
        return x

class Model(nn.Module):
    def __init__(self, num_embeddings, dim, num_layers, num_heads, window_size = -1, max_seq_len = 8192):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.window_size = window_size
        self.embed = nn.Embedding(num_embeddings, dim)
        self.blocks = nn.ModuleList([Block(dim, num_heads, window_size) for i in range(num_layers)])
        self.rope = ROPE(dim//num_heads, 1000, max_seq_len)
        self.un_embed = CastedLinear(dim, num_embeddings, bias=False, dtype=torch.bfloat16)
        self.end_mlp = MLP(dim)
        self.middle_block = Block(dim, num_heads, window_size*2+1)
        self.value_embeds = nn.ModuleList([ShortEmbedding(num_embeddings, dim, 4) for i in range(3)])
        self.smear_gate = SliceLinear(10, 1, 10)
        self.smear_linear = CastedLinear(dim, dim)
        self.end_gate = CastedLinear(dim, 1)
        
        # Weight tying (optional but standard and helps convergence)
        self.embed.weight = self.un_embed.weight
        #with torch.no_grad():
        #    self.embed.weight.copy_(self.un_embed.weight)


    def forward(self, ids, cu_seqlens, max_seqlen, loop_steps = 1, return_output_weights = False):
        assert loop_steps > 0
        x0 = self.embed(ids).to(torch.bfloat16)
        x0 = torch.cat([x0[:1], x0[1:] + torch.sigmoid(self.smear_gate(x0[1:])) * self.smear_linear(x0[:-1])], dim=0)
        x = x0
        half = len(self.blocks)//2
        ve = [embed(ids).to(torch.bfloat16) for embed in self.value_embeds]
        value_embeds = [None, ve[0], ve[1]] + [ve[2]]*(half-3)
        outputs = []
        end_gates = []
        for _ in range(loop_steps):
            skip_connections = []
            attn_args = AttnArgs(self.rope, cu_seqlens, max_seqlen)
            for block, v0 in zip(self.blocks[:half], value_embeds):
                x = block(x, attn_args, xs=v0, x0=x0)
                skip_connections.append(x)
            if len(self.blocks)%2!=0:
                skip_connections.append(None)
            x = self.middle_block(x, attn_args)
            for block, xs in zip(self.blocks[half:], reversed(skip_connections)):
                x = block(x, attn_args, xs=xs, x0=x0)
            o = x + self.end_mlp(norm(x)) # mlp that transforms vectors from thought space into embedding space
            outputs.append(o)
            end_gates.append(self.end_gate(x))
        
        cum_continue_prob = 1
        o_weights = [] # weights sum to 1
        for i, (o, cur_end_prob) in enumerate(zip(outputs, end_gates)):
            if i == loop_steps-1:
                cur_end_prob = torch.ones_like(cur_end_prob) # set ending probability to 1
            current_weight = cur_end_prob * cum_continue_prob
            o_weights.append(current_weight)
            cum_continue_prob = cum_continue_prob * 1-cur_end_prob

        outputs = torch._foreach_mul(outputs, o_weights)
        o = torch.sum(torch.stack(outputs), 0)
        if return_output_weights:
            return self.un_embed(norm(o)), o_weights
        return self.un_embed(norm(o))