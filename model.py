import torch
from torch import nn, Tensor
import torch.nn.functional as F
from flash_attn import flash_attn_func








def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up_lin = CastedLinear(dim, dim * 4)
        self.down_lin = CastedLinear(dim * 4, dim)
    
    def forward(self, x):
        return self.down_lin(F.gelu(self.up_lin(x)))

def rotary(x_BTHD: Tensor, cos: Tensor, sin: Tensor):
    cos = cos[None, :x_BTHD.size(-3), None, :]
    sin = sin[None, :x_BTHD.size(-3), None, :]
    x1, x2 = x_BTHD.chunk(2, dim=-1)
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


class CastedLinear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.type_as(x), self.bias.type_as(x) if self.bias is not None else None)

class CausalAttn(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_lin = CastedLinear(dim, dim)
        self.k_lin = CastedLinear(dim, dim)
        self.v_lin = CastedLinear(dim, dim)
        self.o_lin = CastedLinear(dim, dim)
        
    def forward(self, x, rope):
        B, T, D = x.shape
        q = self.q_lin(x).view(B, T, self.num_heads, self.head_dim)#.transpose(1, 2)
        k = self.k_lin(x).view(B, T, self.num_heads, self.head_dim)#.transpose(1, 2)
        v = self.v_lin(x).view(B, T, self.num_heads, self.head_dim)#.transpose(1, 2)
        
        q, k = rope(q), rope(k)
        
        #o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = flash_attn_func(q, k, v, causal=True)
        #o = o.transpose(1, 2).contiguous()
        o = o.view(B, T, D)
        return self.o_lin(o)

class Block(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attn = CausalAttn(dim, num_heads)
        self.mlp = MLP(dim)
    
    def forward(self, x, rope):
        x = x + self.attn(norm(x), rope)
        x = x + self.mlp(norm(x))
        return x

class Model(nn.Module):
    def __init__(self, num_embeddings, dim, num_layers, num_heads):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings, dim)
        self.blocks = nn.ModuleList([Block(dim, num_heads) for _ in range(num_layers)])
        self.rope = ROPE(dim//num_heads, 10000, 8192) # Standard base 10000
        self.un_embed = CastedLinear(dim, num_embeddings, bias=False)
        self.end_mlp = MLP(dim)
        
        # Weight tying (optional but standard and helps convergence)
        self.embed.weight = self.un_embed.weight 


    def forward(self, ids):
        x = self.embed(ids).to(torch.bfloat16)
        for block in self.blocks:
            x = block(x, self.rope)
        x = x + self.end_mlp(norm(x))
        return self.un_embed(norm(x))