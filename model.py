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
        self.down_lin.weight.data.zero_()
    
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

class SliceLinear(nn.Linear):
    def forward(self, x): # can take n_features larger than weight
        return F.linear(x[..., :self.weight.size(-1)], self.weight.type_as(x), self.bias.type_as(x) if self.bias is not None else None)

class CausalAttn(nn.Module):
    def __init__(self, dim, num_heads, window_size = -1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.q_lin = CastedLinear(dim, dim)
        self.k_lin = CastedLinear(dim, dim)
        self.v_lin = CastedLinear(dim, dim)
        self.o_lin = CastedLinear(dim, dim)
        self.o_lin.weight.data.zero_()
        self.attn_gate = SliceLinear(20, num_heads)
        
    def forward(self, x, rope):
        B, T, D = x.shape
        q = self.q_lin(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_lin(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v_lin(x).view(B, T, self.num_heads, self.head_dim)
        
        q, k = rope(q), rope(k)

        #q = q.transpose(1, 2)
        #k = k.transpose(1, 2)
        #v = v.transpose(1, 2)
        
        #o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = flash_attn_func(q, k, v, causal=True, window_size=(self.window_size, 0))
        o = o * torch.sigmoid(self.attn_gate(x).view(B, T, self.num_heads, 1))
        #o = o.transpose(1, 2).contiguous()
        o = o.view(B, T, D)
        return self.o_lin(o)

class Block(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attn = CausalAttn(dim, num_heads)
        self.mlp = MLP(dim)
        self.scalars = nn.Parameter(torch.zeros(1))
    
    def forward(self, x, rope, x0 = None):
        if x0 is not None:
            x = x * (1-self.scalars[0]) + x0 * self.scalars[0]
        x = x + self.attn(norm(x), rope)
        x = x + self.mlp(norm(x))
        return x

class Model(nn.Module):
    def __init__(self, num_embeddings, dim, num_layers, num_heads, max_seq_len = 8192):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings, dim)
        self.blocks = nn.ModuleList([Block(dim, num_heads) for i in range(num_layers)])
        self.rope = ROPE(dim//num_heads, 10000, max_seq_len)
        self.un_embed = CastedLinear(dim, num_embeddings, bias=False)
        self.end_mlp = MLP(dim)
        self.middle_block = Block(dim, num_heads)
        
        # Weight tying (optional but standard and helps convergence)
        self.embed.weight = self.un_embed.weight 


    def forward(self, ids, recursion_steps = 4):
        x = self.embed(ids).to(torch.bfloat16)
        half = len(self.blocks)//2
        skip_connections = []
        for block in self.blocks[:half]:
            skip_connections.append(x)
            x = block(x, self.rope)
        if len(self.blocks)%2!=0:
            skip_connections.append(None)
        for i in range(recursion_steps):
            x = self.middle_block(x, self.rope)
        
        for block, x0 in zip(self.blocks[half:], reversed(skip_connections)):
            x = block(x, self.rope, x0=x0)
        x = x + self.end_mlp(norm(x))
        return self.un_embed(norm(x))