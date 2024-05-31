import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter

class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_heads, dropout=0.1, pos_embed=False, old=False, export=False):
        super(Transformer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.pos_embed = pos_embed
        self.export = export
        
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout)
        self.transformer_layers = nn.ModuleList([TransformerLayer(hidden_dim, num_heads, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.old = old
        if old:
            self.axis = Parameter(torch.tensor([0, -1, 0]).view(1, 3), requires_grad=False)
        else:
            self.yaxis = Parameter(torch.tensor([0, -1, 0]).view(1, 3), requires_grad=False)
            self.xaxis = Parameter(torch.tensor([-1, 0, 0]).view(1, 3), requires_grad=False)
            self.zeroaxis = Parameter(torch.tensor([0, 0, 0]).view(1, 3), requires_grad=False)

    def create_axis_batch(self, object_ids):
        if self.old:
            return self.axis.repeat(object_ids.shape[0], 1)
        
        if self.export:
            return self.yaxis.repeat(object_ids.shape[0], 1)

        bs = object_ids.shape[0]
        axis = torch.zeros(bs, 3).to(object_ids.device)
        for i in range(bs):
            if object_ids[i] in [1, 2, 3, 4, 10, 11, 13]:
                axis[i] = self.yaxis
            elif object_ids[i] in [5, 6]:
                axis[i] = self.xaxis
            else:
                axis[i] = self.zeroaxis
        # axis = self.axis.repeat(bs, 1)
        return axis

    def forward(self, x):
        bs = x.shape[0]
        object_id = x[:, 0, 4:18]
        object_id = torch.argmax(object_id, dim=1)
        axis = self.create_axis_batch(object_id)
        # print(axis)

        x = self.embedding(x)
        for i in range(self.num_layers):
            x = self.transformer_layers[i](x)
        x = self.fc(x)
        # average across keypoints
        kps = x[:, :, :3]
        pose = x[:, :, 3:]
        pose = torch.mean(pose, dim=1)

        # articulation
        a = torch.relu(pose[:, -1]).view(-1, 1)
        a_axis = axis * a
        pose = torch.cat((pose[:, :-1], a_axis), dim=1)

        return pose, kps
    
class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super(TransformerLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.feed_forward = FeedForward(hidden_dim, dropout)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        x = self.layer_norm1(x + self.multi_head_attention(x))
        x = self.layer_norm2(x + self.feed_forward(x))
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        batch_size = x.shape[0]
        query = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        attention = torch.matmul(scores, value)
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        x = self.fc(attention)
        return x
    
class FeedForward(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
