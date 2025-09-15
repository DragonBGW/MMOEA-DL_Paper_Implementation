# pointer_net.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Simple pointer network encoder-decoder with attention (Nazari-style)
class Encoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        # x: batch x n x 2
        h = F.relu(self.fc(x))  # batch x n x hid
        out, hn = self.gru(h)   # out: batch x n x hid
        return out, hn  # encoder outputs, hidden

class Decoder(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        # attention parameters
        self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.vt = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, enc_outs, dec_input, dec_hidden, mask):
        # enc_outs: batch x n x hid
        # dec_input: batch x 1 x hid (context)
        # dec_hidden: 1 x batch x hid
        out, h = self.gru(dec_input, dec_hidden)  # out: batch x 1 x hid
        # attention
        # score_{i} = vt^T * tanh(W1 * enc_i + W2 * out)
        w1 = self.W1(enc_outs)  # batch x n x hid
        w2 = self.W2(out)       # batch x 1 x hid
        scores = torch.tanh(w1 + w2)  # broadcast
        scores = torch.matmul(scores, self.vt)  # batch x n
        scores = scores.masked_fill(mask, float('-inf'))
        probs = F.softmax(scores, dim=-1)
        return probs, h

class PointerNet(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim)
        self.hidden_dim = hidden_dim
        # projection for starting decoder input
        self.start_proj = nn.Linear(input_dim, hidden_dim)

    def forward(self, points, decode_len=None):
        # points: batch x n x 2
        batch, n, _ = points.size()
        enc_outs, enc_h = self.encoder(points)  # enc_outs: batch x n x hid
        # start decoder input: mean of points projected
        start = points.mean(dim=1)  # batch x 2
        dec_input = self.start_proj(start).unsqueeze(1)  # batch x1 x hid
        dec_hidden = enc_h  # 1 x batch x hid
        mask = torch.zeros(batch, n, dtype=torch.bool, device=points.device)
        seqs = []
        logps = []
        if decode_len is None:
            decode_len = n
        for t in range(decode_len):
            probs, dec_hidden = self.decoder(enc_outs, dec_input, dec_hidden, mask)
            # sample / greedy
            dist = torch.distributions.Categorical(probs)
            idx = dist.sample().detach()  # batch
            logp = dist.log_prob(idx)
            seqs.append(idx)
            logps.append(logp)
            # update mask
            mask = mask.clone()
            mask[torch.arange(batch), idx] = True
            # new dec_input = encoder output of chosen idx
            dec_input = enc_outs[torch.arange(batch), idx].unsqueeze(1)
        seqs = torch.stack(seqs, dim=1)  # batch x n
        logps = torch.stack(logps, dim=1)  # batch x n
        return seqs, logps.sum(dim=1)  # return sequence and total logp
