# -*- coding: utf-8 -*-

"""
@author: xansar
@software: PyCharm
@file: test_hyperbolic_layers.py
@time: 2023/5/15 13:16
@e-mail: xansar@ruc.edu.cn
"""
import torch
from hybo_modules.modules import \
    LorentzEmbeddings, \
    LorentzTransformerEncoderLayer, \
    LorentzTransformerDecoderLayer, \
    LorentzGraphConvolution, \
    LorentzSelfAttend
from hybo_modules.manifolds import Lorentz

def test_self_attn():
    seqs = torch.randn(2, 8, 10)
    seq_masks = torch.randn(2, 8)
    manifold = Lorentz()
    self_attn = LorentzSelfAttend(manifold, 10)

    output = self_attn(seqs, seq_masks)
    print(f'output: {output}')
    print(f'output.shape: {output.shape}')
    loss = output.mean()
    loss.backward()

def test_gcn():
    n_nodes = 3
    vocab_size = 3
    padding_idx = -1
    embed_dim = 2
    in_features = embed_dim
    out_features = embed_dim
    use_bias = False
    dropout = 0.
    use_att = False
    local_agg = False
    manifold = Lorentz()
    gcn = LorentzGraphConvolution(in_features, out_features, use_bias, dropout, use_att, local_agg, nonlin=None)
    embedding = LorentzEmbeddings(manifold, embed_dim, vocab_size, padding_idx)

    x = embedding(torch.arange(n_nodes), use_position_encoding=False)
    adj = torch.randint(0, 2, (n_nodes, n_nodes)).float().to_sparse_coo()
    print(f'x: {x}')
    print(f'x.shape: {x.shape}')
    print(f'adj: {adj}')
    print(f'adj.shape: {adj.shape}')
    inputs = (x, adj)
    outputs, adj = gcn(inputs)
    print(f'outputs: {outputs}')
    print(f'outputs.shape: {outputs.shape}')
    loss = outputs.mean()
    loss.backward()

def test_transformer():
    vocab_size = 10
    padding_idx = -1
    embed_dim = 10
    heads = 2
    d_ff = embed_dim
    drop_out = 0.3
    attention_dropout = 0.3
    manifold = Lorentz()
    # embedding layer
    embedding = LorentzEmbeddings(manifold, embed_dim, vocab_size, padding_idx)

    # encoder layer
    encoder = LorentzTransformerEncoderLayer(manifold, embed_dim, heads, d_ff, drop_out, attention_dropout)

    # decoder layer
    decoder = LorentzTransformerDecoderLayer(manifold, embed_dim, heads, d_ff, drop_out, attention_dropout)

    src_seq = torch.randint(0, 10, (2, 5))
    tgt_seq = torch.randint(0, 10, (2, 7))
    encoder_mask = torch.randint(0, 1, (2, 5, 5))
    src_pad_mask = torch.randint(0, 2, (2, 7, 5))
    tgt_pad_mask = torch.randint(0, 2, (2, 7, 7))
    print(f'src_seq: {src_seq}')
    print(f'tgt_seq: {tgt_seq}')
    print(f'encoder_mask: {encoder_mask}')
    print(f'src_pad_mask: {src_pad_mask}')
    print(f'tgt_pad_mask: {tgt_pad_mask}')
    # src_seq = src_seq.transpose(0, 1).unsqueeze(2)
    print(src_seq.shape)
    embed = embedding(src_seq)
    print(f'embed: {embed}')
    print(f'embed shape: {embed.shape}')
    # encoder需要两个参数，输入embed和mask
    encoded_embed = encoder(embed, encoder_mask)
    print(f'encoded_embed: {encoded_embed}')
    print(f'encoded_embed shape: {encoded_embed.shape}')

    # decoder需要四个输入embed, memory_bank, src_pad_mask, and tgt_pad_mask
    # src_pad_mask应该是batch，num_tgt, num_src
    tgt_embed = embedding(tgt_seq)
    output, top_attn, attn_align = decoder(tgt_embed, encoded_embed, src_pad_mask, tgt_pad_mask)
    print(f'output: {output}')
    print(f'output shape: {output.shape}')

    print(f'top_attn: {top_attn}')
    print(f'top_attn shape: {top_attn.shape}')

    print(f'attn_align: {attn_align}')
    # print(f'attn_align shape: {attn_align.shape}')
    loss = output.mean()
    loss.backward()


if __name__ == '__main__':
    test_transformer()
    test_gcn()
    test_self_attn()
