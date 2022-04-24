import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from config import _cuda
import numpy as np
import math
import pickle as pkl
import functools
from initiate_functions import *

@functools.total_ordering
class Prioritize:

    def __init__(self, priority, item):
        self.priority = priority
        self.item = item

    def __eq__(self, other):
        return self.priority == other.priority

    def __lt__(self, other):
        return self.priority < other.priority


class BeamSearchNode(object):
    def __init__(self, hidden_words, hidden_sents, context_vector, coverage,
                 previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h_word = hidden_words
        self.h_sent = hidden_sents
        self.c_t = context_vector
        self.coverage = coverage
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.len = length
        self.avg_score = -self.logp / float(self.len - 1 + 1e-6)

    def cal(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.len - 1 + 1e-6) + alpha * reward


class GCN(nn.Module):
    def __init__(self, hidden_size):
        super(GCN, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size // 2))
        self.b = nn.Parameter(torch.FloatTensor(self.hidden_size // 2))

        self.init()

    def init(self):
        stdv = 1 / math.sqrt(self.hidden_size)
        self.W.data.uniform_(-stdv, stdv)
        self.b.data.uniform_(-stdv, stdv)

    def forward(self, inp, adj, is_relu=True):
        out = torch.matmul(inp, self.W) + self.b
        out = torch.matmul(adj, out)

        if is_relu:
            out = F.relu(out)

        return out


class WordGraphEncoder(nn.Module):
    def __init__(self, vocab_size, num_pos, embedding_size=256, pos_embedding_size=128,
                 hidden_size=256, rnn_layer=2, gcn_layer=2, dropout=0.5):
        super(WordGraphEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.num_pos = num_pos
        self.embedding_size = embedding_size
        self.pos_embedding_size = pos_embedding_size
        self.hidden_size = hidden_size
        self.rnn_layer = rnn_layer
        self.gcn_layer = gcn_layer
        self.dropout = nn.Dropout(dropout)
        self.word_attention_network = Attention(hidden_size)
        self.W_word = nn.Linear(hidden_size, hidden_size)
        self.word_layer_norm_2 = nn.LayerNorm(self.hidden_size*2)

        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=PAD_token)
        #init_wt_normal(self.embedding.weight)
        if args['pre_train']:
            weights_matrix = pkl.load(open(args['word_embedding_path'], 'rb'))[:input_size,:]
            print(np.shape(weights_matrix))
            self.embedding.load_state_dict({'weight': torch.Tensor(weights_matrix)})
            if args['embed_notrain']:
                self.embedding.weight.requires_grad = False
            print('pre_train word embedding loaded!')

        if args['use_pos']:
            #self.word_layer_norm_1 = nn.LayerNorm(self.embedding_size + self.pos_embedding_size)
            self.pos_embedding = nn.Embedding(num_pos, pos_embedding_size)
            #init_wt_normal(self.pos_embedding.weight)
            self.rnn = nn.GRU(embedding_size+pos_embedding_size, self.hidden_size, num_layers=rnn_layer,
                              batch_first=True, dropout=dropout, bidirectional=True)
        else:
            #self.word_layer_norm_1 = nn.LayerNorm(self.embedding_size)
            self.rnn = nn.GRU(embedding_size, self.hidden_size, num_layers=rnn_layer,
                              batch_first=True, dropout=dropout, bidirectional=True)
        #init_rnn_wt(self.rnn)
        self.gcn_fw = nn.ModuleList([GCN(self.hidden_size*2) for _ in range(gcn_layer)])
        self.gcn_bw = nn.ModuleList([GCN(self.hidden_size*2) for _ in range(gcn_layer)])
        self.layer_norm_gcn = nn.ModuleList(nn.LayerNorm(self.hidden_size*2) for _ in range(gcn_layer))

        self.rnn_out = nn.GRU(self.hidden_size*2, self.hidden_size, batch_first=True)
        #init_rnn_wt(self.rnn_out)

    def forward(self, inp, pos, dep_fw, dep_bw, sentences_real_len):
        # inp: multi documents with multi sentences with shape B x T_sents x T_words.
        # pos: same shape as inp with the POS tag of each word
        # dep_fw, dep_bw: dependency matrix for each word pair within one sentence,
        #                 shape B x T_sents x T_words x T_words
        # sentences_real_len: the real length of each sentences in every document

        batch_size, document_len, sentence_len = inp.size()
        inp = self.embedding(inp)
        inp = inp.contiguous().view(-1, sentence_len, self.embedding_size)
        if args['use_pos']:
            pos = self.pos_embedding(pos)
            pos = pos.contiguous().view(-1, sentence_len, self.pos_embedding_size)
            inp = torch.cat([inp, pos], dim=2)
        #inp = self.word_layer_norm_1(inp)
        inp = self.dropout(inp)
        sentences_real_len = sentences_real_len.view(-1)
        pack_inp = nn.utils.rnn.pack_padded_sequence(inp, sentences_real_len.tolist(),
                                                     batch_first=True, enforce_sorted=False)
        #pack_inp = self.dropout(pack_inp)
        words_rnn_outputs, _ = self.rnn(pack_inp)
        words_rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(words_rnn_outputs, batch_first=True,
                                                                total_length=sentence_len)
        
        dep_fw = dep_fw.contiguous().view(-1, sentence_len, sentence_len)
        dep_bw = dep_bw.contiguous().view(-1, sentence_len, sentence_len)
        
        
        for i in range(self.gcn_layer):
            #words_rnn_outputs = self.layer_norm_gcn[i](words_rnn_outputs)
            out_fw = self.gcn_fw[i](words_rnn_outputs, dep_fw)
            out_bw = self.gcn_bw[i](words_rnn_outputs, dep_bw)
            words_rnn_outputs = torch.cat([out_fw, out_bw], dim=2)
            words_rnn_outputs = self.dropout(words_rnn_outputs)
        
        #words_rnn_outputs = self.word_layer_norm_2(words_rnn_outputs)
       
        words_encoder_output, hidden = self.rnn_out(words_rnn_outputs)

        # TODO: with word level attention to obtain dynamic input of sentences graph
        sentences_mask = (sentences_real_len.view(-1, 1, 1) - 1).expand(words_encoder_output.size())
        last_words_encoder_outputs = words_encoder_output.gather(1, sentences_mask)[:, 0, :]
        words_encoder_output = words_encoder_output.view(batch_size, document_len,
                                                         sentence_len, -1)
        # B x T_sents x T_words x embeddind_dim
        last_words_encoder_outputs = last_words_encoder_outputs.view(batch_size, document_len, -1)
        # B x T_sents x embedding_dim
        return words_encoder_output, last_words_encoder_outputs, hidden[-1]


class SentsGraphEncoder(nn.Module):
    def __init__(self, hidden_size=256, rnn_layer=2, gcn_layer=2, dropout=0.5):
        super(SentsGraphEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_layer = rnn_layer
        self.gcn_layer = gcn_layer
        self.dropout = nn.Dropout(dropout)

        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers=rnn_layer, batch_first=True, dropout=dropout)
        self.gcn = nn.ModuleList([GCN(self.hidden_size) for _ in range(gcn_layer)])
        self.transform = nn.ModuleList([nn.Linear(hidden_size//2, hidden_size) for _ in range(gcn_layer-1)])
        self.out = nn.Linear(self.hidden_size, hidden_size)
        #init_linear_wt(self.out)
        self.sent_layer_norm = nn.LayerNorm(self.hidden_size)
        self.sent_layer_nrom_gcn = nn.LayerNorm(self.hidden_size)

    def forward(self, inp, adj, documents_real_len):

        doc_len = inp.size()[1]
        inp = self.dropout(inp)
        #documents_real_len = documents_real_len.view(-1)
        #inp = self.sent_layer_norm(inp)
        pack_inp = nn.utils.rnn.pack_padded_sequence(inp, documents_real_len, #.tolist(),
                                                     batch_first=True, enforce_sorted=False)
        #pack_inp = self.dropout(pack_inp)
        sents_rnn_outputs, hidden = self.rnn(pack_inp)
        sents_rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(sents_rnn_outputs, batch_first=True,
                                                                total_length=doc_len)

        #sents_rnn_outputs = self.dropout(inp) #sents_rnn_outputs)
        '''
        for i in range(self.gcn_layer):
            sents_rnn_outputs = self.sent_layer_nrom_gcn(sents_rnn_outputs)
            #print(i, sents_rnn_outputs.size(), adj.size())
            sents_rnn_outputs = self.gcn[i](sents_rnn_outputs, adj)
            if i < self.gcn_layer-1:
                sents_rnn_outputs = self.transform[i](sents_rnn_outputs)
            sents_rnn_outputs = self.dropout(sents_rnn_outputs)
        '''
        sents_encoder_output = self.out(sents_rnn_outputs)

        return sents_encoder_output, hidden


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        # attention
        #if args['is_coverage']:
        self.W_c = nn.Linear(1, hidden_size, bias=False)
        self.decode_proj = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask, coverage, is_word=True):
        batch_size, seq_len, hidden_dim = encoder_outputs.size()
        dec_fea = self.decode_proj(s_t_hat) # B x hidden_dim
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(batch_size, seq_len, hidden_dim).contiguous()
        # B x t_k x hidden_dim
        dec_fea_expanded = dec_fea_expanded.view(-1, hidden_dim)  # B * t_k x hidden_dim
        att_features = encoder_feature + dec_fea_expanded # B * t_k x hidden_dim
        if args['is_coverage'] and is_word:
            coverage_input = coverage.view(-1, 1)  # B * t_k x 1
            coverage_feature = self.W_c(coverage_input)  # B * t_k x hidden_dim
            att_features = att_features + coverage_feature

        e = torch.tanh(att_features) # B * t_k x hidden_dim
        scores = self.v(e)  # B * t_k x 1
        scores = scores.view(-1, seq_len)  # B x t_k
        if is_word:
            #print(torch.sum(enc_padding_mask, 1))
            #attn_dist_ = F.sigmoid(scores) * enc_padding_mask.float()  # B x t_k
            attn_dist_ = F.softmax(scores, dim=1) * enc_padding_mask.float()  # B x t_k
            normalization_factor = attn_dist_.sum(1, keepdim=True) + 1e-20
            attn_dist_ = attn_dist_ / normalization_factor
            attn_dist = attn_dist_
        else:
            #attn_dist_ = F.softmax(scores, dim=1) * enc_padding_mask.float()  # B x t_k
            attn_dist_ = torch.sigmoid(scores) * enc_padding_mask  # since the target word may respect to multi sentences
            #normalization_factor = attn_dist_.sum(1, keepdim=True)
            #attn_dist_ = attn_dist_ / normalization_factor
            attn_dist = attn_dist_
        #print('eeee', torch.sum(attn_dist))

        attn_dist = attn_dist.unsqueeze(1)  # B x 1 x t_k
        c_t = torch.bmm(attn_dist, encoder_outputs).squeeze(1)  # B x hidden_dim

        attn_dist = attn_dist.squeeze(1)  # B x t_k

        return c_t, attn_dist, attn_dist_, scores


class DynDecoder(nn.Module):
    def __init__(self, decoder_embedding, embedding_size, vocab,
                 hidden_size, dropout, n_layers):
        super(DynDecoder, self).__init__()
        self.decoder_embedding = decoder_embedding
        self.vocab = vocab
        self.vocab_size = vocab.n_words
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embedding_size, self.hidden_size, num_layers=n_layers,
                          batch_first=True, dropout=dropout)
        #init_rnn_wt(self.gru)
        self.sent_attention_network = Attention(hidden_size)
        self.W_sent = nn.Linear(hidden_size, hidden_size)
        # decoder

        self.x_context = nn.Linear(hidden_size + embedding_size, embedding_size)

        if args['copy']:
            self.p_gen_linear = nn.Linear(hidden_size * 2 + embedding_size, 1)

        # p_vocab
        self.out1 = nn.Linear(hidden_size * 2, hidden_size)
        #init_linear_wt(self.out1)
        self.out2 = nn.Linear(hidden_size * 2, self.vocab_size)
        #init_linear_wt(self.out2)
        self.layer_norm = nn.LayerNorm(self.vocab_size)
        self.layer_norm_gru = nn.LayerNorm(self.embedding_size)

    def forward(self, inp, hidden, encoder_outputs, encoder_feature, encoder_mask, words_mask, copy_content,
                context_vector, copy_vocab, coverage, words_score, word_attn_dist, step):
        batch_size = inp.size()[0]
        inp = self.decoder_embedding(inp)
        x = self.x_context(torch.cat((context_vector, inp), 1))  # B * (embedding_size)
        #x = self.layer_norm_gru(x)
        out, hidden = self.gru(x.unsqueeze(1), hidden)

        context_vector, sent_attn_dist_norm, sent_attn_dist, \
            sents_score = self.sent_attention_network(hidden[-1], encoder_outputs,
                                                      encoder_feature, encoder_mask,
                                                      coverage, False)
        #sent_attn_dist_ = sent_attn_dist_norm.unsqueeze(2).expand(word_attn_dist.size())
        #attn_dist = sent_attn_dist_ * word_attn_dist
        
        sents_score = sents_score.unsqueeze(2).expand(words_score.size())
        all_socre = (words_score + sents_score).contiguous().view(batch_size, -1)
        attn_dist = F.softmax(all_socre, dim=1) * (words_mask.view(batch_size, -1).float())
        normalization_factor = attn_dist.sum(1, keepdim=True) + 1e-20
        attn_dist = attn_dist / normalization_factor
        '''
        attn_dist_v, attn_dist_idx = attn_dist.topk(1)
        top_attn_dist = (attn_dist == attn_dist_v).float() * attn_dist
        top_attn_dist = top_attn_dist.view(words_score.size())
        '''
        attn_dist = attn_dist.view(words_score.size())
        
        #if args['is_coverage']:
        #    coverage = coverage + attn_dist
        #else:
        #    coverage = torch.zeros_like(attn_dist)

        p_gen = None
        if args['copy']:
            p_gen_input = torch.cat((context_vector, hidden[-1], x), 1)  # B x (2*hidden_dim + emb_dim)
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = torch.sigmoid(p_gen)

        output = torch.cat((out.view(-1, self.hidden_size), context_vector), 1)  # B x hidden_dim * 2
        #output = self.out1(output)  # B x hidden_dim

        #output = F.relu(output)

        output = self.out2(output)  # B x vocab_size
        #print('out_put', output[:4])
        output = self.layer_norm(output)
        vocab_dist = F.softmax(output, dim=1)
        #print('vocab_dist', vocab_dist[:4, :20])
        if args['copy']:
            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * attn_dist.view(batch_size, -1)

            # if copy_vocab.n_words >= self.vocab_size:
            final_dist = _cuda(torch.zeros((batch_size, copy_vocab.n_words)))
            final_dist[:, :self.vocab_size] = vocab_dist_
            final_dist = final_dist.scatter_add(1, copy_content, attn_dist_.view(batch_size, -1))
            #normalization_factor = final_dist.sum(1, keepdim=True)
            #final_dist = final_dist / normalization_factor

            if args['is_coverage']:
                _, v = final_dist.topk(1)
                top_attn_dist = (copy_content == v).float().view(words_score.size()) * attn_dist
                coverage = coverage + top_attn_dist

        else:
            final_dist = vocab_dist
        return final_dist, hidden, context_vector, attn_dist, p_gen, coverage, sent_attn_dist, output


