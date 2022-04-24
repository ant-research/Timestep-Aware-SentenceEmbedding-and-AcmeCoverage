import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch import optim
import torch.nn.functional as F
import random
import numpy as np
import os, subprocess
import json
from queue import PriorityQueue
import operator
import math

from loss_function import *
from config import *
from config import _cuda
from msg_graph import *
from rouge_measure import cal_rouge, cal_rouge_v2
from cal_rouge_offline import cal_rouge as new_cal_rouge


class MSG(nn.Module):
    def __init__(self, pos_vocab_size, embedding_size, pos_embedding_size, hidden_size, vocab,
                 max_subject_len, path, lr, word_rnn_layer, word_gcn_layer, sents_rnn_layer, sents_gcn_layer,
                 decoder_layers, dropout, beam_size, topk):
        super(MSG, self).__init__()
        self.name = "MSG"
        self.input_size = vocab.n_words
        self.output_size = vocab.n_words
        self.pos_vocab_size = pos_vocab_size
        self.embedding_size = embedding_size
        self.pos_embedding_size = pos_embedding_size
        self.hidden_size = hidden_size
        self.vocab = vocab
        self.lr = lr
        self.word_rnn_layer = word_rnn_layer
        self.word_gcn_layer = word_gcn_layer
        self.sents_rnn_layer = sents_rnn_layer
        self.sents_gcn_layer = sents_gcn_layer
        self.decoder_layers = decoder_layers
        self.dropout = dropout
        self.beam_size = beam_size
        self.topk = topk
        self.max_subject_len = max_subject_len
        self.softmax = nn.Softmax(dim=0)

        layer_info = str(self.word_rnn_layer) + '_' + str(self.word_gcn_layer) + '_' + str(self.decoder_layers)
        self.name_suffix = str(args['addName']) + 'EMS' + str(self.embedding_size) + 'HDS' + str(self.hidden_size) + \
                            'BSZ' + str(args['batch_size']) + 'DR' + str(self.dropout) + 'L' + layer_info + \
                            'lr' + str(self.lr) + 'MODE' + str(args['mode']) + 'DER' + str(args['decay_rate']) +\
                            'minlr' + str(args['min_lr']) + 'sender' + str(args['user']) + \
                            'TFR' + str(args['teacher_forcing_ratio']) + 'beam_size' + str(args['beam_size']) + \
                            'copy' + str(args['copy']) + 'CLIP' + str(args['clip']) + 'hard' + str(args['hard']) + \
                            'thres' + str(args['thres']) + 'coverage' + str(args['is_coverage'])

        name_data = "MSG/"
        self.directory = 'save/' + args["addName"] + name_data + self.name_suffix

        if path:
            if args['use_cuda']:
                print("MODEL {} LOADED".format(str(path)))
                self.word_graph = torch.load(str(path) + '_word.th')
                self.sents_graph = torch.load(str(path) + '_sent.th')
                self.decoder = torch.load(str(path) + '_dec.th')
            else:
                print("MODEL {} LOADED".format(str(path)))
                self.word_graph = torch.load(str(path) + '_word.th', lambda storage, loc: storage)
                self.sents_graph = torch.load(str(path) + '_sent.th', lambda storage, loc: storage)
                self.decoder = torch.load(str(path) + '_dec.th', lambda storage, loc: storage)
        else:
            self.word_graph = WordGraphEncoder(vocab.n_words, pos_vocab_size, embedding_size, pos_embedding_size,
                                               hidden_size, word_rnn_layer, word_gcn_layer, dropout)
            self.sents_graph = SentsGraphEncoder(hidden_size, sents_rnn_layer, sents_gcn_layer, dropout)

            self.decoder = DynDecoder(self.word_graph.embedding, embedding_size, vocab,
                                      hidden_size, dropout, decoder_layers)

            # Initialize optimizers and criterion
        self.word_graph_optimizer = optim.Adam(self.word_graph.parameters(), lr=lr)
        self.sents_graph_optimizer = optim.Adam(self.sents_graph.parameters(), lr=lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.decoder_optimizer,
                                                        mode='max', factor=float(args['decay_rate']),
                                                        patience=2, min_lr=float(args['min_lr']),
                                                        verbose=True)
        self.word_scheduler = lr_scheduler.ReduceLROnPlateau(self.word_graph_optimizer,
                                                             factor=float(args['decay_rate']),
                                                             patience=2, min_lr=float(args['min_lr']),
                                                             verbose=True)
        self.sents_scheduler = lr_scheduler.ReduceLROnPlateau(self.sents_graph_optimizer,
                                                              factor=float(args['decay_rate']),
                                                              patience=2, min_lr=float(args['min_lr']),
                                                              verbose=True)
        self.decoder_scheduler = lr_scheduler.ReduceLROnPlateau(self.decoder_optimizer,
                                                                factor=float(args['decay_rate']),
                                                                patience=2, min_lr=float(args['min_lr']),
                                                                verbose=True)

        self.reset()

        if args['use_cuda']:
            self.word_graph.cuda()
            self.sents_graph.cuda()
            self.decoder.cuda()

    def get_decoder_input(self, topvi):
        decoder_input = []
        for i in range(topvi.size(0)):
            if topvi[i].item() < len(self.vocab.word2index):
                decoder_input.append(topvi[i].item())
            else:
                decoder_input.append(self.vocab.word2index['UNK'])
        return _cuda(torch.LongTensor(decoder_input))

    def print_loss(self):
        print_loss_avg = self.loss / self.print_every
        print_loss_g = self.loss_g / self.print_every
        print_loss_s = self.loss_s / self.print_every
        print_loss_c = self.loss_c / self.print_every
        self.print_every += 1
        return 'L:{:.4f},LG:{:.4f},LS:{:.4f},LC:{:.4f}'.format(print_loss_avg, print_loss_g,
                                                               print_loss_s, print_loss_c)

    def save_model(self, model_name):
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        torch.save(self.word_graph, self.directory + '/'+model_name+'_word.th')
        torch.save(self.sents_graph, self.directory + '/'+model_name+'_sent.th')
        torch.save(self.decoder, self.directory + '/'+model_name+'_dec.th')

    def reset(self):
        self.loss, self.print_every, self.loss_g, self.loss_s, self.loss_c = 0, 1, 0, 0, 0

    def cal_loss(self, data, prob, predict_class):
        if args['copy']:
            loss_g = masked_cross_entropy(
                prob.contiguous(),
                data['copy_subject'].contiguous(),
                data['subject_len'])
        else:
            loss_g = masked_cross_entropy(
                prob.contiguous(),
                data['subject'].contiguous(),
                data['subject_len'])
        loss_s = masked_binary_cross_entropy(
            predict_class.contiguous(),
            data['subject_class'].contiguous(),
            data['subject_len'])
        return loss_g, loss_s

    def train_batch(self, data, clip, reset=0):
        if reset: self.reset()
        # Zero gradients of both optimizers
        self.word_graph_optimizer.zero_grad()
        self.sents_graph_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        # Encode and Decode
        use_teacher_forcing = random.random() < args['teacher_forcing_ratio']
        max_target_length = max(data['subject_len']) #+ 1

        all_decoder_outputs_vocab, predict_class, _, coverage_loss = \
            self.encode_and_decode(data, max_target_length, 'train')

        # Loss calculation and backpropagation
        loss_g, loss_s = self.cal_loss(data, all_decoder_outputs_vocab, predict_class)
        if args['copy']:
            coverage_loss = cal_coverage_loss(coverage_loss, data['subject_len'], data['copy_subject'])
        else:
            coverage_loss = cal_coverage_loss(coverage_loss, data['subject_len'], data['subject'])
        loss = loss_g + 0 * loss_s
        if args['is_coverage']:
            loss += args['coverage_weight'] * coverage_loss
        loss.backward()

        # Clip gradient norms
        wgc = torch.nn.utils.clip_grad_norm_(self.word_graph.parameters(), clip)
        sgc = torch.nn.utils.clip_grad_norm_(self.sents_graph.parameters(), clip)
        dc = torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), clip)

        # Update parameters with optimizers
        self.word_graph_optimizer.step()
        self.sents_graph_optimizer.step()
        self.decoder_optimizer.step()
        self.loss += loss.item()
        self.loss_g += loss_g.item()
        self.loss_s += loss_s.item()
        self.loss_c += coverage_loss.item()
        return self.loss / self.print_every, self.loss_g / self.print_every, \
               self.loss_s / self.print_every, self.loss_c / self.print_every

    def eval_dev(self, data):
        self.reset()

        self.word_graph.train(False)
        self.sents_graph.train(False)
        self.decoder.train(False)

        pbar = tqdm(enumerate(data), total=len(data))
        new_precision, new_recall, new_f1_score = 0, 0, 0

        for j, batch_data in pbar:
            max_target_length = max(batch_data['subject_len']) #+ 1

            all_decoder_outputs_vocab, predict_class, _, coverage_loss = \
                self.encode_and_decode(batch_data, max_target_length, 'dev')

            # Loss calculation but not back propogation
            loss_g, loss_s = self.cal_loss(batch_data, all_decoder_outputs_vocab, predict_class)
            loss = loss_g + loss_s

            if args['is_coverage']:
                if args['copy']:
                    target = batch_data['copy_subject']
                else:
                    target = batch_data['subject']
                coverage_loss = cal_coverage_loss(coverage_loss, batch_data['subject_len'], target)
            else:
                coverage_loss = torch.tensor(0)
            loss += args['coverage_weight'] * coverage_loss

            self.loss += loss.item()
            self.loss_g += loss_g.item()
            self.loss_s += loss_s.item()
            self.loss_c += coverage_loss.item()
            pbar.set_description(self.print_loss())

        self.word_graph.train(True)
        self.sents_graph.train(True)
        self.decoder.train(True)

        return self.loss / (j + 1), self.loss_g / (j + 1), self.loss_s / (j + 1), self.loss_c / (j + 1)

    def encode_and_decode(self, data, max_target_length, mode='train'):
        #print(max_target_length)
        words_encoder_output, last_words_encoder_outputs, hidden_words = self.word_graph(data['content'], data['pos'],
                                                                                         data['dep_fw'], data['dep_bw'],
                                                                                         data['sentence_len'])
        batch_size, T_d, T_s, hidden_dim = words_encoder_output.size()
        words_encoder_output = words_encoder_output.view(batch_size*T_d, T_s, -1)
        words_encoder_feature = self.word_graph.W_word(words_encoder_output.contiguous().view(-1, hidden_dim))
        words_mask = data['sentence_mask'].view(-1, T_s) # (B*T_d) x T_s
        coverage = _cuda(torch.zeros(batch_size, T_d, T_s))
        sents_mask = data['content_mask'].view(-1, T_d) # B x T_d
        context_vector = _cuda(torch.zeros(batch_size, hidden_dim))
        coverage_loss = []
        predict_class = []
        predict_word = []
        predict_word_prob = []
        decoder_input = data['decoder_inp'][:, 0]

        for t in range(1, max_target_length+1):
            if t > 1:
                hidden_words = hidden[-1].unsqueeze(1).expand(batch_size, T_d,
                                                              hidden_dim).contiguous().view(-1, hidden_dim)
            sents_encoder_input, words_attn_dist, _, words_score = \
                self.word_graph.word_attention_network(hidden_words, words_encoder_output, words_encoder_feature,
                                                       words_mask, coverage)
            words_attn_dist = words_attn_dist.contiguous().view(batch_size, T_d, T_s)
            words_score = words_score.contiguous().view(batch_size, T_d, T_s)
            sents_encoder_input = sents_encoder_input.view(batch_size, T_d, -1)
            
            sents_encoder_output, hidden_sent = self.sents_graph(sents_encoder_input, data['sents_adj'],
                                                    data['content_len'])
            sents_encoder_feature = self.decoder.W_sent(sents_encoder_output.view(-1, hidden_dim))

            if t == 1:
                #hidden = _cuda(torch.zeros(2, batch_size, hidden_dim)) #hidden_sents
                hidden = hidden_sent

            final_dist_t, hidden, context_vector, attn_dist, p_gen, next_coverage, sents_attn_dist, \
                logits = self.decoder(decoder_input, hidden, sents_encoder_output, sents_encoder_feature,
                                      sents_mask, data['sentence_mask'], data['copy_content'].view(batch_size, -1),
                                      context_vector, data['copy_vocab'], coverage, words_score, words_attn_dist, t)

            #_, topvi = final_dist_t.data.topk(1)
            #print(topvi.squeeze().data)

            if args['is_coverage']:
                coverage_loss.append(torch.sum(torch.min(attn_dist, coverage), (1, 2)))
                coverage = next_coverage
            else:
                coverage_loss.append(torch.sum(coverage, (1, 2)))

            predict_class.append(sents_attn_dist.unsqueeze(1))
            predict_word_prob.append(final_dist_t.unsqueeze(1))
            if mode == 'test':
                _, topvi = final_dist_t.data.topk(1)
                topvi = topvi.squeeze(0)
                temp_c = []
                for bi in range(batch_size):
                    token = topvi[bi].item() #topvi[:,0][bi].item()
                    if args['copy']:
                        temp_c.append(data['copy_vocab'].index2word[token])
                    else:
                        temp_c.append(self.vocab.index2word[token])
                predict_word.append(temp_c)
            if mode != 'test':
                decoder_input = data['decoder_inp'][:, t]
            else:
                if args['copy']:
                    decoder_input = self.get_decoder_input(topvi)
                else:
                    decoder_input = topvi.squeeze()
        if mode == 'train':
            return _cuda(torch.cat(predict_word_prob, 1)), torch.cat(predict_class, 1), \
                   predict_word, torch.stack(coverage_loss, 1)
        elif mode == 'dev':
            return _cuda(torch.cat(predict_word_prob, 1)), torch.cat(predict_class, 1), \
                   predict_word, torch.stack(coverage_loss, 1)
        else:
            return predict_word

    def beam_search(self, data, beam_size, topk, max_target_len):
        words_encoder_output, last_words_encoder_outputs, hidden_words = self.word_graph(data['content'], data['pos'],
                                                                                   data['dep_fw'], data['dep_bw'],
                                                                                   data['sentence_len'])
        batch_size, T_d, T_s, hidden_dim = words_encoder_output.size()
        words_encoder_output = words_encoder_output.view(batch_size*T_d, T_s, -1)
        words_encoder_feature = self.word_graph.W_word(words_encoder_output.contiguous().view(-1, hidden_dim))
        words_mask = data['sentence_mask'].view(-1, T_s) # (B*T_d) x T_s
        coverage = _cuda(torch.zeros(batch_size, T_d, T_s))
        sents_mask = data['content_mask'].view(-1, T_d) # B x T_d
        context_vector = _cuda(torch.zeros(batch_size, hidden_dim))
        sender = data['sender']
        if args['copy']:
            beam_vocab = data['copy_vocab']
        else:
            beam_vocab = self.vocab
        predict_word = []

        for idx in range(batch_size):
            hidden_words_idx = hidden_words[idx*T_d:(idx+1)*T_d, :]
            hidden_idx = _cuda(torch.zeros(2, 1, hidden_dim))
            words_encoder_output_idx = words_encoder_output[idx*T_d:(idx+1)*T_d, :, :]
            words_encoder_feature_idx = words_encoder_feature[idx*T_d*T_s:(idx+1)*T_d*T_s, :]
            words_mask_idx = words_mask[idx*T_d:(idx+1)*T_d, :]
            words_mask_idx_2 = data['sentence_mask'][idx, :, :]
            coverage_idx = coverage[idx, :, :].unsqueeze(0)
            sents_mask_idx = sents_mask[idx, :].unsqueeze(0)
            context_vector_idx = context_vector[idx, :].unsqueeze(0)
            copy_content_idx = data['copy_content'][idx, :, :].view(1, -1)
            sents_adj_idx = data['sents_adj'][idx, :, :].unsqueeze(0)
            content_len_idx = [data['content_len'][idx]]

            if not args['user']:
                decoder_input = _cuda(torch.LongTensor([1]))
            else:
                decoder_input = _cuda(sender[idx].view(1))
            # print(decoder_input.size())

            # Number of sentence to generate
            endnodes = []
            number_required = min((topk + 1), topk - len(endnodes))

            # starting node -  hidden vector, previous node, word id, logp, length
            node = BeamSearchNode(hidden_words_idx, hidden_idx, context_vector_idx, coverage_idx,
                                  None, decoder_input, 0, 1)
            nodes = PriorityQueue()

            # start the queue
            nodes.put(Prioritize(-node.cal(), node))
            qsize = 1
            # start beam search
            while True:
                # give up when decoding takes too long
                if qsize > 1000: break

                # fetch the best node
                tmp = nodes.get()
                score = tmp.priority
                n = tmp.item
                decoder_input = n.wordid
                hidden_idx = n.h_sent
                hidden_words_idx = n.h_word
                context_vector_idx = n.c_t
                coverage_idx = n.coverage
                avg_score = n.avg_score

                if (n.wordid.item() == 2 or n.len > max_target_len) and n.prevNode != None:
                    endnodes.append((avg_score, n))
                    # if we reached maximum # of sentences required
                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue

                # decode for one step using decoder
                sents_encoder_input, words_attn_dist, _, words_score = \
                    self.word_graph.word_attention_network(hidden_words_idx, words_encoder_output_idx,
                                                           words_encoder_feature_idx,
                                                           words_mask_idx, coverage_idx)
                words_score = words_score.unsqueeze(0)
                words_attn_dist = words_attn_dist.contiguous().view(1, T_d, T_s)
                sents_encoder_input = sents_encoder_input.view(1, T_d, -1)

                sents_encoder_output, hidden_sents_idx = self.sents_graph(sents_encoder_input, sents_adj_idx,
                                                                         content_len_idx)
                if qsize == 1:
                    hidden_idx = hidden_sents_idx
                sents_encoder_feature = self.decoder.W_sent(sents_encoder_output.view(-1, hidden_dim))

                final_dist_t, hidden_idx, context_vector_idx, attn_dist, p_gen, coverage_idx, sents_attn_dist, \
                    logits = self.decoder(decoder_input, hidden_idx, sents_encoder_output, sents_encoder_feature,
                                          sents_mask_idx, words_mask_idx_2, copy_content_idx, context_vector_idx,
                                          beam_vocab, coverage_idx, words_score, words_attn_dist, 0)
                p_vocab = final_dist_t

                # PUT HERE REAL BEAM SEARCH OF TOP
                log_prob, indexes = torch.topk(p_vocab, beam_size)
                nextnodes = []

                for new_k in range(beam_size):
                    if indexes[0][new_k].item() < self.vocab.n_words:
                        decoded_t = _cuda(indexes[0][new_k].view(1))
                    else:
                        decoded_t = _cuda(torch.LongTensor([self.vocab.word2index['UNK']]))
                    log_p = np.log(log_prob[0][new_k].item())
                    hidden_words_idx = hidden_idx[-1].unsqueeze(1).expand(1, T_d,
                                                                          hidden_dim).contiguous().view(-1, hidden_dim)
                    node = BeamSearchNode(hidden_words_idx, hidden_idx, context_vector_idx, coverage_idx,
                                          n, decoded_t, n.logp + log_p, n.len + 1)
                    score = -node.cal()
                    nextnodes.append((score, node))

                # put them into queue
                for i in range(len(nextnodes)):
                    score, nn = nextnodes[i]
                    nodes.put(Prioritize(score, nn))
                    # increase qsize
                qsize += len(nextnodes) - 1

            # choose nbest paths, back trace them
            if len(endnodes) == 0:
                endnodes = []
                for _ in range(topk):
                    tmp_node = nodes.get()
                    endnodes.append((tmp_node.item.avg_score, tmp_node.item))
                # endnodes = [nodes.get() for _ in range(topk)]

            utterances = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterance = []
                utterance.append(beam_vocab.index2word[n.wordid.item()])
                # back trace
                while n.prevNode != None:
                    n = n.prevNode
                    utterance.append(beam_vocab.index2word[n.wordid.item()])

                utterance = utterance[::-1]
                utterances.append(utterance[1:])

            predict_word.append(utterances)
        return predict_word

    def evaluate(self, dev, matric_best, is_test=False, early_stop=None):
        if is_test:
            prefix = 'test'
            print("STARTING TEST")
        else:
            prefix = 'dev'
            print("STARTING EVALUATION")
        # Set to not-training mode to disable dropout
        self.word_graph.train(False)
        self.sents_graph.train(False)
        self.decoder.train(False)

        ann0, ann1, ann2, ref, hyp = [], [], [], [], []
        loss, loss_g, loss_s, coverage_loss = [], [], [], []

        pbar = tqdm(enumerate(dev), total=len(dev))
        new_precision, new_recall, new_f1_score = 0, 0, 0

        for j, data_dev in pbar:
            # Encode and Decode
            #max_target_length = math.ceil(np.mean(data_dev['subject_len'])) + 1
            max_target_length = max(data_dev['subject_len']) #+ 1
            if self.beam_size >= 1:
                decoded_words = self.beam_search(data_dev, self.beam_size, self.topk, max_target_length)
                decoded_words = [i[0] for i in decoded_words]
            else:
                decoded_words = self.encode_and_decode(data_dev, max_target_length, 'test')
                decoded_words = np.transpose(decoded_words)

            for bi, row in enumerate(decoded_words):
                st = ''
                for e in row:
                    if e == 'EOS':
                        break
                    else:
                        st += e + ' '
                pred_sent = st.lstrip().rstrip()
                gold_sent = ' '.join(data_dev['plain_subject'][bi])
                ann0.append(data_dev['ann0'][bi])#[4:])
                ann1.append(data_dev['ann1'][bi])#[4:])
                ann2.append(data_dev['ann2'][bi])#[4:])
                ref.append(gold_sent)
                hyp.append(pred_sent)

                if args['genSample']:
                    self.print_examples(bi, data_dev, pred_sent, pred_sent, gold_sent)

        # Set back to training mode
        self.word_graph.train(True)
        self.sents_graph.train(True)
        self.decoder.train(True)

        with open('./res/' + prefix + '_system_out_' + self.name_suffix, 'w') as fw:
            for i in hyp:
                fw.write(i + '\n')
        with open('./res/' + prefix + '_groundtruth_' + self.name_suffix, 'w') as fw:
            for i in range(len(ref)):
                if ref[i] != '':
                    fw.write(ref[i] + '\n')
                else:
                    fw.write(ann0[i] + '\n')
        with open('./res/' + prefix + '_ann_2_3_' + self.name_suffix, 'w') as fw:
            for i, j in zip(ann1, ann2):
               fw.write(i + '\n' + j + '\n')

        meteor_score_auto = subprocess.check_output(['/bin/sh', './auto_meteor.sh', prefix, self.name_suffix],
                                                    stderr=subprocess.STDOUT)
        meteor_score_ann = subprocess.check_output(['/bin/sh', './annotation_meteor.sh', prefix, self.name_suffix],
                                                   stderr=subprocess.STDOUT)
        meteor_score_auto = float(meteor_score_auto.decode('utf-8').strip().split(' ')[-1])
        meteor_score_ann = float(meteor_score_ann.decode('utf-8').strip().split(' ')[-1])
        print("METEOR SCORE:\t" + str(meteor_score_auto) + '\t' + str(meteor_score_ann))
        print("ROUGE-auto:")
        r1_auto, r2_auto, rl_auto = cal_rouge('./res/' + prefix + '_system_out_' + self.name_suffix,
                                              './res/' + prefix + '_groundtruth_' + self.name_suffix)

        #r1_g, r2_g, rl_g = new_cal_rouge('./res/', prefix + '_system_out_' + self.name_suffix,
        #                                 prefix + '_groundtruth_' + self.name_suffix)
        print("ROUGE-ann:")
        r1_ann, r2_ann, rl_ann = cal_rouge('./res/' + prefix + '_system_out_' + self.name_suffix,
                                           './res/' + prefix + '_ann_2_3_' + self.name_suffix, 2)
        #r1_g, r2_g, rl_g = new_cal_rouge('./res/', prefix + '_system_out_' + self.name_suffix,
        #                                 prefix + '_ann_2_3_' + self.name_suffix, 2)
        return meteor_score_auto, meteor_score_ann, ref, hyp, ann0, ann1, ann2, self.directory, \
               r1_auto, r2_auto, rl_auto, r1_ann, r2_ann, rl_ann
        '''
        if early_stop == 'METEOR':
            #if (meteor_score_auto >= matric_best) and not is_test:
            #    self.save_model('meteor-' + str(meteor_score_auto))
            #    print("MODEL SAVED")
            return meteor_score_auto, meteor_score_ann, ref, hyp, ann0, ann1, ann2, self.directory
        elif early_stop == 'ROUGE':
            #if (rouge >= matric_best) and not is_test:
            #    self.save_model('rouge-' + str(rouge))
            #    print("MODEL SAVED")
            return rouge_auto, rouge_ann, ref, hyp, ann0, ann1, ann2
        '''

    def print_examples(self, batch_idx, data, pred_sent, gold_sent):
        print('Final System Response : ', pred_sent)
        print('Gold System Response : ', gold_sent)
        print('\n')

