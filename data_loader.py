import numpy as np
import torch
import torch.utils.data as data
import torch.nn as nn
from collections import Counter
from itertools import chain
from config import *
from config import _cuda
from data_process import get_word_dep, get_sents_adj, Vocab
import copy
import spacy
nlp = spacy.load('en_core_web_lg')


def get_spacy_tokenizer(text):
    sent_doc = nlp(' '.join(text))
    sent_words = [token.text for token in sent_doc]
    return sent_words

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data_info, vocab):
        """Reads source and target sequences from txt files."""
        self.data_info = {}
        for k in data_info.keys():
            self.data_info[k] = data_info[k]

        self.num_total_seqs = len(data_info['content'])
        self.vocab = vocab
        self.src_word2id = vocab.word2index
        self.trg_word2id = vocab.word2index
    
    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        content = self.data_info['content'][index]
        sender = self.data_info['sender'][index]
        sender = self.src_word2id[sender] if sender in self.src_word2id else 3
        content = self.preprocess(content, self.src_word2id, trg=False)
        plain_subject = get_spacy_tokenizer(self.data_info['subject'][index])
        subject = self.preprocess(plain_subject, self.trg_word2id)
        
        # processed information
        data_info = {}
        for k in self.data_info.keys():
            try:
                data_info[k] = locals()[k]
            except:
                data_info[k] = self.data_info[k][index]

        # additional plain information
        data_info['plain_content'] = self.data_info['content'][index]
        data_info['plain_subject'] = plain_subject

        return data_info

    def __len__(self):
        return self.num_total_seqs
    
    def preprocess(self, sequence, word2id, trg=True):
        """Converts words to ids."""
        if trg:
            story = [1] + [word2id[word] if word in word2id else 3 for word in sequence] + [2]
        else:
            story = []
            for i, word_triple in enumerate(sequence):
                story.append([])
                for ii, word in enumerate(word_triple):
                    temp = word2id[word] if word in word2id else 3
                    story[i].append(temp)
        story = torch.Tensor(story)
        return story

    def collate_fn(self, data):
        def process_content_subject(content, plain_content, subject, plain_subject, sents_adj, dep_fw, dep_bw, 
                                    pos, content_len, subject_class, sentence_len, normal_vocab):
            copy_vocab = copy.deepcopy(normal_vocab)
            for d in plain_content:
                for w in chain(*d):
                    copy_vocab.index_word(w)
            for w in chain(*plain_subject):
                copy_vocab.index_word(w)
            #print(copy_vocab.n_words)
            
            copy_content = []
            copy_subject = []
            for d, s in zip(plain_content, plain_subject):
                copy_content.append(self.preprocess(d, copy_vocab.word2index, False))
                copy_subject.append(self.preprocess(s, copy_vocab.word2index))
            #copy_content = torch.tensor(copy_content).contiguous()
            #copy_subject = torch.tensor(copy_subject).contiguous()
            
            each_sents_len = max(chain(*sentence_len))  # len(content[0][0])
            max_content_len = 1 if max(content_len)==0 else max(content_len)
            padded_content = torch.zeros(len(content), max_content_len, each_sents_len).long()
            padded_copy_content = torch.zeros(len(content), max_content_len, each_sents_len).long()
            padded_sentence_len = torch.ones(len(content), max_content_len, 1).long()
            tmp_eye = torch.eye(each_sents_len)
            padded_sents_adj = torch.eye(max_content_len).expand(len(content), max_content_len, max_content_len).float()
            padded_dep_fw = tmp_eye.expand(len(content), max_content_len, each_sents_len, each_sents_len).contiguous().float()
            padded_dep_bw = tmp_eye.expand(len(content), max_content_len, each_sents_len, each_sents_len).contiguous().float()
            padded_pos = torch.zeros(len(content), max_content_len, each_sents_len).long()
            content_mask = torch.zeros(len(content), max_content_len)
            for i, s in enumerate(zip(content, sentence_len, copy_content, sents_adj, dep_fw, dep_bw, pos)):
                seq = s[0]
                s_len = s[1]
                copy_seq = s[2]
                adj = s[3]
                fw = s[4]
                bw = s[5]
                p = s[6]
                end = content_len[i]
                if len(seq) != 0:
                    padded_content[i, :end, :] = seq[:end, :each_sents_len]
                    padded_pos[i, :end, :] = torch.tensor(p)[:end, :each_sents_len]
                    padded_copy_content[i, :end, :] = copy_seq[:end, :each_sents_len]
                    padded_sentence_len[i, :end, :] = torch.Tensor(s_len).view(-1, 1)
                    padded_sents_adj[i, :end, :end] = torch.tensor(adj).gt(args['threshold_adj']).float()
                    padded_dep_fw[i, :end, :, :] = torch.tensor(fw)[:end, :each_sents_len, :each_sents_len]
                    padded_dep_bw[i, :end, :, :] = torch.tensor(bw)[:end, :each_sents_len, :each_sents_len]
                    content_mask[i, :end] = 1
            sentence_mask = padded_content.gt(0).long()
            subject_len = [len(seq)-1 for seq in subject]
            max_subject_len = 1 if max(subject_len)==0 else max(subject_len)
            #max_subject_len += 1
            padded_decoder_inp = torch.zeros(len(subject), max_subject_len+1).long()
            padded_subject = torch.zeros(len(subject), max_subject_len).long()
            padded_copy_subject = torch.zeros(len(subject), max_subject_len).long()
            for i, s in enumerate(zip(subject, copy_subject)):
                seq = s[0]
                copy_seq = s[1]
                end = subject_len[i] #+ 1
                padded_decoder_inp[i, :end+1] = seq[:end+1]
                padded_subject[i, :end] = seq[1:end+1]
                padded_copy_subject[i, :end] = copy_seq[1:end+1]
                
            padded_subject_class = torch.zeros(len(content), max_subject_len, max_content_len).float()
            for i, subject_c in enumerate(subject_class):
                for j, subject in enumerate(subject_c):
                    padded_subject_class[i, j, subject] = 1
                    break  # only the first sentence contains target word will be used!
                    
            subject_class_mask = torch.zeros(len(content), max_subject_len, max_content_len).long()
            for i, j in enumerate(zip(subject_len, content_len)):
                subject_class_mask[i, :j[0], :j[1]] = 1
            
            return padded_content, padded_sentence_len, content_mask, sentence_mask, subject_len,\
                   padded_subject, padded_subject_class, subject_class_mask, padded_copy_content, \
                   padded_copy_subject, padded_pos, padded_dep_bw, padded_dep_bw, padded_sents_adj,\
                   padded_decoder_inp, copy_vocab
        
        # sort a list by sequence length (descending order) to use pack_padded_sequence
        data.sort(key=lambda x: len(x['content']), reverse=True) 
        item_info = {}
        for key in data[0].keys():
            item_info[key] = [d[key] for d in data]

        # merge sequences 
        content, sentence_len, content_mask, sentence_mask, subject_len, subject, \
        subject_class, subject_class_mask, copy_content, copy_subject, pos, dep_fw, \
        dep_bw, sents_adj, decoder_inp, copy_vocab = process_content_subject(item_info['content'], item_info['plain_content'],
                                                                             item_info['subject'], item_info['plain_subject'],
                                                                             item_info['sents_adj'], item_info['dep_fw'],
                                                                             item_info['dep_bw'], item_info['pos'],
                                                                             item_info['content_len'], item_info['subject_class'],
                                                                             item_info['sentence_len'], self.vocab)

        #content_len = torch.Tensor(item_info['content_len'])
        #sentence_len = torch.Tensor(item_info['sentence_len'])
        
        # convert to contiguous and cuda
        content = _cuda(content.contiguous())
        copy_content = _cuda(copy_content.contiguous())
        sentence_mask = _cuda(sentence_mask.contiguous())
        #subject_len = _cuda(subject_len.contiguous())
        subject = _cuda(subject.contiguous())
        copy_subject = _cuda(copy_subject.contiguous())
        decoder_inp = _cuda(decoder_inp.contiguous())
        subject_class = _cuda(subject_class.contiguous())
        subject_class_mask = _cuda(subject_class_mask.contiguous())
        #content_len = _cuda(content_len.contiguous())
        sentence_len = _cuda(sentence_len.contiguous())
        content_mask = _cuda(content_mask.contiguous())
        pos = _cuda(pos.contiguous())
        dep_fw = _cuda(dep_fw.contiguous())
        dep_bw = _cuda(dep_bw.contiguous())
        sents_adj = _cuda(sents_adj.contiguous())
        # processed information
        data_info = {}
        for k in item_info.keys():
            try:
                data_info[k] = locals()[k]
            except:
                data_info[k] = item_info[k]

        # additional plain information
        data_info['copy_content'] = copy_content
        data_info['copy_subject'] = copy_subject
        data_info['copy_vocab'] = copy_vocab
        data_info['decoder_inp'] = decoder_inp
        data_info['content_mask'] = content_mask
        data_info['sentence_mask'] = sentence_mask
        data_info['subject_class_mask'] = subject_class_mask
        return data_info


def get_seq(pairs, vocab, batch_size, is_train=True):
    data_info = {}
    for k in pairs[0].keys():
        data_info[k] = []
    
    for pair in pairs:
        for k in pair.keys():
            data_info[k].append(pair[k])
            
    dataset = Dataset(data_info, vocab)
    data_loader = torch.utils.data.DataLoader(dataset = dataset,
                                              batch_size = batch_size,
                                              shuffle = is_train,
                                              collate_fn = dataset.collate_fn)
    return data_loader
