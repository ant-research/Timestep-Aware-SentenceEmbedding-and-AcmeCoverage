import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import os
from collections import Counter
from itertools import chain
from config import *
from multiprocessing import Pool
nlp = spacy.load('en_core_web_lg')


class Vocab:
    def __init__(self, max_vocab_size):
        self.word2index = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS", 3: 'UNK', 4: 'UNKPOS'}
        self.n_words = len(self.index2word)  # Count default tokens
        self.word2index = dict([(v, k) for k, v in self.index2word.items()])
        self.max_vocab_size = max_vocab_size

    def bulid_vocab(self, story, trg=False):
        all_words = chain(*story)
        all_words_dict = Counter(all_words)
        vocab_words = all_words_dict.most_common(self.max_vocab_size)
        for word in vocab_words:
            self.index_word(word[0])

    def add_vocab(self, story, add_story, trg=False):
        all_words = chain(*story)
        all_words_dict = Counter(all_words)
        vocab_words = all_words_dict.most_common(self.max_vocab_size)
        for word in vocab_words:
            self.index_word(word[0])

        add_words = chain(*add_story)
        add_words_dict = Counter(add_words)
        add_vocab_words = add_words_dict.most_common(self.max_vocab_size)
        for word in add_vocab_words:
            if word[0] not in self.word2index:
                self.index_word(word[0])

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1


def remove_special_num(s):
    return ' '.join([re.sub(r'(.*)[0-9](.*)', 'NUM', art) for art in s.split(' ')])


def process_data(data_type, path, vocab=None, max_content_len=64, max_sents_len=128):
    all_data = []
    max_sentence_len = 0
    max_subject_len = 0
    all_words = []

    for file in os.listdir(path):
        print(file)
        data = {}
        file_content = open(path + file).read()
        file_content = file_content.replace('\\', ' ')
        sender = file.split('_')[0]
        if data_type == 'train':
            content, subject = file_content.split('\n\n@')
            ann0, ann1, ann2 = '', '', ''
        else:
            content, subject, ann0, ann1, ann2 = file_content.split('\n\n@')
            #ann0 = re.sub(r, ' ', ann0.split('\n')[1])
            #ann1 = re.sub(r, ' ', ann1.split('\n')[1])
            #ann2 = re.sub(r, ' ', ann2.split('\n')[1])

            #ann0 = re.sub(r'(.*)[0-9](.*)', ' NUM ', ann0.lower())
            #ann1 = re.sub(r'(.*)[0-9](.*)', ' NUM ', ann1.lower())
            #ann2 = re.sub(r'(.*)[0-9](.*)', ' NUM ', ann2.lower())
            ann0 = re.sub(r, ' ', remove_special_num(ann0.split('\n')[1].lower()))
            ann1 = re.sub(r, ' ', remove_special_num(ann1.split('\n')[1].lower()))
            ann2 = re.sub(r, ' ', remove_special_num(ann2.split('\n')[1].lower()))

        #content = re.sub(r, ' ', content).lower()
        #subject = re.sub(r, ' ', subject).lower()
        #content = re.sub(r'(.*)[0-9](.*)', ' NUM ', content)
        #subject = re.sub(r'(.*)[0-9](.*)', ' NUM ', subject)
        content = re.sub(r, ' ', remove_special_num(content.lower()))
        subject = re.sub(r, ' ', remove_special_num(subject.lower()))
        subject = subject.split('\n')[1]
        subject = subject.replace('\n', ' ').replace('\t', ' ').split()
        subject_len = len(subject)
        if subject_len > max_subject_len:
            max_subject_len = subject_len

        content = content.split('\n')
        content = [i.replace('\t', ' ').replace('\n', ' ').split() for i in content]
        content = list(filter(lambda x: len(x) > 0, content))
        content = [' '.join(i) for i in content]
        content, sents_adj, dep_fw, dep_bw, pos = cal_dep_adj(content)
        content_len = len(content)
        if content_len > max_content_len:
            continue

        subject_class = []
        for sub in subject:
            tmp = []
            for i in range(content_len):
                if sub in content[i]:
                    tmp.append(i)
            # if len(tmp) == 0:
            #    tmp.append(content_len)
            subject_class.append(tmp)
        subject_class.append([])
        sentence_len = [len(i) for i in content]
        if max(sentence_len) > max_sents_len:
            continue
        max_sentence_len = max(max(sentence_len), max_sentence_len)

        data['sender'] = sender
        data['content'] = content
        data['subject'] = subject
        data['content_len'] = content_len
        data['sentence_len'] = sentence_len
        data['subject_len'] = subject_len
        data['subject_class'] = subject_class
        data['sents_adj'] = sents_adj
        data['dep_fw'] = dep_fw
        data['dep_bw'] = dep_bw
        data['pos'] = pos
        data['ann0'] = ' '.join(ann0.split())
        data['ann1'] = ' '.join(ann1.split())
        data['ann2'] = ' '.join(ann2.split())
        all_data.append(data)
        if vocab:
            all_words = all_words + content + [subject] + [[sender]]
        else:
            all_words = all_words + content
    # print(max_sentence_len)
    for i in range(len(all_data)):
        tmp = []
        for sent in all_data[i]['content']:
            tmp.append(sent + ['PAD'] * (max_sentence_len - len(sent)))
        all_data[i]['content'] = tmp  # .append(['PAD']*max_sentence_len)
        pos_tmp = []
        for p in all_data[i]['pos']:
            pos_tmp.append(p + [4] * (max_sentence_len - len(p)))
        all_data[i]['pos'] = pos_tmp
        padded_dep_fw = []
        padded_dep_bw = []
        for j in range(len(all_data[i]['content'])):
            tmp_fw = np.identity(max_sentence_len)
            tmp_bw = np.identity(max_sentence_len)
            end = len(all_data[i]['dep_fw'][j])
            tmp_fw[:end, :end] = all_data[i]['dep_fw'][j]
            tmp_bw[:end, :end] = all_data[i]['dep_bw'][j]
            padded_dep_fw.append(tmp_fw)
            padded_dep_bw.append(tmp_bw)
        all_data[i]['dep_fw'] = np.greater(np.array(padded_dep_fw), 0).astype(int)
        all_data[i]['dep_bw'] = np.greater(np.array(padded_dep_bw), 0).astype(int)

    if data_type == 'train':
        vocab.bulid_vocab(all_words)
    return all_data, max_subject_len, all_words


def get_word_dep(sents):
    sent_doc = nlp(sents)
    sent_words = [token.text for token in sent_doc]
    text_tag = {}
    text_tag["text"] = sent_words

    text_tag["pos"] = []
    text_tag['pos_name'] = []
    text_tag["dep_fw"] = np.identity(len(sent_words))
    # [[0] * len(sent_words) for i in range(len(sent_words))]
    text_tag["dep_bw"] = np.identity(len(sent_words))
    # [[0] * len(sent_words) for i in range(len(sent_words))]
    for token in sent_doc:
        # pos: [seq_length]. The part-of-speech tag of each word.
        text_tag["pos"].append(token.pos)
        text_tag['pos_name'].append(token.pos_)
        # dep_fw: [seq_length, seq_length]. The dependency adjacency matrix (forward edge) of each word-pair.
        # dep_bw: [seq_length, seq_length]. The dependency adjacency matrix (backward edge) of each word-pair.
        if token.i >= token.head.i:
            text_tag["dep_fw"][token.i, token.head.i] = token.dep
        else:
            text_tag["dep_bw"][token.i, token.head.i] = token.dep

    return text_tag


def get_sents_adj(docs, threshold):
    # docs: list of each sentence in the document
    vectorizer = TfidfVectorizer()
    doc = vectorizer.fit_transform(docs).toarray()
    sents_adj = np.identity(len(doc))
    for i in range(len(doc)):
        for j in range(i+1, len(doc)):
            sents_adj[i, j] = sents_adj[j, i] = cosine_similarity(doc[i, :].reshape(1, -1),
                                                                  doc[j, :].reshape(1, -1))

    #adj = np.greater(sents_adj, threshold).astype(int)
    #adj = adj * sents_adj
    return sents_adj

def cal_dep_adj(doc):
    dep_fw = []
    dep_bw = []
    pos = []
    pos_name = []
    text = []
    sents_adj = get_sents_adj(doc, args['threshold_adj'])
    p = Pool()
    sents_dep = p.map(get_word_dep, doc)
    for tmp in sents_dep:
        # tmp = get_word_dep(sent)
        dep_fw.append(tmp['dep_fw'])
        dep_bw.append(tmp['dep_bw'])
        pos.append(tmp['pos'])
        pos_name.append(tmp['pos_name'])
        text.append(tmp['text'])
    # dep_fw = np.greater(np.array(dep_fw), 0).astype(int)
    # dep_bw = np.greater(np.array(dep_bw), 0).astype(int)
    # pos = np.array(pos)

    return text, sents_adj, dep_fw, dep_bw, pos


