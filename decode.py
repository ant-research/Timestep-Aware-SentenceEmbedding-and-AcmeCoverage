import os
from tqdm import tqdm
import numpy as np
import random
import pickle as pkl
from config import *
from DynMSG import *
from data_process import Vocab, process_data
from data_loader import get_seq, Dataset

direc = './decode_res'
if not os.path.exists(direc):
    os.makedirs(direc)
vocab = pkl.load(open(args['data_path']+'vocab_'+str(args['vocab_size'])+'.pkl', 'rb'))

if os.path.exists(args['data_path']+'train.pkl'):
    all_dev_data = pkl.load(open(args['data_path']+'train.pkl', 'rb'))
    all_dev_words = pkl.load(open(args['data_path']+'train_words.pkl', 'rb'))
    max_dev_target_len = 20
else:
    print('processing dev data...')
    all_dev_data, max_dev_target_len, \
        all_dev_words = process_data('dev', '../../data/AESLC/enron_subject_line/dev/',
                                     None, int(args['max_content_len']),
                                     int(args['max_sentence_len']))
    pkl.dump(all_dev_data, open(args['data_path']+'dev.pkl', 'wb'))
    pkl.dump(all_dev_words, open(args['data_path'] + 'dev_words.pkl', 'wb'))
    print('max_subject_len:',  max_dev_target_len)

if os.path.exists(args['data_path']+'test.pkl'):
    all_test_data = pkl.load(open(args['data_path']+'test.pkl', 'rb'))
    all_test_words = pkl.load(open(args['data_path']+'test_words.pkl', 'rb'))
    max_test_target_len = 17
else:
    print('processing test data...')
    all_test_data, max_test_target_len, \
        all_test_words = process_data('test', '../../data/AESLC/enron_subject_line/test/',
                                      None, int(args['max_content_len']),
                                      int(args['max_sentence_len']))
    pkl.dump(all_test_data, open(args['data_path']+'test.pkl', 'wb'))
    pkl.dump(all_test_words, open(args['data_path'] + 'test_words.pkl', 'wb'))
    print('max_subject_len:',  max_test_target_len)

#copy_vocab.add_vocab(all_train_words, all_dev_words+all_test_words)

#train = get_seq(all_train_data, vocab, int(args['batch_size']))
dev = get_seq(random.sample(all_dev_data, 2000), vocab, int(args['batch_size']), False)
test = get_seq(all_test_data, vocab, int(args['batch_size']), False)

model = MSG(int(args['pos_vocab_size']), int(args['embed_size']), int(args['pos_embedding_size']),
            int(args['hidden_size']), vocab, 20, args['model_path'], float(args['learning_rate']),
            int(args['word_rnn_layer']), int(args['word_gcn_layer']), int(args['sents_rnn_layer']),
            int(args['sents_gcn_layer']), int(args['decoder_layer']), float(args['dropout']),
            int(args['beam_size']), int(args['topk']))

#model.word_graph.train(False)
#model.sents_graph.train(False)
#model.decoder.train(False)

def get_evaluation_result():
    meteor_score_auto, meteor_score_ann, ref, hyp, ann0, ann1, ann2, _, \
        r1_auto, r2_auto, rl_auto, r1_ann, r2_ann, rl_ann = model.evaluate(dev, 9999, False, args['earlyStop'])

    meteor_score_auto_test, meteor_score_ann_test, ref_test, hyp_test, ann0_test, ann1_test, ann2_test, _, \
        r1_auto_test, r2_auto_test, rl_auto_test, r1_ann_test, r2_ann_test, \
            rl_ann_test = model.evaluate(test, 9999, True, args['earlyStop'])
    '''
    file_tag = 'meteor' + str(meteor_score_auto)[:6] + 'rouge' + str(r1_auto)[:6] + '_' +str(r1_ann)[:6]
    file_tag_test = 'meteor' + str(meteor_score_auto_test)[:6] + 'rouge' + str(r1_auto_test)[:6] + \
                    '_' +str(r1_ann_test)[:6]
    with open(direc+'/dev_system_out_' + file_tag, 'w') as fw:
        for i in hyp:
            fw.write(i+'\n')
    with open(direc+'/dev_groundtruth_' + file_tag, 'w') as fw:
        for i in ref:
            fw.write(i+'\n')
    with open(direc+'/dev_ann_' + file_tag, 'w') as fw:
        for i, j, k in zip(ann0, ann1, ann2):
            fw.write(i+'\n'+j+'\n'+k+'\n')

    with open(direc+'/test_system_out_' + file_tag_test, 'w') as fw:
        for i in hyp_test:
            fw.write(i+'\n')
    with open(direc+'/test_groundtruth_' + file_tag_test, 'w') as fw:
        for i in ref_test:
            fw.write(i+'\n')
    with open(direc+'/test_ann_' + file_tag_test, 'w') as fw:
        for i, j, k in zip(ann0_test, ann1_test, ann2_test):
            fw.write(i+'\n'+j+'\n'+k+'\n')
    '''

if __name__ == '__main__':
    print("=======starting to evaluate the model's performance======")
    get_evaluation_result()


