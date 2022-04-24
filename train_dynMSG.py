from tqdm import tqdm
import pickle as pkl
from config import *
from DynMSG import *
from data_process import Vocab, process_data
from data_loader import get_seq, Dataset
from cal_rouge_offline import cal_rouge
from torch.utils.tensorboard import SummaryWriter

vocab = Vocab(int(args['vocab_size']))
#copy_vocab = Vocab(1000000)

if os.path.exists(args['data_path']+'train.pkl'):
    print('loading pre-processed data...')
    all_train_data = pkl.load(open(args['data_path']+'train.pkl', 'rb'))
    all_train_words = pkl.load(open(args['data_path']+'train_words.pkl', 'rb'))
    max_train_target_len = 19
    vocab = pkl.load(open(args['data_path']+'vocab_'+str(args['vocab_size'])+'.pkl', 'rb'))
else:
    print('prepare experimental data...')
    if not os.path.exists(args['data_path']):
        os.makedirs(args['data_path'])
    all_train_data, max_train_target_len, \
        all_train_words = process_data('train', './data/AESLC/enron_subject_line/train/',
                                       vocab, int(args['max_content_len']),
                                       int(args['max_sentence_len']))
    pkl.dump(all_train_data, open(args['data_path']+'train.pkl', 'wb'))
    pkl.dump(all_train_words, open(args['data_path'] + 'train_words.pkl', 'wb'))
    pkl.dump(vocab, open(args['data_path'] + 'vocab_'+str(args['vocab_size'])+'.pkl', 'wb'))
    print('train data processing, done! max_subject_len:',  max_train_target_len)

# with open('./vocab_words_35004', 'w') as fw:
#    for key,val in vocab.word2index.items():
#        fw.write(key+'\t'+str(val)+'\n')

if os.path.exists(args['data_path']+'filtered_dev.pkl'):
    all_dev_data = pkl.load(open(args['data_path']+'filtered_dev.pkl', 'rb'))
    all_dev_words = pkl.load(open(args['data_path']+'dev_words.pkl', 'rb'))
    max_dev_target_len = 20
else:
    print('processing dev data...')
    all_dev_data, max_dev_target_len, \
        all_dev_words = process_data('dev', './data/AESLC/enron_subject_line/dev/',
                                     None, int(args['max_content_len']),
                                     int(args['max_sentence_len']))
    pkl.dump(all_dev_data, open(args['data_path']+'dev.pkl', 'wb'))
    pkl.dump(all_dev_words, open(args['data_path'] + 'dev_words.pkl', 'wb'))
    print('max_subject_len:',  max_dev_target_len)

if os.path.exists(args['data_path']+'filtered_test.pkl'):
    all_test_data = pkl.load(open(args['data_path']+'filtered_test.pkl', 'rb'))
    all_test_words = pkl.load(open(args['data_path']+'test_words.pkl', 'rb'))
    max_test_target_len = 17
else:
    print('processing test data...')
    all_test_data, max_test_target_len, \
        all_test_words = process_data('test', './data/AESLC/enron_subject_line/test/',
                                      None, int(args['max_content_len']),
                                      int(args['max_sentence_len']))
    pkl.dump(all_test_data, open(args['data_path']+'test.pkl', 'wb'))
    pkl.dump(all_test_words, open(args['data_path'] + 'test_words.pkl', 'wb'))
    print('max_subject_len:',  max_test_target_len)

#copy_vocab.add_vocab(all_train_words, all_dev_words+all_test_words)

train = get_seq(all_train_data, vocab, int(args['batch_size']))
dev = get_seq(all_dev_data, vocab, int(args['batch_size']), False)
test = get_seq(all_test_data, vocab, int(args['batch_size']), False)

model = MSG(int(args['pos_vocab_size']), int(args['embed_size']), int(args['pos_embedding_size']),
            int(args['hidden_size']), vocab, max_train_target_len, args['model_path'], float(args['learning_rate']),
            int(args['word_rnn_layer']), int(args['word_gcn_layer']), int(args['sents_rnn_layer']),
            int(args['sents_gcn_layer']), int(args['decoder_layer']), float(args['dropout']),
            int(args['beam_size']), int(args['topk']))

meteor_best = 0.0
best_dev_loss = 9999
rouge_best = 0.0
r2_best = 0.0
rl_best = 0.0
writer = SummaryWriter(filename_suffix=args['addName'])

for epoch in range(20):
    if epoch >= 2:
        args['is_coverage'] = True
    print("Epoch:{}".format(epoch))  
    # Run the train function
    pbar = tqdm(enumerate(train), total=len(train))
    for i, data in pbar:
        #print(data['copy_vocab'].n_words)
        l, lg, ls, lc = model.train_batch(data, int(args['clip']), reset=(i==0))
        pbar.set_description(model.print_loss())
    print('Start calculating the loss on dev set')

    dev_l, dev_lg, dev_ls, dev_lc = model.eval_dev(dev)

    #if (epoch+1) % int(args['evalp']) == 0:
    meteor_score_auto, meteor_score_ann, ref, hyp, ann0, ann1, ann2, direc, \
        r1_auto, r2_auto, rl_auto, r1_ann, r2_ann, rl_ann = model.evaluate(dev, best_dev_loss, False, args['earlyStop'])

    meteor_score_auto_test, meteor_score_ann_test, ref_test, hyp_test, ann0_test, ann1_test, ann2_test, direc, \
        r1_auto_test, r2_auto_test, rl_auto_test, r1_ann_test, r2_ann_test, \
            rl_ann_test = model.evaluate(test, best_dev_loss, True, args['earlyStop'])
    '''
        meteor, meteor_ann, ref, hyp, ann0, ann1, ann2, direc = model.evaluate(dev, meteor_best, 
                                                                               False, args['earlyStop'])
        _, _, ref_test, hyp_test, ann0_test, ann1_test, ann2_test, _ = model.evaluate(test, meteor_best,
                                                                                      True, args['earlyStop']
    '''
    writer.add_scalars('Loss', {'train_Loss': l, 'train_Loss_g': lg, 'train_Loss_s': ls, 'train_Loss_c': lc,
                                'dev_Loss': dev_l, 'dev_Loss_g': dev_lg, 'dev_Loss_s': dev_ls, 'dev_Loss_c': dev_lc},
                       epoch)
    writer.add_scalars('METEOR', {'dev_meteor_auto': meteor_score_auto, 'dev_meteor_ann': meteor_score_ann,
                                  'test_meteor_auto': meteor_score_auto_test, 'test_meteor_ann': meteor_score_ann_test},
                       epoch)
    writer.add_scalars('ROUGE_auto', {'dev_rouge_1': r1_auto, 'dev_rouge_2': r2_auto, 'dev_rouge_l': rl_auto,
                                      'test_rouge_1': r1_auto_test, 'test_rouge_2': r2_auto_test,
                                      'test_rouge_l': rl_auto_test}, epoch)
    writer.add_scalars('ROUGE_ann', {'dev_rouge_1': r1_ann, 'dev_rouge_2': r2_ann, 'dev_rouge_l': rl_ann,
                                     'test_rouge_1': r1_ann_test, 'test_rouge_2': r2_ann_test,
                                     'test_rouge_l': rl_ann_test}, epoch)
    file_tag = 'meteor' + str(meteor_score_auto)[:6] + 'rouge' + str(r1_auto)[:6] + '_' + str(rl_auto)[:6]
    file_tag_test = 'meteor' + str(meteor_score_auto_test)[:6] + 'rouge' + str(r1_auto_test)[:6] + \
                    '_' + str(rl_auto_test)[:6]
    if not os.path.exists(direc):
        os.makedirs(direc)
        
    with open(direc + '/dev_system_out_' + file_tag, 'w') as fw:
        for i in hyp:
            fw.write(i + '\n')
    with open(direc + '/dev_groundtruth_' + file_tag, 'w') as fw:
        for i in range(len(ref)):
            if ref[i] != '':
                fw.write(ref[i] + '\n')
            else:
                fw.write(ann0[i] + '\n')
    with open(direc + '/dev_ann_' + file_tag, 'w') as fw:
        for i, j, k in zip(ann0, ann1, ann2):
            fw.write(i + '\n' + j + '\n' + k + '\n')

    with open(direc + '/test_system_out_' + file_tag_test, 'w') as fw:
        for i in hyp_test:
            fw.write(i + '\n')
    with open(direc + '/test_groundtruth_' + file_tag_test, 'w') as fw:
        for i in ref_test:
            fw.write(i + '\n')
    with open(direc + '/test_ann_' + file_tag_test, 'w') as fw:
        for i, j, k in zip(ann0_test, ann1_test, ann2_test):
            fw.write(i + '\n' + j + '\n' + k + '\n')

    print('evaluation of dev set with pyrouge')
    r1_g, r2_g, rl_g = cal_rouge(direc + '/', 'dev_system_out_' + file_tag, 'dev_groundtruth_' + file_tag)
    r1_a, r2_a, rl_a = cal_rouge(direc + '/', 'dev_system_out_' + file_tag, 'dev_ann_' + file_tag, 3)
    print('evaluation of test set with pyrouge')
    r1_g_test, r2_g_test, rl_g_test = cal_rouge(direc + '/', 'test_system_out_' + file_tag_test,
                                                'test_groundtruth_' + file_tag_test)
    r1_a_test, r2_a_test, rl_a_test = cal_rouge(direc + '/', 'test_system_out_' + file_tag_test,
                                                      'test_ann_' + file_tag_test, 3)

    writer.add_scalars('ROUGE_auto_pyrouge', {'dev_rouge_1': r1_g, 'dev_rouge_2': r2_g, 'dev_rouge_l': rl_g,
                                              'test_rouge_1': r1_g_test, 'test_rouge_2': r2_g_test,
                                              'test_rouge_l': rl_g_test}, epoch)
    writer.add_scalars('ROUGE_ann_pyrouge', {'dev_rouge_1': r1_a, 'dev_rouge_2': r2_a, 'dev_rouge_l': rl_a,
                                             'test_rouge_1': r1_a_test, 'test_rouge_2': r2_a_test,
                                             'test_rouge_l': rl_a_test}, epoch)
    #model.word_scheduler.step(dev_loss)
    #model.sents_scheduler.step(dev_loss)
    model.decoder_scheduler.step(dev_lg)
    if dev_lg < best_dev_loss or meteor_score_auto > meteor_best or r1_auto > rouge_best \
            or r2_auto > r2_best or rl_auto > rl_best:
        cnt = 0
        best_dev_loss = min(dev_lg, best_dev_loss)
        meteor_best = max(meteor_score_auto, meteor_best)
        rouge_best = max(r1_auto, rouge_best)
        r2_best = max(r2_auto, r2_best)
        rl_best = max(rl_auto, rl_best)
        file_tag_model = 'dev_loss' + str(best_dev_loss)[:6] + 'meteor' + str(meteor_score_auto)[:6] + \
                   'rouge' + str(r1_auto)[:6]
        #file_tag_test = 'meteor' + str(meteor_score_auto_test)[:6] + 'rouge' + str(r1_auto_test)[:6]
        model.save_model(file_tag_model)
        print('MODEL SAVED!!!')
        #model.scheduler.step(meteor)

        #if meteor >= meteor_best:
        #    meteor_best = meteor
        #    cnt = 0
    else:
        cnt += 1

    if cnt == 20 or best_dev_loss == 0:
        print("Ran out of patient, early stop...")
        break

