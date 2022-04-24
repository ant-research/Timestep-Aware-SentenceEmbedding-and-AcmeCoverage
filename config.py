import os
import argparse
from tqdm import tqdm

PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3
UNKPOS_token = 4

MAX_LENGTH = 10
r = '[!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~]+'

parser = argparse.ArgumentParser(description='Seq_TO_Seq Subject Generation')
parser.add_argument('-uc','--use_cuda', help='use cuda', required=False, type=bool, default=True)
parser.add_argument('-up','--use_pos', help='use POS', required=False, type=bool, default=True)
parser.add_argument('-coverage','--is_coverage', help='use coverage', required=False, type=bool, default=False)
parser.add_argument('-cov_weight','--coverage_weight', help='the weight of coverage loss', required=False, type=float,
                    default=1.0)
parser.add_argument('-msl','--max_sentence_len', help='maximum length of each sentence', required=False, type=int,
                    default=128)
parser.add_argument('-mcl','--max_content_len', help='maxumum length of each mail', required=False, type=int,
                    default=64)
parser.add_argument('-dec','--decoder', help='decoder model', required=False)
parser.add_argument('-hdd','--hidden_size', help='Hidden size', required=False, type=int, default=256)
parser.add_argument('-bsz','--batch_size', help='Batch_size', required=False, type=int, default=8)
parser.add_argument('-lr','--learning_rate', help='Learning Rate', required=False, type=float, default=0.0005)
parser.add_argument('-dlr','--decay_rate', help='Leaning Rate Decay Rate', required=False, type=float, default=0.95)
parser.add_argument('-mlr','--min_lr', help='Minimum Learning Rate', required=False, type=float, default=0.00005)
parser.add_argument('-dr','--dropout', help='Drop Out', required=False, type=float, default=0.5)
parser.add_argument('-ruim','--rand_unif_init_mag', help='random magnitude of unif init', required=False,
                    type=float, default=0.02)
parser.add_argument('-tnis','--trunc_norm_init_std', help='random magnitude of unif init', required=False,
                    type=float, default=1e-4)
parser.add_argument('-um','--unk_mask', help='mask out input token to UNK', type=int, required=False, default=1)
parser.add_argument('-wrl','--word_rnn_layer', help='Encoder RNN Layer Number', required=False, type=int, default=2)
parser.add_argument('-wgl','--word_gcn_layer', help='Encoder GCN Layer Number', required=False, type=int, default=2)
parser.add_argument('-srl','--sents_rnn_layer', help='Encoder RNN Layer Number of sents', required=False, type=int,
                    default=2)
parser.add_argument('-sgl','--sents_gcn_layer', help='Encoder GCN Layer Number of sents', required=False, type=int,
                    default=2)
parser.add_argument('-dl','--decoder_layer', help='Decoder Layer Number', required=False, type=int, default=2)
parser.add_argument('-vs','--vocab_size', help='maximum vocaulary size', required=False, type=int, default=35000)
parser.add_argument('-pvs','--pos_vocab_size', help='maximum vocaulary size of POS', required=False, type=int,
                    default=150)
parser.add_argument('-ems','--embed_size', help='embedding size', required=False, type=int, default=256)
parser.add_argument('-pos_ems','--pos_embedding_size', help='pos embedding size', required=False, type=int, default=256)
parser.add_argument('-pt','--pre_train', help='use pre_train word embedding', required=False, type=bool, default=False)
parser.add_argument('-wep','--word_embedding_path', help='pre_train embedding path', 
                    required=False,default='./vocab_embedding.pkl')
parser.add_argument('-emno','--embed_notrain', help='whether fine tune word embedding', required=False, type=bool,
                    default=False)
parser.add_argument('-mp','--model_path', help='path of the file to load', type=str, required=False)
parser.add_argument('-clip','--clip', help='gradient clipping', required=False, type=float, default=10.0)
parser.add_argument('-tfr','--teacher_forcing_ratio', help='teacher_forcing_ratio', type=float, required=False, default=0.5)
parser.add_argument('-thres','--thres', help='classify threshold', type=float, required=False, default=0.5)
parser.add_argument('-thres_adj','--threshold_adj', help='sents adj threshold', type=float, required=False, default=0.2)
parser.add_argument('-dp','--data_path', help='path of the data to load', type=str,
                    required=False, default='./data/AESLC/enron_subject_line/expr_data_nopunc/')
parser.add_argument('-sample','--sample', help='Number of Samples', required=False,default=None)
parser.add_argument('-evalp','--evalp', help='evaluation period', required=False, type=int, default=1)
parser.add_argument('-an','--addName', help='An add name for the save folder', required=False, type=str, default='')
parser.add_argument('-gs','--genSample', help='Generate Sample', required=False, type=int, default=0)
parser.add_argument('-es','--earlyStop', help='Early Stop Criteria, METEOR or ROUGE', required=False, type=str,
                    default='METEOR')
parser.add_argument('-mode','--mode', help='special training mode', required=False, default='softmax')
parser.add_argument('-rec','--record', help='use record function during inference', type=int, required=False, default=0)
parser.add_argument('-bese','--beam_search', help='use beam_search during inference, default is greedy search', 
                    required=False, type=bool, default=False)
parser.add_argument('-copy','--copy', help='copy mechanism', required=False, type=bool, default=False)
parser.add_argument('-hard','--hard', help='hard weight', required=False, type=bool, default=False)
parser.add_argument('-user','--user', help='whether use sender or receiver info', required=False, type=bool,
                    default=False)
parser.add_argument('-besize','--beam_size', help='beam size when use beam search', type=int, required=False, default=4)
parser.add_argument('-tk','--topk', help='return topk results when use beam search', type=int, required=False, default=1)

args = vars(parser.parse_args())
print(str(args))
print("USE_CUDA: "+str(args['use_cuda']))


def _cuda(x):
    if args['use_cuda']:
        return x.cuda()
    else:
        return x
