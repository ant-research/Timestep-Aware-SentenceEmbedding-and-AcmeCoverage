import os,re
import numpy as np
from rouge import Rouge
from pyrouge import Rouge155
import logging

rouge = Rouge()


def cal_rouge(hyp, ref, ref_num=1):
    with open(ref) as fr:
        ref = fr.readlines()
    with open(hyp) as fr:
        hyp = fr.readlines()

    if ref_num > 1:
        refs = [[] for i in range(ref_num)]
        for i in range(0, len(ref), ref_num):
            for j in range(ref_num):
                refs[j].append(ref[i+j])

    else:
        refs = [ref]

    R_1_f = []
    R_2_f = []
    R_L_f = []
    
    for i in refs:
        scores = rouge.get_scores(hyp, i, avg=True)
        R_1_f.append(scores['rouge-1']['f'])
        R_2_f.append(scores['rouge-2']['f'])
        R_L_f.append(scores['rouge-l']['f']) 

    print('ROUGE with rouge  R-1:', np.mean(R_1_f), 'R-2:', np.mean(R_2_f), 'R-L', np.mean(R_L_f))
    return np.mean(R_1_f), np.mean(R_2_f), np.mean(R_L_f)


def cal_rouge_v2(hyp, ref, ref_num=1):
    r = Rouge155()
    logging.getLogger('global').setLevel(logging.WARNING)
    r.system_dir = hyp
    r.model_dir = ref
    r.system_filename_pattern = 'pred_(\d+)'
    if ref_num == 1:
        r.model_filename_pattern = 'truth_#ID#'
    else:
        r.model_filename_pattern = 'truth_[A-Z]_#ID#'

    output = r.convert_and_evaluate()
    output_dict = r.output_to_dict(output)
    r1 = output_dict['rouge_1_f_score']
    r2 = output_dict['rouge_2_f_score']
    rl = output_dict['rouge_l_f_score']
    print('ROUGE with pyrouge  R-1:', r1, 'R-2:', r2, 'R-L', rl)
    return r1, r2, rl



