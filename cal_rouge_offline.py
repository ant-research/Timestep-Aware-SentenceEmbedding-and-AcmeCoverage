import os
import sys
from rouge_measure import cal_rouge_v2


def split_file(source_path, file_name, s_num=1):
    data_type, sents_type = file_name.split('_')[:2]
    if sents_type == 'system':
        sents_type = 'pred'
    if sents_type in ['groundtruth', 'ground1']:
        sents_type = 'truth'

    save_path = source_path + '/' + data_type + '/' + sents_type + '/'
    print('save_path', save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        os.system('rm -rf ' + save_path + '*')

    if s_num == 1:
        os.system('split -l 1 ' + source_path + '/' + file_name + ' -d -a 4 ' + save_path + sents_type + '_')
        #os.system('mv ./' + sents_type + '_* ' + save_path)
    else:
        os.system('split -l ' + str(s_num) + ' ' + source_path + '/' + file_name + ' -d -a 4 ' +
                  save_path + sents_type + '_')
        #os.system('mv ./' + sents_type + '_* ' + save_path)

    return save_path


def cal_rouge(source_path, file_name1, file_name2, sents_num=1):
    save_path1 = split_file(source_path, file_name1)
    save_path2 = split_file(source_path, file_name2, sents_num)
    if sents_num == 1:
        r1, r2, rl = cal_rouge_v2(save_path1, save_path2)
    else:
        a = ['A', 'B', 'C']
        for f in os.listdir(save_path2):
            f_name = f.split('_')[1]
            all_ann = open(save_path2 + f).readlines()
            for i in range(sents_num-2, sents_num):
                with open(save_path2 + f.split('_')[0] + '_' + a[i] + '_' + f_name, 'w') as fw:
                    fw.write(all_ann[i])
        os.system('rm -rf ' + save_path2 + 'ann_0*')
        os.system('rm -rf ' + save_path2 + 'ann_1*')
        r1, r2, rl = cal_rouge_v2(save_path1, save_path2, 2)
    return r1, r2, rl


if __name__ == '__main__':
    source_path, file_name1, file_name2, sents_num = sys.argv[1:]
    sents_num = eval(sents_num)
    cal_rouge(source_path, file_name1, file_name2, sents_num)
