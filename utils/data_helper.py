import codecs
import os
import csv
import numpy as np
import jieba
import re


def get_data(file_name, threshold1=0.6, threshold2=None):
    with codecs.open(file_name, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        score = set()
        result = []
        for line in lines:
            parts = line.split('\t')
            if parts[-1].startswith('匹配度') or parts[-1].startswith('完全匹配') or parts[-1].startswith('推断基本匹配')\
                    or parts[-1].startswith('不匹配') or parts[-1].startswith('基本匹配') or parts[-1].startswith('近似基本匹配')\
                    or parts[-1].startswith('推断近似基本匹配') or parts[-1].startswith('推断完全匹配'):
                res = []
                match_1 = parts[0] + parts[1]
                match_2 = re.sub('\.', '', parts[2])
                if parts[-1].startswith('完全匹配') or parts[-1].startswith('推断基本匹配') or parts[-1].startswith('基本匹配') \
                        or parts[-1].startswith('近似基本匹配') or parts[-1].startswith('推断近似基本匹配') or parts[-1].startswith('推断完全匹配'):
                    if threshold2 == None:
                        label = 1
                    else:
                        label = 2
                elif parts[-1].startswith('不匹配'):
                    label = 0
                else:
                    index = parts[-1].index('%')
                    if threshold2 == None:
                        if float('0.' + parts[-1][4:index]) > threshold1:
                            label = 1
                        else:
                            label = 0
                    else:
                        if float('0.' + parts[-1][4:index]) > threshold1:
                            label = 2
                        elif float('0.' + parts[-1][4:index]) > threshold2:
                            label = 1
                        else:
                            label = 0
                res.append(match_1)
                res.append(match_2)
                res.append(re.sub('\r\n', '',parts[-1]))
                res.append(label)
                result.append(res)
            else:
                continue
    if threshold2 == None:
        csvfile = open('res_label_v1.csv', 'w', newline='\n')  # 设置newline，否则两行之间会空一行
    else:
        csvfile = open('res_label_v2.csv', 'w', newline='\n')
    writer = csv.writer(csvfile)
    writer.writerow(['combine_column', 'background description', 'match_score', 'label'])
    writer.writerows(result)
    csvfile.close()


def load_data(file_name):
    with codecs.open(file_name, 'r', encoding='utf-8') as file:
        lines = file.readlines()[1:]
        corpus = []
        labels = []
        corpus_1 = []
        corpus_2 = []
        for line in lines:
            match_parts = line.split(',')

            #new_line =  + ' '.join(jieba.cut(match_parts[1]))
            corpus.append(' '.join(jieba.cut(match_parts[0])))
            corpus.append(' '.join(jieba.cut(match_parts[1])))
            corpus_1.append(' '.join(jieba.cut(match_parts[0])))
            corpus_2.append(' '.join(jieba.cut(match_parts[1])))
            labels.append(int(match_parts[-1]))
        return corpus, corpus_1, corpus_2, labels


def iter_minibatches(vec_corpus, labels, minibatch_size=100):
    '''
    迭代器
    给定文件流（比如一个大文件），每次输出minibatch_size行，默认选择1k行
    将输出转化成numpy输出，返回X, y
    '''
    corpus_batch = []
    labels_batch = []
    x = []
    y = []
    cur_line_num = 0

    for vec, label in zip(vec_corpus, labels):
        y.append(float(label))
        x.append(vec)  # 这里要将数据转化成float类型

        cur_line_num += 1
        if cur_line_num >= minibatch_size:
            corpus_batch.append(np.array(x))
            labels_batch.append(np.array(y))
            x, y = [], []
            cur_line_num = 0
    return np.array(corpus_batch), np.array(y)


def get_wordvector_dict(file_name, vocab_bag_list):
    f = codecs.open(file_name, 'r', encoding='utf-8')
    txt = f.readline()
    vocab_size_layer = txt.split()
    f.readline()
    txt = f.readline()
    num = 0
    while txt:
        if txt != '\r\n':
            num += 1
            vocab_line = txt.split()
            vocab = vocab_line[0]
            vocab_vec = vocab_line[1:]
            vocab_bag_list[vocab] = vocab_vec
        txt = f.readline()


def get_line_vector(line, vocab_bag_list):
    vector = []
    words = line.split(' ')
    for key in vocab_bag_list.keys():
        length = len(vocab_bag_list[key])
        word_vector = [0.0 for i in range(length)]
        vector.append(word_vector)
        break
    for word in words:
        if word in vocab_bag_list.keys():
            a = vocab_bag_list[word]
            word_vector = [np.float64(a[i]) for i in range(len(a))]
            vector.append(word_vector)
    mat = np.array(vector)
    res = np.mean(mat, axis=0)
    return res


def get_matrix(file_name, vocab_bag_list, output_filename):
    with codecs.open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        vectors = []
        for line in lines:
            if line != '\n':
                a = get_line_vector(line, vocab_bag_list)
                vectors.append(a)
    write_file = codecs.open(output_filename, 'w', encoding='utf-8')
    for vector in vectors:
        b = [str(vector[i]) for i in range(len(vector))]
        st = ' '.join(b)
        write_file.write(st + '\n')
    return np.array(st + '\n')


def get_train_data(file_name, output_file):
    with codecs.open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]
        train_data = []
        for line in lines:
            parts = line.split(',')
            train_data.append(' '.join(jieba.cut(parts[0])))
            train_data.append(' '.join(jieba.cut(parts[1])))
    write_file = codecs.open(output_file, 'w', encoding='utf-8')
    for line in train_data:
        write_file.write(line + '\n')


def get_matrix_by_line(line, vocab_bag_list):
    vector = []
    words = ' '.join(jieba.cut(line)).split(' ')
    for key in vocab_bag_list.keys():
        length = len(vocab_bag_list[key])
        word_vector = [0.0 for i in range(length)]
        vector.append(word_vector)
        break
    for word in words:
        if word in vocab_bag_list.keys():
            a = vocab_bag_list[word]
            word_vector = [np.float64(a[i]) for i in range(len(a))]
            vector.append(word_vector)
    mat = np.array(vector)
    return mat


def cosine_similarity(a, b):
    u_v = np.dot(a, b)
    u_v_sqrt = np.sqrt(np.sum(np.square(a))) * np.sqrt(np.sum(np.square(b)))
    score = u_v / u_v_sqrt if u_v_sqrt != 0 else 0
    return score


def get_vector_by_combine_matrix_1(mat_1, mat_2):
    ratio = []
    for i in range(len(mat_1)):
        tmp = []
        for j in range(len(mat_2)):

            score = cosine_similarity(mat_1[i], mat_2[j])
            tmp.append(score)
        ratio.append(max(tmp))
    new_mat = []
    for i in range(len(mat_1)):
        new_mat.append(ratio[i] * mat_1[i])
    mat = np.array(new_mat)
    res = np.mean(mat, axis=0)
    return res


def get_vector_by_combine_matrix_2(mat_1, mat_2):
    res_1 = get_vector_by_combine_matrix_1(mat_1, mat_2)
    res_2 = get_vector_by_combine_matrix_1(mat_2, mat_1)
    tmp = np.concatenate(res_1, res_2, axis=0)
    res = np.mean(tmp, axis=0)
    return res


def get_matrix_1(file_name, vocab_bag_list, flag=True):
    mat = []
    labels = []
    with codecs.open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]
        for line in lines:
            parts = line.split(',')
            if flag:
                mat_1 = get_matrix_by_line(parts[0], vocab_bag_list)
                mat_2 = get_matrix_by_line(parts[1], vocab_bag_list)
            else:
                mat_2 = get_matrix_by_line(parts[0], vocab_bag_list)
                mat_1 = get_matrix_by_line(parts[1], vocab_bag_list)
            vec = get_vector_by_combine_matrix_1(mat_1, mat_2)
            mat.append(vec)
            labels.append(int(parts[-1]))
    return np.array(mat), np.array(labels)


def get_matrix_2(file_name, vocab_bag_list):
    mat = []
    labels = []
    with codecs.open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]
        for line in lines:
            parts = line.split(',')
            mat_1 = get_matrix_by_line(parts[0], vocab_bag_list)
            mat_2 = get_matrix_by_line(parts[1], vocab_bag_list)
            vec = get_vector_by_combine_matrix_2(mat_1, mat_2)
            mat.append(vec)
            labels.append(int(parts[-1]))
    return np.array(mat), np.array(labels)


def convert_line_to_int(line, vocab_to_int_dict, max_line_length=20):
    words = " ".join(jieba.cut(line)).split(" ")
    res = []
    for word in words:
        if word not in vocab_to_int_dict:
            res.append(0)
        else:
            res.append(vocab_to_int_dict[word])
    vocab_size = len(vocab_to_int_dict)
    if len(res) > max_line_length:
        res = res[:max_line_length]
    else:
        res.extend([0] * (max_line_length - len(res)))
    return res


def convert_lines_to_int(lines, vocab_to_int_dict):
    result = []
    for line in lines:
        res = convert_line_to_int(line, vocab_to_int_dict)
        result.append(res)
    return np.array(result)


def get_vocab_to_int_dict(corpus):
    vocab_to_int_dict = {}
    int_to_vocab_dict = {}
    vocab_set = set()
    for line in corpus:
        words = line.split(" ")
        [vocab_set.add(word) for word in words]

    for i, word in enumerate(list(vocab_set)):
        vocab_to_int_dict[word] = i + 1
        int_to_vocab_dict[i + 1] = word
    return vocab_to_int_dict, int_to_vocab_dict


if __name__ == '__main__':
    get_data('../data.txt', 0.75, 0.5)
    # #get_train_data(os.path.join('.', 'res_label.csv'), os.path.join('.', 'train.txt'))
    # vocab_bag_list = {}
    # get_wordvector_dict(os.path.join('.', 'vectors.txt'), vocab_bag_list)
    # mat, labels = get_matrix_1(os.path.join('.', 'res_label.csv'), vocab_bag_list)
    # print(mat.shape)
    # print(labels.shape)
    # # with codecs.open(os.path.join('.', 'res_label.csv'), 'r', encoding='utf-8') as f:
    # #     lines = f.readlines()[1:]
    # #     vectors = []
    # #     for line in lines[:5]:
    # #         parts = line.split(',')
    # #         mat_1 = get_matrix_by_line(parts[0], vocab_bag_list)
    # #         mat_2 = get_matrix_by_line(parts[1], vocab_bag_list)
    # #         #print(cosine_similarity(mat_1[0].reshape(-1, 1), mat_2[0].reshape(-1, 1))[0])
    # #         vec = get_vector_by_combine_matrix_1(mat_1, mat_2)
    # #         print(vec)
    #
    # #get_matrix(file_name, vocab_bag_list, output_filename)
    # #print(vocab_bag_list)
    with codecs.open('../data.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()
        score = set()
        result = []
        for line in lines:
            parts = line.split('\t')
            if parts[-1].startswith('匹配度') or parts[-1].startswith('完全匹配') or parts[-1].startswith('推断基本匹配')\
                    or parts[-1].startswith('不匹配') or parts[-1].startswith('基本匹配') or parts[-1].startswith('近似基本匹配')\
                    or parts[-1].startswith('推断近似基本匹配') or parts[-1].startswith('推断完全匹配'):
                res = []
                match_1 = parts[0] + parts[1]
                match_2 = re.sub('\.', '', parts[2])
                if parts[-1].startswith('完全匹配') or parts[-1].startswith('推断基本匹配') or parts[-1].startswith('基本匹配') \
                        or parts[-1].startswith('近似基本匹配') or parts[-1].startswith('推断近似基本匹配') or parts[-1].startswith('推断完全匹配'):
                    label = 2
                elif parts[-1].startswith('不匹配'):
                    label = 0
                else:
                    index = parts[-1].index('%')
                    if float('0.' + parts[-1][4:index]) > 0.7:
                        label = 2
                    elif float('0.' + parts[-1][4:index]) > 0.5:
                        label =1
                    else:
                        label = 0
                res.append(match_1)
                res.append(match_2)
                res.append(re.sub('\r\n', '',parts[-1]))
                res.append(label)
                result.append(res)
            else:
                continue