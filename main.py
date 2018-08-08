from xbg_classifier import update_model, load_model, predict, eval
from utils.data_helper import  get_matrix_by_line, get_vector_by_combine_matrix_1
from sklearn.externals import joblib
import codecs
vocab_bag_list = {}
model = None


if __name__ == '__main__':
    vocab_bag_list = joblib.load('model/vocab_bag_list.bin')
    model = load_model('model/model.bin')
    if not model:
        print("model file not exists!!!")
    else:
        print("load model success!!!")
    with codecs.open('data/test_data.txt') as f:
        lines = f.readlines()
        total = len(lines)
        count = 0
        for line in lines:
            words = line.split(',')
            mat_1 = get_matrix_by_line(words[0], vocab_bag_list)
            mat_2 = get_matrix_by_line(words[1], vocab_bag_list)
            vec = get_vector_by_combine_matrix_1(mat_1, mat_2)
            vec = vec.reshape(1, -1)
            true = words[-2]
            pred = predict(model, vec)[0]
            if int(true) != int(pred):
                count += 1
                print(true + '===>' + str(int(pred)))
        print("total...", total)
        print("count...", count)
    print("percent...", 1 - (float(count) / total))
    #print(lines[:1][0].split(','))
