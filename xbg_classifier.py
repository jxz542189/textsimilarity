import xgboost as xgb
from utils.data_helper import  get_wordvector_dict, get_matrix_1, get_matrix_2, get_matrix_by_line
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from utils.config import params
import os


def build_model(X_train, y_train, params, num_rounds):
    plst = params.items()
    dtrain = xgb.DMatrix(X_train, y_train)
    model = xgb.train(plst, dtrain, num_rounds, xgb_model=None)
    return model


def save_model(model, model_name='bst.model'):
    model.save_model(model_name)


def load_model(model_file='bst.model'):
    model = xgb.Booster(model_file=model_file)
    return model


def update_model(model, x_train, y_train, params, num_rounds):
    plst = params.items()
    dtrain = xgb.DMatrix(x_train, y_train)
    model_1 = xgb.train(plst, dtrain, num_rounds, xgb_model=model)
    return model_1


def predict(model, data):
    data = xgb.DMatrix(data)
    res = model.predict(data)
    return res


def eval(model, data, labels):
    pre_labels = predict(model, data)
    print(classification_report(labels, pre_labels))

vocab_bag_list = {}
get_wordvector_dict(os.path.join('/root/PycharmProjects/similarity/utils', 'vectors.txt'), vocab_bag_list)
vec_corpus, labels = get_matrix_1('/root/PycharmProjects/similarity/utils/res_label_v1.csv', vocab_bag_list)
X_train, X_test, y_train, y_test = train_test_split(vec_corpus, labels, random_state=1234565)
model_1 = build_model(X_train[:len(X_train) // 2], y_train[:len(y_train) // 2], params, 500)
eval(model_1, vec_corpus, labels)
model_2 = update_model(model_1, X_train[len(X_train) // 2 :], y_train[len(y_train) // 2 : ], params, 500)
eval(model_2, vec_corpus, labels)
model_3 = update_model(model_2, X_test, y_test, params, 500)
eval(model_3, X_test, y_test)
