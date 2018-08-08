from flask import Flask, request, jsonify
from xbg_classifier import update_model, load_model, predict, build_model
from utils.data_helper import  get_matrix_by_line, get_vector_by_combine_matrix_1
from sklearn.externals import joblib
import codecs
import json
import logging
from pymysqlpool import ConnectionPool
import configparser
import os
import traceback
from utils.config import params
import numpy as np
from optparse import OptionParser
import argparse


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename='log.txt')
logger = logging.getLogger(__name__)
path = os.path.dirname(os.path.realpath(__file__))
#print(path)
app = Flask(__name__)

# model_directory = 'model'
# model_file_name = '%s/model.bin' % model_directory
# vocab_bag_list_name = '%s/vocab_bag_list.bin' % model_directory

vocab_bag_list = {}
model = None
pool = None
#curl -H "Content-Type:application/json" -X POST --data '[{"index":["*"],"preference":"1503652289983","ignore_unavaile"},{"sort":[{"timestamp":{"order":"desc"}}],"query":{"must_not":[],"bool":{"must":[{"query_string":{"query":"cluster"}},{"range":{"timestamp":{"gte":"1503667558137","lte":"1503667558137"}}}]}},"from":"0","size":"500","version":"true"}]' http://192.168.220.186:9000/predict
#curl -H "Content-Type:application/json" -X POST  --data '{"index":["*"],"preference":"1503652289983","ignore_unavaile"}' http://192.168.220.186:9000/predict


def connection_pool():
    pool = ConnectionPool(**dict(config.items("db")))
    return pool


#curl -H "Content-Type:application/json" -X POST --data '{"start_id":1,"end_id":101, "table_name":"similarity"}' http://192.168.220.187:9000/predict
#curl -H "Content-Type:application/json" -X POST --data '{"line_1":"一致跳闸断路器三相不同步","line_2":"仙永线5012开2开关测控仙永线5012开关三相不一致动作"}' http://192.168.220.187:9000/predict
@app.route('/predict', methods=['post'])
def model_predict():
    global model
    try:
        try:
            json_data = request.get_data()
            json_data = json.loads(json_data)
            start_id = int(json_data['start_id'])
            end_id = int(json_data['end_id'])
            table_name = json_data['table_name']
            # print(json_data)
        except Exception as e:
            logger.warning("json data not correct!!!")
            return jsonify({'state': "json data failed", 'trace': traceback.format_exc()})
        if not pool:
            return jsonify({'state': 'db failed'})
        with pool.cursor() as cursor:
            cursor.execute('SELECT * FROM %s WHERE id between %d and %d' % (table_name, start_id, end_id))
            pred_res = []
            for res in cursor:
                line_1 = res['MERGED_MESSAGE_AND_SIGNAL']
                line_2 = res['BACK_END_MESSAGE']
                try:
                    mat_1 = get_matrix_by_line(line_1, vocab_bag_list)
                    mat_2 = get_matrix_by_line(line_2, vocab_bag_list)
                    vec = get_vector_by_combine_matrix_1(mat_1, mat_2)
                    vec = vec.reshape(1, -1)
                except Exception as e:
                    logger.warning("data convert failed!!!")
                    continue
                try:
                    pred = int(predict(model, vec)[0])
                except Exception as e:
                    logger.warning("predict failed")
                    continue
                pred_res.append((line_1, line_2, pred))
            cursor.execute('SELECT max(id) as start FROM similarity_res')
            for res in cursor:
                if not res['start']:
                    start_id = 1
                else:
                    start_id = res['start'] + 1
            cursor.executemany('insert into similarity_res (MERGED_MESSAGE_AND_SIGNAL,'
                               ' BACK_END_MESSAGE, label) VALUES (%s, %s, %s)', pred_res)
            end_id = start_id + len(pred_res) - 1
    except Exception as e:
        return jsonify({'state': "json data failed", 'trace': traceback.format_exc()})

    return jsonify({'state': 'predict success', 'start_id': start_id, 'end_id': end_id})


#curl -H "Content-Type:application/json" -X POST --data '{"model_name":"model.bin"}' http://192.168.220.187:9000/load
@app.route('/load', methods=['POST'])
def model_load():
    global model
    try:
        try:
            json_data = request.get_data()
            json_data = json.loads(json_data)
            #print(json_data)
        except Exception as e:
            logger.warning("json data not correct!!!")
            return jsonify({'state':"json data failed", 'trace':traceback.format_exc()})
        model_name = json_data['model_name']
        model_1 = load_model(os.path.join(os.path.join(path, 'model'), model_name))
        if model_1 == model:
            return jsonify({'state':'model load failed or model file not update'})
        model = model_1
    except Exception as e:
        return jsonify({'state':"json data failed", 'trace':traceback.format_exc()})
    return jsonify({'state':'model load success'})

#train_table:id, line_1, line_2, label
#curl -H "Content-Type:application/json" -X POST --data '{"start_id":1, "end_id":1000, "table_name": "train_table"}' http://192.168.220.187:9000/train
@app.route('/train', methods=['POST'])
def model_train():
    global model
    model = None
    try:
        try:
            json_data = request.get_data()
            json_data = json.loads(json_data)
            start_id = int(json_data['start_id'])
            end_id = int(json_data['end_id'])
            table_name = json_data['table_name']
            print(json_data)
        except Exception as e:
            logger.warning("json data not correct!!!")
            return jsonify({'state':"json data failed", 'trace':traceback.format_exc()})
        trains = []
        labels = []
        with pool.cursor() as cursor:
            cursor.execute('SELECT * FROM %s WHERE id between %d and %d' % (table_name, start_id, end_id))
            pred_res = []
            for res in cursor:
                line_1 = res['MERGED_MESSAGE_AND_SIGNAL']
                line_2 = res['BACK_END_MESSAGE']
                label = int(res['label'])
                try:
                    mat_1 = get_matrix_by_line(line_1, vocab_bag_list)
                    mat_2 = get_matrix_by_line(line_2, vocab_bag_list)
                    vec = get_vector_by_combine_matrix_1(mat_1, mat_2)
                    trains.append(vec)
                    labels.append(label)
                except Exception as e:
                    logger.warning("data convert failed!!!")
                    continue
        if len(labels) == 0:
            return jsonify({"state":"db failed"})
        model = build_model(np.array(trains), np.array(labels), params, 500)
        if not model:
            return jsonify({"state": "model train failed"})
    except Exception as e:
        return jsonify({'state':"json data failed", 'trace':traceback.format_exc()})
    return jsonify({'state': 'model train success'})


#update_table:id, line_1, line_2, label
#curl -H "Content-Type:application/json" -X POST --data '{"start_id":1, "end_id":1000, "table_name": "update_table"}' http://192.168.220.187:9000/update
@app.route('/update', methods=['POST'])
def model_update():
    global model
    try:
        try:
            json_data = request.get_data()
            json_data = json.loads(json_data)
            start_id = int(json_data['start_id'])
            end_id = int(json_data['end_id'])
            table_name = json_data['table_name']
            print(json_data)
        except Exception as e:
            logger.warning("json data not correct!!!")
            return jsonify({'state':"json data failed", 'trace': traceback.format_exc()})
        trains = []
        labels = []
        with pool.cursor() as cursor:
            cursor.execute('SELECT * FROM %s WHERE id between %d and %d' % (table_name, start_id, end_id))
            pred_res = []
            for res in cursor:
                line_1 = res['MERGED_MESSAGE_AND_SIGNAL']
                line_2 = res['BACK_END_MESSAGE']
                label = int(res['label'])
                try:
                    mat_1 = get_matrix_by_line(line_1, vocab_bag_list)
                    mat_2 = get_matrix_by_line(line_2, vocab_bag_list)
                    vec = get_vector_by_combine_matrix_1(mat_1, mat_2)
                    trains.append(vec)
                    labels.append(label)
                except Exception as e:
                    logger.warning("data convert failed!!!")
                    continue
        if len(labels) == 0:
            return jsonify({"state":"db failed"})

        model_1 = update_model(model, np.array(trains), np.array(labels), params, 500)
        if model_1 == model:
            return jsonify({"state":"model update failed"})
    except Exception as e:
        return jsonify({'state':"json data failed", 'trace': traceback.format_exc()})
    model = model_1
    return jsonify({"state":"model update success"})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--host", default='192.168.220.205')
    parser.add_argument("-p", "--post", default=9000)
    options = vars(parser.parse_args())
    retry_num = 100

    vocab_bag_list_dir = os.path.join(os.path.join(path, 'model'), 'vocab_bag_list.bin')
    model_dir = os.path.join(os.path.join(path, 'model'), 'model.bin')
    dbconf_dir = os.path.join(os.path.join(path, 'conf'), 'mysql.conf')
    while retry_num > 0 and not vocab_bag_list:
        vocab_bag_list = joblib.load(vocab_bag_list_dir)
        retry_num -= 1

    if not vocab_bag_list:
        logger.error("retry 100 numbers, vocab_bag_list load failed!!!")
    retry_num = 100

    while retry_num > 0 and not model:
        model = load_model(model_dir)
        retry_num -= 1

    if not model:
        logger.error("retry 100 numbers, model load failed!!!")
    config = configparser.ConfigParser()
    config.read(dbconf_dir)
    retry_num = 10
    while retry_num > 0 and not pool:
        pool = connection_pool()
        retry_num -= 1
    if not pool:
        logger.error("retry 10 numbers, db connect failed!!!")
    app.run(host=options['host'], port=int(options['post']), debug=False)




