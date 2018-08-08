# from pymysqlpool import ConnectionPool
# import configparser
# import pandas as pd
# import codecs
# import os
#
# print(os.path.realpath(__file__))
# path = os.path.dirname(os.path.realpath(__file__))
# print(path)
# config = configparser.ConfigParser()
# config.read("/root/PycharmProjects/text_similarity/conf/mysql.conf")
# print(config["db"]['pool_name'])
# print(dict(config.items("db")))
# def connection_pool():
#     pool = ConnectionPool(**dict(config.items("db")))
#     return pool
# # pool = connection_pool()
# # if pool:
# #     print("success")
# # with connection_pool().connection() as conn:
# #     df = pd.read_sql('select * from similarity', conn)
# #     print(df[:10])
# # #一致跳闸断路器三相不同步,仙永线5012开关仙永线5012开关测控仙永线5012开关三相不一致动作
# with connection_pool().cursor() as cursor:
#     datas = []
#     with codecs.open('/root/PycharmProjects/text_similarity/data/test_data.txt') as f:
#         lines = f.readlines()
#         for line in lines:
#             words = line.split(',')
#             datas.append((words[0][:160], words[1][:160], int(words[3])))
#     result = cursor.executemany('insert into train_table (line_1, line_2, label) VALUES (%s, %s, %s)', datas)
#     cursor.execute('SELECT * FROM train_table')
#
#     # result = cursor.execute('INSERT INTO similarity (line_1, line_2) VALUES (%s, %s)', ("一致跳闸断路器三相不同步", "仙永线5012开关仙永线5012开关测控仙永线5012开关三相不一致动作"))
#     # result = cursor.execute('select * from similarity where id between 1 and 5')
#     # data = [("一致跳闸断路器三相不同步", "仙永线5012开关仙永线5012开关测控仙永线5012开关三相不一致动作")]
#     # datas = data * 100
#     # print(datas)
#     # result = cursor.executemany('insert into similarity (line_1, line_2) VALUES (%s, %s)', datas)
#     # # print(result)
#     # # for a in cursor:
#     # #     print(a)
#     # cursor.execute('UPDATE similarity SET line_1="断路器三相不同步" WHERE id = 1')
#     # cursor.execute('SELECT * FROM similarity WHERE id = 1')
#     # # for a in cursor:
#     # #     print(a)
#     # print(cursor.fetchone())
#     # cursor.execute('DELETE FROM similarity WHERE id between 10 and 10000')