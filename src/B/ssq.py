# tar =[i for i in range(1,17)]
# squen = [i for i in range(1,34)]
#
# import re
# from pprint import pprint
#
# data =open('ssq').readlines()
# X0_list = []
#
# for i in data:
#     i = i.split('\t')
#     X0 = i[0].split(' ')
#     X1 = i[1].strip()
#     X0.append(X1)
#     X0_list.append(X0)
#
#
# pprint(X0_list)

import tensorflow as tf
print(tf.__version__)
