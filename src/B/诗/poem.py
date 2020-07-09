import math
import re
import numpy as np
import tensorflow as tf
from collections import Counter

import warnings
warnings.filterwarnings("ignore")

# 数据路径
DATA_PATH = 'poetry.txt'
# 单行诗最大长度
MAX_LEN = 64
# 禁用的字符，拥有以下符号的诗将被忽略
DISALLOWED_WORDS = ['（', '）', '(', ')', '__', '《', '》', '【', '】', '[', ']']

# 一首诗（一行）对应一个列表的元素
poetry = []

# 按行读取数据 poetry.txt
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    lines = f.readlines()
# 遍历处理每一条数据
for line in lines:
    # 利用正则表达式拆分标题和内容
    fields = re.split(r"[:：]", line)
    # 跳过异常数据
    if len(fields) != 2:
        continue
    # 得到诗词内容（后面不需要标题）
    content = fields[1]
    # 跳过内容过长的诗词
    if len(content) > MAX_LEN - 2:
        continue
    # 跳过存在禁用符的诗词
    if any(word in content for word in DISALLOWED_WORDS):
        continue

    poetry.append(content.replace('\n', ''))  # 最后要记得删除换行符


'''
过滤低频  字，这里没用词
'''
# 最小词频
MIN_WORD_FREQUENCY = 8

# 统计词频，利用Counter可以直接按单个字符进行统计词频
counter = Counter()
for line in poetry:
    counter.update(line)
# 过滤掉低词频的词
tokens = [token for token, count in counter.items() if count >= MIN_WORD_FREQUENCY]


print(tokens[1:3])

'''

'''
# 补上特殊词标记：填充字符标记、未知词标记、开始标记、结束标记
tokens = ["[PAD]", "[NONE]", "[START]", "[END]"] + tokens
# 映射: 词 -> 编号
word_idx = {}
# 映射: 编号 -> 词
idx_word = {}
for idx, word in enumerate(tokens):
    word_idx[word] = idx
    idx_word[idx] = word


class Tokenizer:
    """
    分词器
    """
    def __init__(self, tokens):
        # 词汇表大小
        self.dict_size = len(tokens)
        # 生成映射关系
        self.token_id = {}  # 映射: 词 -> 编号
        self.id_token = {}  # 映射: 编号 -> 词
        for idx, word in enumerate(tokens):
            self.token_id[word] = idx
            self.id_token[idx] = word

        # 各个特殊标记的编号id，方便其他地方使用
        self.start_id = self.token_id["[START]"]
        self.end_id = self.token_id["[END]"]
        self.none_id = self.token_id["[NONE]"]
        self.pad_id = self.token_id["[PAD]"]

    def id_to_token(self, token_id):
        """
        编号 -> 词
        """
        return self.id_token.get(token_id)

    def token_to_id(self, token):
        """
        词 -> 编号
        """
        return self.token_id.get(token, self.none_id)

    def encode(self, tokens):
        """
        词列表 -> [START]编号 + 编号列表 + [END]编号
        """
        token_ids = [self.start_id, ]  # 起始标记
        # 遍历，词转编号
        for token in tokens:
            token_ids.append(self.token_to_id(token))
        token_ids.append(self.end_id)  # 结束标记
        return token_ids

    def decode(self, token_ids):
        """
        编号列表 -> 词列表(去掉起始、结束标记)
        """
        # 起始、结束标记
        flag_tokens = {"[START]", "[END]"}

        tokens = []
        for idx in token_ids:
            token = self.id_to_token(idx)
            # 跳过起始、结束标记
            if token not in flag_tokens:
                tokens.append(token)
        return tokens


tokenizer = Tokenizer(tokens)


class PoetryDataSet:
    """
    古诗数据集生成器
    """

    def __init__(self, data, tokenizer, batch_size):
        # 数据集
        self.data = data
        self.total_size = len(self.data)
        # 分词器，用于词转编号
        self.tokenizer = tokenizer
        # 每批数据量
        self.batch_size = batch_size
        # 每个epoch迭代的步数
        self.steps = int(math.floor(len(self.data) / self.batch_size))

    def pad_line(self, line, length, padding=None):
        """
        对齐单行数据
        """
        if padding is None:
            padding = self.tokenizer.pad_id

        padding_length = length - len(line)
        if padding_length > 0:
            return line + [padding] * padding_length
        else:
            return line[:length]

    def __len__(self):
        return self.steps

    def __iter__(self):
        # 打乱数据
        np.random.shuffle(self.data)
        # 迭代一个epoch，每次yield一个batch
        for start in range(0, self.total_size, self.batch_size):
            end = min(start + self.batch_size, self.total_size)
            data = self.data[start:end]

            max_length = max(map(len, data))

            batch_data = []
            for str_line in data:
                # 对每一行诗词进行编码、并补齐padding
                encode_line = self.tokenizer.encode(str_line)
                pad_encode_line = self.pad_line(encode_line, max_length + 2)  # 加2是因为tokenizer.encode会添加START和END
                batch_data.append(pad_encode_line)

            batch_data = np.array(batch_data)
            # yield 特征、标签
            yield batch_data[:, :-1], batch_data[:, 1:]

    def generator(self):
        while True:
            yield from self.__iter__()


BATCH_SIZE = 32
dataset = PoetryDataSet(poetry, tokenizer, BATCH_SIZE)




model = tf.keras.Sequential([
    # 词嵌入层
    tf.keras.layers.Embedding(input_dim=tokenizer.dict_size, output_dim=150),
    # 第一个LSTM层
    tf.keras.layers.LSTM(150, dropout=0.5, return_sequences=True),
    # 第二个LSTM层
    tf.keras.layers.LSTM(150, dropout=0.5, return_sequences=True),
    # 利用TimeDistributed对每个时间步的输出都做Dense操作(softmax激活)
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(tokenizer.dict_size, activation='softmax')),
])


model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.sparse_categorical_crossentropy
)


# model.fit(
#     dataset.generator(),
#     steps_per_epoch=dataset.steps,
#     epochs=10
# )

# model.save('poem_model.h5')
model = tf.keras.models.load_model('poem_model.h5')

def predict(model, token_ids):
    """
    在概率值为前100的词中选取一个词(按概率分布的方式)
    :return: 一个词的编号(不包含[PAD][NONE][START])
    """
    # 预测各个词的概率分布
    # -1 表示只要对最新的词的预测
    # 3: 表示不要前面几个标记符
    _probas = model.predict([token_ids, ])[0, -1, 3:]
    # 按概率降序，取前100
    p_args = _probas.argsort()[-100:][::-1] # 此时拿到的是索引
    p = _probas[p_args] # 根据索引找到具体的概率值
    p = p / sum(p) # 归一
    # 按概率抽取一个
    target_index = np.random.choice(len(p), p=p)
    # 前面预测时删除了前几个标记符，因此编号要补上3位，才是实际在tokenizer词典中的编号
    return p_args[target_index] + 3



def generate_random_poem(tokenizer, model, text=""):
    """
    随机生成一首诗
    :param tokenizer: 分词器
    :param model: 古诗模型
    :param text: 古诗的起始字符串，默认为空
    :return: 一首古诗的字符串
    """
    # 将初始字符串转成token_ids，并去掉结束标记[END]
    token_ids = tokenizer.encode(text)[:-1]
    while len(token_ids) < MAX_LEN:
        # 预测词的编号
        target = predict(model, token_ids)
        # 保存结果
        token_ids.append(target)
        # 到达END
        if target == tokenizer.end_id:
            break

    return "".join(tokenizer.decode(token_ids))


def generate_acrostic_poem(tokenizer, model, heads):
    """
    生成一首藏头诗
    :param tokenizer: 分词器
    :param model: 古诗模型
    :param heads: 藏头诗的头
    :return: 一首古诗的字符串
    """
    # token_ids，只包含[START]编号
    token_ids = [tokenizer.start_id, ]
    # 逗号和句号标记编号
    punctuation_ids = {tokenizer.token_to_id("，"), tokenizer.token_to_id("。")}
    content = []
    # 为每一个head生成一句诗
    for head in heads:
        content.append(head)
        # head转为编号id，放入列表，用于预测
        token_ids.append(tokenizer.token_to_id(head))
        # 开始生成一句诗
        target = -1;
        while target not in punctuation_ids: # 遇到逗号、句号，说明本句结束，开始下一句
            # 预测词的编号
            target = predict(model, token_ids)
            # 因为可能预测到END，所以加个判断
            if target > 3:
                # 保存结果到token_ids中，下一次预测还要用
                token_ids.append(target)
                content.append(tokenizer.id_to_token(target))

    return "".join(content)

#
# print(generate_random_poem(tokenizer, model, "春眠不觉晓，"))
# print(generate_random_poem(tokenizer, model, "白日依山尽，"))
# print(generate_random_poem(tokenizer, model, "秦时明月汉时关，"))

for i in range(100):
    # print(generate_acrostic_poem(tokenizer, model, heads="我爱小鱼"))
    # s= generate_acrostic_poem(tokenizer, model, heads="我爱小鱼")
    # filename = 'test_text.txt'
    # with open(filename, 'a+') as file_object:
    #     file_object.write(s+"\n")
    # print(generate_acrostic_poem(tokenizer, model, heads=""))
    print(generate_random_poem(tokenizer, model, "权哥真的帅，"))
