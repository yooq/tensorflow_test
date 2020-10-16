'''
Attetion(Q,K,V) = softmax(QK/sq(d))V
'''

import tensorflow as tf
import numpy as np
print(tf.__version__)


def scaled_dot_product_attention(q, k, v, mask):
    """   计算注意力权重。
          q, k, v 必须具有匹配的前置维度。
          k, v 必须有匹配的倒数第二个维度，例如：seq_len_k = seq_len_v。
          虽然 mask 根据其类型（填充或前瞻）有不同的形状，
          但是 mask 必须能进行广播转换以便求和。

          参数:
            q: 请求的形状 == (..., seq_len_q, depth)
            k: 主键的形状 == (..., seq_len_k, depth)
            v: 数值的形状 == (..., seq_len_v, depth_v)
            mask: Float 张量，其形状能转换成
                  (..., seq_len_q, seq_len_k)。默认为None。

          返回值:
        输出，注意力权重
    """
    matmul_qk = tf.matmul(q, k,transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # 缩放 matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # 将 mask 加入到缩放的张量上。
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax 在最后一个轴（seq_len_k）上归一化，因此分数
    # 相加等于1。
    attention_weights = tf.nn.softmax(scaled_attention_logits,axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights
