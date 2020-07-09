import tensorflow.keras as keras

class Encoder(keras.layers.Layer):

    def __init__(self, rnn_type,  # rnn类型
                 input_size,
                 output_size,
                 num_layers,  # rnn层数
                 bidirectional=False):
        super(Encoder, self).__init__()
        assert rnn_type in ['GRU', 'LSTM']
        if bidirectional:
            assert output_size % 2 == 0

        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        units = int(output_size / self.num_directions)

        if rnn_type == 'GRU':
            rnnCell = [getattr(keras.layers, 'GRUCell')(units) for _ in range(num_layers)]  #表示多层
        else:
            rnnCell = [getattr(keras.layers, 'LSTMCell')(units) for _ in range(num_layers)]

        self.rnn = keras.layers.RNN(rnnCell, input_shape=(None, None, input_size),
                                    return_sequences=True, return_state=True)
        self.rnn_type = rnn_type
        self.num_layers = num_layers

        if bidirectional:
            self.rnn = keras.layers.Bidirectional(self.rnn, merge_mode='concat')  #表示双向

        self.bidirectional = bidirectional


    def __call__(self, input):  # [batch, timesteps, input_dim]

        outputs = self.rnn(input)

        output = outputs[0]
        states = outputs[1:]

        print(outputs)  # 用于测试的输出
        print(len(outputs))  # 用于测试的输出
        print(len(states))  # 用于测试的输出

        return output, states
