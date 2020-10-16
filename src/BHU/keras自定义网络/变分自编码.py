import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
# 知乎
url='''https://zhuanlan.zhihu.com/p/59481985'''
URL = '''https://www.tensorflow.org/tutorials/structured_data/imbalanced_data'''
# 采样网络
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# 编码器
class Encoder(layers.Layer):
    def __init__(self, latent_dim=32,
                 intermediate_dim=64, name='encoder', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation='relu')
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        h1 = self.dense_proj(inputs)
        z_mean = self.dense_mean(h1)
        z_log_var = self.dense_log_var(h1)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


# 解码器
class Decoder(layers.Layer):
    def __init__(self, original_dim,
                 intermediate_dim=64, name='decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation='relu')
        self.dense_output = layers.Dense(original_dim, activation='sigmoid')

    def call(self, inputs):
        h1 = self.dense_proj(inputs)
        return self.dense_output(h1)


# 变分自编码器
class VAE(tf.keras.Model):
    def __init__(self, original_dim, latent_dim=32,
                 intermediate_dim=64, name='encoder', **kwargs):
        super(VAE, self).__init__(name=name, **kwargs)

        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim,
                               intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim=original_dim,
                               intermediate_dim=intermediate_dim)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)

        kl_loss = -0.5 * tf.reduce_sum(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        self.add_loss(kl_loss)

        return reconstructed


(x_train, _), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
vae = VAE(784, 32, 64)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

vae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
vae.fit(x_train, x_train, epochs=3, batch_size=64)

'''
自定义训练
'''
train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)
original_dim = 784
vae = VAE(original_dim, 64, 32)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
mse_loss_fn = tf.keras.losses.MeanSquaredError()
loss_metric = tf.keras.metrics.Mean()
# 每个epoch迭代.
for epoch in range(3):
    print('Start of epoch %d' % (epoch,))

# 取出每个batch的数据并训练.
for step, x_batch_train in enumerate(train_dataset):
    with tf.GradientTape() as tape:
        reconstructed = vae(x_batch_train)
        # 计算 reconstruction loss
        loss = mse_loss_fn(x_batch_train, reconstructed)
        loss += sum(vae.losses)  # 添加 KLD regularization loss

    grads = tape.gradient(loss, vae.trainable_variables)
    optimizer.apply_gradients(zip(grads, vae.trainable_variables))

    loss_metric(loss)

    if step % 100 == 0:
        print('step %s: mean loss = %s' % (step, loss_metric.result()))
