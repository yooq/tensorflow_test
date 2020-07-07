'''
完成一个残差网络

'''

from tensorflow.keras import layers,Sequential
import tensorflow as tf

class BaseBlock(layers.Layer):
    '''
        残差网络18 的一个小结构  ,一个BB有两个卷积
    '''
    def __init__(self,filter_num,stride =1):
        super(BaseBlock,self).__init__()
        self.conv1 = layers.Conv2D(filter_num,kernel_size=(3,3),strides=stride,padding='same')
        self.bn1 = layers.BatchNormalization()
        self.activ = layers.Activation('relu')

        self.conv2 = layers.Conv2D(filter_num,kernel_size=(3,3),strides=1,padding='same')
        self.bn2 = layers.BatchNormalization()

        if stride!=1:
            # 如果stride 不为1，会导致维度变化，最终导致残差不能相加，利用1x1卷积做下采样使得维度一致
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num,kernel_size=(1,1),strides=stride,padding='same'))
        else:
            self.downsample = lambda x:x

    def call(self,inputs,training=None):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.activ(out)

        out = self.conv2(out)
        out = self.bn2(out)

        indentify = self.downsample(inputs)

        out = layers.add([out,indentify])
        out = self.activ(out)

        return out



class ResNet(tf.keras.Model):
     def __init__(self,layer_dims,num_classes = 100): #layer_dims 表示一个残差模块中有几个卷积，对应下面的build_resblock中的blocks
         super(ResNet,self).__init__()
     #     预处理层,第一层
     #     self.stem = Sequential([layers.Conv2D(64,(3,3),strides=(1,1))
     #                            ,layers.BatchNormalization()
     #                             ,layers.Activation('relu')
     #                             ,layers.MaxPool2D(pool_size=(2,2),strides=(1,1),padding='same')
     #                             ])

         self.stem = layers.Conv2D(64,(3,3),strides=(1,1))
         self.stem1 = layers.BatchNormalization()
         self.stem2 = layers.Activation('relu')
         self.stem3 = layers.MaxPool2D(pool_size=(2,2),strides=(1,1),padding='same')
     # block层

         self.layer1 = self.build_resblock(64,layer_dims[0])
         self.layer2 = self.build_resblock(128, layer_dims[1],stride=2)
         self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
         self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)

         # [b,512,h,w]  为标具体数值表示未知
     # 全连接层
         self.avrage = layers.GlobalAveragePooling2D() #作用：转层全连接层,个人认为flatten 也能达到预期目的
         self.fc = layers.Dense(num_classes)


     def build_resblock(self,filter_num,blocks,stride =1):
         '''
         :param filter_num:
         :param blocks:  表示基础模块的数量
         :param stride:
         :return:
         '''
         res_blocks = Sequential()
         res_blocks.add(BaseBlock(filter_num,stride))

         for _ in range(blocks):
                res_blocks.add(BaseBlock(filter_num,stride=1))  #strid 为1主要是为了残差能相加，防止维度变了，如果维度变了需要使用（1,1）网络去做下采样
         return  res_blocks




     def call(self,inputs ,training=None):

        x = self.stem(inputs)
        x = self.stem1(x)
        x = self.stem2(x)
        x = self.stem3(x)



        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x= self.avrage(x)
        out = self.fc(x)

        return out



model = ResNet([2,2,2,2],num_classes=100)
model.build(input_shape=(None,32,32,3))

# input =tf.keras.Input(shape=[32,32,3])
# resnet.build(input_shape=input)

model.summary()


