import importlib

from keras.models import Model
from keras.layers import Dropout, Dense
from keras.optimizers import Adam
from img_plugin.core.utils.losses import earth_movers_distance


class Nima:
    def __init__(self, base_model_name, n_classes=10, learning_rate=0.001, dropout_rate=0, loss=earth_movers_distance,
                 decay=0, weights='imagenet'):
        self.n_classes = n_classes
        self.base_model_name = base_model_name
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.loss = loss
        self.decay = decay
        self.weights = weights  # 使用默认参数imagenet
        self._get_base_module()

    def _get_base_module(self):
        # import Keras base model module
        if self.base_model_name == 'InceptionV3':
            self.base_module = importlib.import_module('keras.applications.inception_v3')
        elif self.base_model_name == 'InceptionResNetV2':
            self.base_module = importlib.import_module('keras.applications.inception_resnet_v2')
        else:
            # 导入Keras的MobileNet网络
            self.base_module = importlib.import_module('keras.applications.' + self.base_model_name.lower())

    def build(self):
        # get base model class，提取网络
        BaseCnn = getattr(self.base_module, self.base_model_name)

        # load pre-trained model, weights不为空, 则加载默认参数
        self.base_model = BaseCnn(input_shape=(224, 224, 3), weights=self.weights, include_top=False, pooling='avg')

        # add dropout and dense layer，增加1个dropout和dense全连接
        x = Dropout(self.dropout_rate)(self.base_model.output)
        x = Dense(units=self.n_classes, activation='softmax')(x)

        self.nima_model = Model(self.base_model.inputs, x)

        # pd_labels = self.nima_model.outputs[0]  # 输出
        # print(pd_labels)

    def compile(self):
        self.nima_model.compile(optimizer=Adam(lr=self.learning_rate, decay=self.decay), loss=self.loss)

    def preprocessing_function(self):
        return self.base_module.preprocess_input
