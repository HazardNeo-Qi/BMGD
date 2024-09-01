from tensorflow.keras.layers import BatchNormalization,Conv2D,Dense,Flatten,Input,MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras import Model

# 2分类逻辑回归
def MyModel():
    input_layer = Input([512])
    x = input_layer
    #x = BatchNormalization()(x)
    x = Dense(1)(x)
    output_layer = x
    model = Model(input_layer,output_layer)
    return model