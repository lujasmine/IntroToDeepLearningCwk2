from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D
from keras import regularizers
from concrete_dropout import ConcreteDropout

def get_model():

    # Model parameters
    rows, cols = 28, 28
    input_shape = (rows, cols, 1)

    nb_classes = 10
    hidden_size = 128

    inp = Input(shape=input_shape)

    conv_layer = Conv2D(32, (3, 3), padding='same', activation='relu')(inp)
    max_pool = MaxPooling2D(pool_size=(1, 2))(conv_layer)
    conv_drop = Dropout(0.25)(max_pool)

    conv_layer_2 = Conv2D(32, (3, 3), padding='same', activation='relu')(conv_drop)
    max_pool_2 = MaxPooling2D(pool_size=(1, 2))(conv_layer_2)
    conv_drop_2 = Dropout(0.25)(max_pool_2)

    flat = Flatten()(conv_drop_2)

    hidden_1 = Dense(hidden_size, activation='sigmoid', kernel_initializer='lecun_normal')(flat)
    h1 = BatchNormalization()(hidden_1)

    hidden_2 = Dense(hidden_size, activation='sigmoid', kernel_initializer='lecun_normal')(h1)
    h2 = BatchNormalization()(hidden_2)

    out = Dense(nb_classes, activation='softmax')(h2)

    model = Model(inputs=inp, outputs=out)

    print(model.summary())

    return model


if __name__ == '__main__':

    model = get_model()
