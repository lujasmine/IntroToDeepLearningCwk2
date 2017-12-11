from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization, Conv2D
from keras import regularizers
from concrete_dropout import ConcreteDropout

def get_model():

    # Model parameters
    rows, cols = 28, 28
    input_shape = (rows, cols, 1)

    nb_classes = 10
    hidden_size = 128

    inp = Input(shape=input_shape)
    h_1 = Conv2D(32, (3, 3), padding='same', activation='relu')(inp)
    flat = Flatten()(h_1)

    hidden_1 = ConcreteDropout(Dense(hidden_size, activation='sigmoid', kernel_initializer='glorot_uniform'))(flat)
    # h1_drop = Dropout(0.15)(hidden_1)
    h1 = BatchNormalization()(hidden_1)

    hidden_2 = Dense(hidden_size, activation='sigmoid', kernel_initializer='glorot_uniform')(h1)
    h2_drop = Dropout(0.1)(hidden_2)
    h2 = BatchNormalization()(h2_drop)

    out = Dense(nb_classes, activation='softmax')(h2)

    model = Model(inputs=inp, outputs=out)

    print(model.summary())

    return model


if __name__ == '__main__':

    model = get_model()
