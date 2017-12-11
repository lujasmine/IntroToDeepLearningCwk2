from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from data import load_data
from model import get_model

batch_size = 128
nb_epoch = 20

# Load data
(X_train, y_train, X_test, y_test) = load_data()

# Load and compile model
model = get_model()

model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
              metrics=['accuracy'])

datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, rotation_range=10)
model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size), steps_per_epoch=len(X_train) / batch_size, epochs=nb_epoch)

# model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch,
#           verbose=1, validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, verbose=1)

print("Accuracy:", score[1])
