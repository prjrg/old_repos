from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

from define_model import define_network

train_dir = 'data/train'
val_dir = 'data/test'

train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)

train_generator = train_gen.flow_from_directory(train_dir, target_size=(48, 48), batch_size=1, color_mode="grayscale", class_mode="categorical")
validation_generator = val_gen.flow_from_directory(val_dir, target_size=(48, 48), batch_size=1, color_mode="grayscale", class_mode="categorical")

model = define_network(48)

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])

model_history = model.fit_generator(train_generator, steps_per_epoch=28709 // 1, epochs=15, validation_data=validation_generator, validation_steps=7178 // 1)

model.save_weights('data/model.h5')