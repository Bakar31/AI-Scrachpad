import tensorflow as tf

input_height = 224
input_width = 224

vgg_model = tf.keras.models.Sequential([
    tf.keras.Input((input_height, input_width, 3)),
    tf.keras.layers.Conv2D(64, (3, 3), 1, padding = 'same', activation='relu'),
    tf.keras.layers.Conv2D(64, (3, 3), 1, padding = 'same', activation='relu'),
    tf.keras.layers.MaxPool2D((2, 2), 2),

    tf.keras.layers.Conv2D(128, (3, 3), 1, padding = 'same', activation='relu'),
    tf.keras.layers.Conv2D(128, (3, 3), 1, padding = 'same', activation='relu'),
    tf.keras.layers.MaxPool2D((2, 2), 2),

    tf.keras.layers.Conv2D(256, (3, 3), 1, padding = 'same', activation='relu'),
    tf.keras.layers.Conv2D(256, (3, 3), 1, padding = 'same', activation='relu'),
    tf.keras.layers.Conv2D(256, (3, 3), 1, padding = 'same', activation='relu'),
    tf.keras.layers.Conv2D(256, (3, 3), 1, padding = 'same', activation='relu'),
    tf.keras.layers.MaxPool2D((2, 2), 2),

    tf.keras.layers.Conv2D(512, (3, 3), 1, padding = 'same', activation='relu'),
    tf.keras.layers.Conv2D(512, (3, 3), 1, padding = 'same', activation='relu'),
    tf.keras.layers.Conv2D(512, (3, 3), 1, padding = 'same', activation='relu'),
    tf.keras.layers.Conv2D(512, (3, 3), 1, padding = 'same', activation='relu'),
    tf.keras.layers.MaxPool2D((2, 2), 2),

    tf.keras.layers.Conv2D(512, (3, 3), 1, padding = 'same', activation='relu'),
    tf.keras.layers.Conv2D(512, (3, 3), 1, padding = 'same', activation='relu'),
    tf.keras.layers.Conv2D(512, (3, 3), 1, padding = 'same', activation='relu'),
    tf.keras.layers.Conv2D(512, (3, 3), 1, padding = 'same', activation='relu'),
    tf.keras.layers.MaxPool2D((2, 2), 2),

    tf.keras.layers.Dense(4096, activation = 'relu'),
    tf.keras.layers.Dense(4096, activation = 'relu'),
    tf.keras.layers.Dense(1000, activation = 'softmax')
])

vgg_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(vgg_model.summary())