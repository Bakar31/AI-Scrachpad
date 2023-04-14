import tensorflow as tf

input_height = 224
input_width = 224

# model implementation
model = tf.keras.models.Sequential(
    [
        tf.keras.Input((input_height, input_width, 3)),
        tf.keras.layers.Conv2D(96, (11, 11), 4, activation= 'relu'),
        tf.keras.layers.MaxPool2D((3, 3), strides = (2, 2)),
        tf.keras.layers.Conv2D(256, (5, 5), padding = 'same', activation = 'relu'),
        tf.keras.layers.MaxPool2D((3, 3), strides = (2, 2)),
        tf.keras.layers.Conv2D(384, (3, 3), padding= 'same' , activation = 'relu'),
        tf.keras.layers.Conv2D(384, (3, 3), padding= 'same', activation = 'relu'),
        tf.keras.layers.Conv2D(256, (3, 3), padding= 'same', activation = 'relu'),
        tf.keras.layers.MaxPool2D((3, 3), (2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096,activation= 'relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096,activation= 'relu'),
        tf.keras.layers.Dense(1000, activation= 'softmax')
        
    ]
)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())