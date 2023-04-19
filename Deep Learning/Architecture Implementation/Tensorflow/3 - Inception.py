import tensorflow as tf

input_height = 224
input_width = 224

def inception_block(x, filter_1x1, filter_Reduce_3x3, filter_3x3, filter_Reduce_5x5, filter_5x5, filter_pool):
    conv_1x1 = tf.keras.layers.Conv2D(filter_1x1, (1, 1), 1, padding = 'same', activation = 'relu')(x)

    conv_reduce_3x3 = tf.keras.layers.Conv2D(filter_Reduce_3x3, (1, 1), 1, padding = 'same', activation = 'relu')(x)
    conv_3x3 = tf.keras.layers.Conv2D(filter_3x3, (3, 3), 1, padding = 'same', activation = 'relu')(conv_reduce_3x3)

    conv_reduce_5x5 = tf.keras.layers.Conv2D(filter_Reduce_5x5, (1, 1), 1, padding = 'same', activation = 'relu')(x)
    conv_5x5 = tf.keras.layers.Conv2D(filter_5x5, (5, 5), 1, padding = 'same', activation = 'relu')(conv_reduce_5x5)

    maxpool = tf.keras.layers.MaxPool2D((3, 3), (1, 1), padding = 'same')(x)
    pool_proj = tf.keras.layers.Conv2D(filter_pool, (1, 1), 1, padding = 'same', activation = 'relu')(maxpool)

    return tf.keras.layers.concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis = -1)

def Inception():
    input_layer = tf.keras.Input((input_height, input_width, 3))
    x = tf.keras.layers.Conv2D(64, (7, 7), 2, padding = 'same', activation = 'relu')(input_layer)
    x = tf.keras.layers.MaxPool2D((3, 3), (2, 2), padding = 'same')(x)
    x = tf.keras.layers.Conv2D(192, (3, 3), 1, padding = 'same', activation = 'relu')(x)
    x = tf.keras.layers.MaxPool2D((3, 3), (2, 2), padding = 'same')(x)
    x = inception_block(x, 64, 96, 128, 16, 32, 32)
    x = inception_block(x, 128, 128, 192, 32, 96, 64)
    x = tf.keras.layers.MaxPool2D((3, 3), (2, 2), padding = 'same')(x)
    x = inception_block(x, 192, 96, 128, 16, 48, 64)
    x = inception_block(x, 160, 112, 224, 24, 64, 64)
    x = inception_block(x, 128, 128, 256, 24, 64, 64)
    x = inception_block(x, 112, 144, 288, 32, 64, 64)
    x = inception_block(x, 256, 160, 320, 32, 128, 128)
    x = tf.keras.layers.MaxPool2D((3, 3), (2, 2), padding = 'same')(x)
    x = inception_block(x, 256, 160, 320, 32, 128, 128)
    x = inception_block(x, 384, 192, 384, 48, 128, 128)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Flatten()(x)
    output_layer = tf.keras.layers.Dense(1000, activation = 'softmax')(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

inception_model = Inception()
print(inception_model.summary())