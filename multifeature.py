def multiscale_feature_extraction(x):
    # Branch 1
    branch1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(x)
    branch1 = tf.keras.layers.BatchNormalization()(branch1)

    # Branch 2
    branch2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(x)
    branch2 = tf.keras.layers.BatchNormalization()(branch2)
    branch2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(branch2)
    branch2 = tf.keras.layers.BatchNormalization()(branch2)

    # Branch 3
    branch3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(x)
    branch3 = tf.keras.layers.BatchNormalization()(branch3)
    branch3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same')(branch3)
    branch3 = tf.keras.layers.BatchNormalization()(branch3)

    # Branch 4
    branch4 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    branch4 = tf.keras.layers.BatchNormalization()(branch4)
    branch4 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(branch4)
    branch4 = tf.keras.layers.BatchNormalization()(branch4)
    branch4 = tf.keras.layers.UpSampling2D(size=(2, 2))(branch4)

    merged_features = tf.keras.layers.concatenate([branch1, branch2, branch3, branch4], axis=-1)

    return merged_features
