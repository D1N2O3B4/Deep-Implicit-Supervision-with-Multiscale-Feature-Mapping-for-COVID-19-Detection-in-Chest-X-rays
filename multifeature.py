import tensorflow as tf
from tensorflow import keras

def multiscale_feature_extraction(input_tensor):
    # Branch 1: 1x1 Convolution
    branch1 = keras.layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_tensor)

    # Branch 2: 3x3 Convolution
    branch2 = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(input_tensor)

    # Branch 3: 5x5 Convolution
    branch3 = keras.layers.Conv2D(filters=64, kernel_size=(5, 5), activation='relu')(input_tensor)

    # Concatenate the output feature maps from all branches
    merged_features = keras.layers.concatenate([branch1, branch2, branch3])

    return merged_features

# Create an example input tensor (adjust the shape according to your data)
input_tensor = keras.layers.Input(shape=(224, 224, 3))

# Apply multiscale feature extraction to the input tensor
output_features = multiscale_feature_extraction(input_tensor)

# Create a model
model = keras.Model(inputs=input_tensor, outputs=output_features)

# Print model summary
model.summary()
