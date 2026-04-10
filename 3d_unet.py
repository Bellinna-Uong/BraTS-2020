
"""
Commented 3D U-Net architecture for volumetric medical image segmentation.

This file only defines the model architecture.
It does not load data and does not train the model.
It is imported by the training script.
"""

from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate, Dropout


def conv_block(x, n_filters, dropout=0.0):
    """
    Standard convolution block used in U-Net.

    Each block applies:
    - Conv3D + ReLU
    - optional Dropout
    - Conv3D + ReLU
    """
    x = Conv3D(n_filters, (3, 3, 3), activation="relu", kernel_initializer="he_uniform", padding="same")(x)
    if dropout > 0:
        x = Dropout(dropout)(x)
    x = Conv3D(n_filters, (3, 3, 3), activation="relu", kernel_initializer="he_uniform", padding="same")(x)
    return x


def simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS, num_classes):
    """
    Build a simple 3D U-Net for multi-class segmentation.

    Input shape:
        (IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS)

    Output shape:
        (IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, num_classes)
    """
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS))

    # Encoder path
    c1 = conv_block(inputs, 16, dropout=0.1)
    p1 = MaxPooling3D((2, 2, 2))(c1)

    c2 = conv_block(p1, 32, dropout=0.1)
    p2 = MaxPooling3D((2, 2, 2))(c2)

    c3 = conv_block(p2, 64, dropout=0.2)
    p3 = MaxPooling3D((2, 2, 2))(c3)

    c4 = conv_block(p3, 128, dropout=0.2)
    p4 = MaxPooling3D((2, 2, 2))(c4)

    # Bottleneck
    c5 = conv_block(p4, 256, dropout=0.3)

    # Decoder path with skip connections
    u6 = UpSampling3D((2, 2, 2))(c5)
    u6 = concatenate([u6, c4])
    c6 = conv_block(u6, 128, dropout=0.2)

    u7 = UpSampling3D((2, 2, 2))(c6)
    u7 = concatenate([u7, c3])
    c7 = conv_block(u7, 64, dropout=0.2)

    u8 = UpSampling3D((2, 2, 2))(c7)
    u8 = concatenate([u8, c2])
    c8 = conv_block(u8, 32, dropout=0.1)

    u9 = UpSampling3D((2, 2, 2))(c8)
    u9 = concatenate([u9, c1])
    c9 = conv_block(u9, 16, dropout=0.1)

    # Output layer
    outputs = Conv3D(num_classes, (1, 1, 1), activation="softmax")(c9)

    model = Model(inputs=[inputs], outputs=[outputs], name="Simple_3D_UNet")
    return model
