from datetime import datetime

import tensorflow as tf
import os
import numpy as np
from typing import List


class TransferVGG2(object):

    @classmethod
    def get_model(cls, img_w=256, img_h=256, decoding_start_f=512):
        vgg: tf.keras.Model = tf.keras.applications.VGG19(include_top=False, weights="imagenet",
                                                          input_shape=(img_h, img_w, 3))
        vgg_layers: List[tf.keras.layers.Layer] = vgg.layers

        k_init = "he_normal"

        # Decoder
        layer_idx = [-2, -10, 8, 4, 1]
        decoder_depth = 4
        prev_layer = vgg_layers[layer_idx[0]].output

        for i in range(0, decoder_depth):
            up_sampling = tf.keras.layers.Conv2DTranspose(filters=decoding_start_f // (2**i), strides=(2, 2),
                                                          padding="same", activation="relu",
                                                          kernel_size=(3, 3),
                                                          kernel_initializer=k_init)(prev_layer)
            merge = tf.keras.layers.concatenate([up_sampling, vgg_layers[layer_idx[i+1]].output])
            merge = tf.keras.layers.Conv2D(filters=decoding_start_f // (2**i), strides=(1, 1),
                                           padding="same", activation="relu", kernel_size=(3, 3),
                                           kernel_initializer=k_init)(merge)
            prev_layer = merge

        # Output
        output = tf.keras.layers.Conv2DTranspose(filters=1,
                                                 kernel_size=(3, 3), activation='sigmoid',
                                                 padding='same',
                                                 kernel_initializer='glorot_normal')(prev_layer)

        model = tf.keras.Model(inputs=vgg_layers[0].input, outputs=output)

        optimizer = 'adam'
        loss = bce_dice_loss
        metrics = [map_iou]

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        return model


class TransferVGG(object):

    @classmethod
    def get_model(cls, img_w=256, img_h=256, decoding_start_f=512, keep_last_max_pooling=True):
        vgg: tf.keras.Model = tf.keras.applications.VGG19(include_top=False, weights="imagenet",
                                                          input_shape=(img_h, img_w, 3))
        vgg_layers: List[tf.keras.layers.Layer] = vgg.layers

        k_init = "he_normal"

        # Decoder
        decoder_depth = 4
        if keep_last_max_pooling:
            layer_idx = [-1, -10, 8, 4, 1]
            prev_layer = vgg_layers[layer_idx[0]].output
            up_sampling = tf.keras.layers.Conv2DTranspose(filters=decoding_start_f, strides=(2, 2),
                                                          padding="same", activation="relu",
                                                          kernel_size=(3, 3),
                                                          kernel_initializer=k_init)(prev_layer)
            prev_layer = up_sampling
        else:
            layer_idx = [-2, -10, 8, 4, 1]
            prev_layer = vgg_layers[layer_idx[0]].output

        for i in range(0, decoder_depth):
            up_sampling = tf.keras.layers.Conv2DTranspose(filters=decoding_start_f // (2**i), strides=(2, 2),
                                                          padding="same", activation="relu",
                                                          kernel_size=(3, 3),
                                                          kernel_initializer=k_init)(prev_layer)
            merge = tf.keras.layers.concatenate([up_sampling, vgg_layers[layer_idx[i+1]].output])
            merge = tf.keras.layers.Conv2D(filters=decoding_start_f // (2**i), strides=(1, 1),
                                           padding="same", activation="relu", kernel_size=(3, 3),
                                           kernel_initializer=k_init)(merge)
            prev_layer = merge

        # Output
        output = tf.keras.layers.Conv2DTranspose(filters=1,
                                                 kernel_size=(3, 3), activation='sigmoid',
                                                 padding='same',
                                                 kernel_initializer='glorot_normal')(prev_layer)

        model = tf.keras.Model(inputs=vgg_layers[0].input, outputs=output)

        optimizer = 'adam'
        loss = bce_dice_loss
        metrics = [map_iou]

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        return model


class TransferResNet50V2(object):
    @classmethod
    def get_model(cls, img_w=256, img_h=256, decoding_start_f=256):
        k_init = 'he_normal'  # kernel initializer

        # Encoder
        inception_resnet: tf.keras.Model = tf.keras.applications.ResNet50V2(include_top=False,
                                                                            weights='imagenet',
                                                                            input_shape=(img_h, img_w, 3))
        encoder_layers: List[tf.keras.layers.Layer] = inception_resnet.layers.copy()

        for layer in encoder_layers:
            layer.trainable = False

        # Decoder
        up_sampling_1 = tf.keras.layers.Conv2DTranspose(filters=decoding_start_f, strides=(2, 2),
                                                        padding="same", activation="relu",
                                                        kernel_size=(3, 3),
                                                        kernel_initializer=k_init)(encoder_layers[-1].output)
        merge_1 = tf.keras.layers.concatenate([encoder_layers[-46].output, up_sampling_1])
        merge_1 = tf.keras.layers.Conv2D(kernel_size=(3, 3), kernel_initializer=k_init, activation="relu",
                                         padding="same", filters=decoding_start_f)(merge_1)

        up_sampling_2 = tf.keras.layers.Conv2DTranspose(filters=decoding_start_f // 2, strides=(2, 2),
                                                        padding="same", activation="relu",
                                                        kernel_size=(3, 3), kernel_initializer=k_init)(merge_1)
        merge_2 = tf.keras.layers.concatenate([encoder_layers[-112].output, up_sampling_2])
        merge_2 = tf.keras.layers.Conv2D(kernel_size=(3, 3), kernel_initializer=k_init, activation="relu",
                                         padding="same", filters=decoding_start_f // 2)(merge_2)

        up_sampling_3 = tf.keras.layers.Conv2DTranspose(filters=decoding_start_f // 4, strides=(2, 2),
                                                        padding="same", activation="relu",
                                                        kernel_size=(3, 3), kernel_initializer=k_init)(merge_2)

        merge_3 = tf.keras.layers.concatenate([encoder_layers[-158].output, up_sampling_3])
        merge_3 = tf.keras.layers.Conv2D(kernel_size=(3, 3), kernel_initializer=k_init, activation="relu",
                                         padding="same", filters=decoding_start_f // 4)(merge_3)

        up_sampling_4 = tf.keras.layers.Conv2DTranspose(filters=decoding_start_f // 4, strides=(2, 2),
                                                        padding="same", activation="relu",
                                                        kernel_initializer=k_init, kernel_size=(3, 3))(merge_3)
        merge_4 = tf.keras.layers.concatenate([encoder_layers[2].output, up_sampling_4])
        merge_4 = tf.keras.layers.Conv2D(filters=decoding_start_f // 4, strides=(1, 1),
                                         padding="same", activation="relu", kernel_initializer=k_init,
                                         kernel_size=(3, 3))(merge_4)

        up_sampling_5 = tf.keras.layers.Conv2DTranspose(filters=decoding_start_f // 8, strides=(2, 2),
                                                        padding="same", activation="relu",
                                                        kernel_size=(3, 3), kernel_initializer=k_init)(merge_4)
        up_sampling_5 = tf.keras.layers.Conv2D(filters=decoding_start_f // 16, strides=(1, 1),
                                               padding="same", activation="relu",
                                               kernel_size=(3, 3), kernel_initializer=k_init)(up_sampling_5)

        # Output layer
        output = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(3, 3), activation='sigmoid', padding='same',
                                                 kernel_initializer='glorot_normal')(up_sampling_5)

        model = tf.keras.Model(inputs=encoder_layers[0].input, outputs=output)

        optimizer = 'adam'
        loss = bce_dice_loss
        metrics = [map_iou]

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        return model


class Unet(object):

    @classmethod
    def get_model(cls, img_w=256, img_h=256):
        """
        Taken from https://www.kaggle.com/weiji14/yet-another-keras-u-net-data-augmentation#Part-2---Build-model
        """
        n_ch_exps = [4, 5, 6, 7, 8, 9]  # the n-th deep channel's exponent i.e. 2**n 16,32,64,128,256
        k_size = (3, 3)  # size of filter kernel
        k_init = 'he_normal'  # kernel initializer

        ch_axis = 3
        input_shape = (img_w, img_h, 3)

        inp = tf.keras.layers.Input(shape=input_shape)
        encodeds = []

        # encoder
        enc = inp
        print(n_ch_exps)
        for l_idx, n_ch in enumerate(n_ch_exps):
            enc = tf.keras.layers.Conv2D(filters=2 ** n_ch, kernel_size=k_size, activation='relu', padding='same',
                                         kernel_initializer=k_init)(enc)
            enc = tf.keras.layers.Dropout(0.1 * l_idx, )(enc)
            enc = tf.keras.layers.Conv2D(filters=2 ** n_ch, kernel_size=k_size, activation='relu', padding='same',
                                         kernel_initializer=k_init)(enc)
            encodeds.append(enc)
            # print(l_idx, enc)
            if n_ch < n_ch_exps[-1]:  # do not run max pooling on the last encoding/downsampling step
                enc = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(enc)

        # decoder
        dec = enc
        print(n_ch_exps[::-1][1:])
        decoder_n_chs = n_ch_exps[::-1][1:]
        for l_idx, n_ch in enumerate(decoder_n_chs):
            l_idx_rev = len(n_ch_exps) - l_idx - 2  #
            dec = tf.keras.layers.Conv2DTranspose(filters=2 ** n_ch, kernel_size=k_size, strides=(2, 2),
                                                  activation='relu', padding='same', kernel_initializer=k_init)(dec)
            dec = tf.keras.layers.concatenate([dec, encodeds[l_idx_rev]], axis=ch_axis)
            dec = tf.keras.layers.Conv2D(filters=2 ** n_ch, kernel_size=k_size, activation='relu', padding='same',
                                         kernel_initializer=k_init)(dec)
            dec = tf.keras.layers.Dropout(0.1 * l_idx)(dec)
            dec = tf.keras.layers.Conv2D(filters=2 ** n_ch, kernel_size=k_size, activation='relu', padding='same',
                                         kernel_initializer=k_init)(dec)

        outp = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=k_size, activation='sigmoid', padding='same',
                                               kernel_initializer='glorot_normal')(dec)

        model = tf.keras.Model(inputs=[inp], outputs=[outp])

        optimizer = 'adam'
        loss = bce_dice_loss
        metrics = [map_iou]

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        return model


class TransposeSkipConn(object):
    """
    Model improving from TransponseConvModel in the sense that now skip connections are present,
    in order to combine local and global information about the image.

    This model is implemented with functional API, instead of Sequential model
    """
    model_name = "TransponseConvolutionSkipConnections"

    @classmethod
    def get_model(cls, start_f, img_h=256, img_w=256):
        x = tf.keras.Input(shape=(img_w, img_h, 1))  # Input layer

        # Encoder module
        conv1 = tf.keras.layers.Conv2D(filters=start_f, kernel_size=(4, 4), strides=(2, 2),
                                       padding="same", activation="relu")(x)
        conv2 = tf.keras.layers.Conv2D(filters=start_f * 2, kernel_size=(3, 3), strides=(1, 1),
                                       padding="same", activation="relu")(conv1)
        maxpool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv2)
        conv3 = tf.keras.layers.Conv2D(filters=start_f * 4, kernel_size=(3, 3), strides=(2, 2),
                                       padding="same", activation="relu")(maxpool1)
        maxpool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv3)
        conv4 = tf.keras.layers.Conv2D(filters=start_f * 8, kernel_size=(3, 3), strides=(2, 2),
                                       padding="same", activation="relu")(maxpool2)

        # Decoder module
        up_sampling1 = tf.keras.layers.Conv2DTranspose(filters=(start_f * 4), strides=(2, 2),
                                                       padding="same", activation="relu",
                                                       kernel_size=(3, 3))(conv4)
        mix1 = tf.keras.layers.Add()([up_sampling1, maxpool2])

        up_sampling2 = tf.keras.layers.Conv2DTranspose(filters=(start_f * 2), strides=(4, 4),
                                                       padding="same", activation="relu",
                                                       kernel_size=(3, 3))(mix1)
        mix2 = tf.keras.layers.Add()([up_sampling2, maxpool1])

        up_sampling3 = tf.keras.layers.Conv2DTranspose(filters=(start_f * 2), strides=(2, 2),
                                                       padding="same", activation="relu", kernel_size=(3, 3))(mix2)

        mix3 = tf.keras.layers.Add()([up_sampling3, conv2])

        up_sampling4 = tf.keras.layers.Conv2DTranspose(filters=start_f, strides=(1, 1), padding="same",
                                                       activation="relu", kernel_size=(4, 4))(mix3)

        mix4 = tf.keras.layers.Add()([up_sampling4, conv1])

        up_sampling5 = tf.keras.layers.Conv2DTranspose(filters=start_f, strides=(2, 2), padding="same",
                                                       activation="relu", kernel_size=(3, 3))(mix4)

        # Output layer

        output_layer = tf.keras.layers.Conv2DTranspose(filters=1,
                                                       kernel_size=(3, 3),
                                                       strides=(1, 1),
                                                       padding='same',
                                                       activation='sigmoid')(up_sampling5)

        model = tf.keras.Model(inputs=x, outputs=output_layer)

        compile_model(model)

        return model


class TransposeConvModel(object):
    """
    Model using transpose convolution layers:
    It is a first improvement over the model provided in the notebook:
    - Number of filters improve with 2^n
    - Up-sampling layers are substituted with transpose convolutions, in order to improve the un-learnability
    of the up-sampling filters

    The model as 1M of parameter (with depth = 4, start_f=8).
    """
    model_name = "TransposeConvolution"

    @classmethod
    def get_model(cls, depth, start_f, img_h=256, img_w=256):
        model = tf.keras.Sequential(name=cls.model_name)

        # Encoder
        for i in range(depth):
            input_shape = [img_h, img_w, 3] if i == 0 else [None]

            model.add(tf.keras.layers.Conv2D(filters=start_f,
                                             kernel_size=(3, 3),
                                             strides=(1, 1),
                                             padding="same",
                                             input_shape=input_shape,
                                             activation="relu"))
            if i >= 1:
                model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

            start_f = start_f * ((i + 1) ** 2)

        # Decoder
        for i in range(depth - 1):
            start_f = start_f // ((depth - i) ** 2)

            model.add(tf.keras.layers.Conv2DTranspose(filters=start_f,
                                                      kernel_size=(3, 3),
                                                      strides=(2, 2),
                                                      padding="same",
                                                      activation="relu"))

        # Prediction layer
        model.add(tf.keras.layers.Conv2DTranspose(filters=1,
                                                  kernel_size=(3, 3),
                                                  strides=(1, 1),
                                                  padding='same',
                                                  activation='sigmoid'))

        compile_model(model)
        return model


class NotebookModel(object):
    """
    Lab notebook model adapted for this Kaggle competition
     - First submission done with {'depth': 6, 'start_f': 16} on 11th epoch
    """

    model_name = "NotebookModel"

    @classmethod
    def get_model(cls, depth, start_f, img_h=256, img_w=256):
        model = tf.keras.Sequential(name=cls.model_name)

        # Encoder
        for i in range(depth):
            input_shape = [img_h, img_w, 3] if i == 0 else [None]

            model.add(tf.keras.layers.Conv2D(filters=start_f,
                                             kernel_size=(3, 3),
                                             strides=(1, 1),
                                             padding='same',
                                             input_shape=input_shape))
            model.add(tf.keras.layers.ReLU())
            model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

            start_f *= 2

        # Decoder
        for i in range(depth):
            model.add(tf.keras.layers.UpSampling2D(2, interpolation='bilinear'))
            model.add(tf.keras.layers.Conv2D(filters=start_f // 2,
                                             kernel_size=(3, 3),
                                             strides=(1, 1),
                                             padding='same'))

            model.add(tf.keras.layers.ReLU())

            start_f = start_f // 2

        # Prediction Layer
        model.add(tf.keras.layers.Conv2D(filters=1,
                                         kernel_size=(1, 1),
                                         strides=(1, 1),
                                         padding='same',
                                         activation='sigmoid'))
        compile_model(model)
        return model


def bce_dice_loss(y_true, y_pred):
    return 0.5 * tf.keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)


def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)


def iou_single(y_true, y_pred):
    # from pobability to predicted class {0, 1}
    y_pred = tf.cast(y_pred > 0.5, tf.float32)  # for sigmoid only

    # A and B
    intersection = tf.reduce_sum(y_true * y_pred)
    # A or B
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    # IoU
    return intersection / union


def map_iou(y_true, y_pred):
    def cast_float(x):
        return tf.keras.backend.cast(x, tf.keras.backend.floatx())

    def cast_bool(x):
        return tf.keras.backend.cast(x, bool)

    def iou_loss_core(true, pred):
        intersection = true * pred
        notTrue = 1 - true
        union = true + (notTrue * pred)

        return (tf.keras.backend.sum(intersection, axis=-1) +
                tf.keras.backend.epsilon()) / (tf.keras.backend.sum(union, axis=-1) + tf.keras.backend.epsilon())

    thresholds = np.linspace(start=0.5, stop=0.95, num=10)

    # flattened images (batch, pixels)
    true = tf.keras.backend.batch_flatten(y_true)
    pred = tf.keras.backend.batch_flatten(y_pred)
    pred = cast_float(tf.keras.backend.greater(pred, 0.5))  # consider class 1 when it is greater than 0.5

    # total white pixels - (batch,)
    true_sum = tf.keras.backend.sum(true, axis=-1)
    pred_sum = tf.keras.backend.sum(pred, axis=-1)

    true1 = cast_float(tf.keras.backend.greater(true_sum, 1))
    pred1 = cast_float(tf.keras.backend.greater(pred_sum, 1))

    true_positive_mask = cast_bool(true1 * pred1)

    # separating only the possible true positives to check iou
    test_true = tf.boolean_mask(true, true_positive_mask)
    test_pred = tf.boolean_mask(pred, true_positive_mask)

    # getting iou and threshold comparisons
    iou = iou_loss_core(test_true, test_pred)
    true_positives = [cast_float(tf.keras.backend.greater(iou, thres)) for thres in thresholds]

    # mean of thresholds for true positives and total sum
    true_positives = tf.keras.backend.mean(tf.keras.backend.stack(true_positives, axis=-1), axis=-1)
    true_positives = tf.keras.backend.sum(true_positives)

    # to get images that don't have mask in both true and pred
    true_negatives = (1 - true1) * (1 - pred1)  # = 1 -true1 - pred1 + true1*pred1
    true_negatives = tf.keras.backend.sum(true_negatives)

    return (true_positives + true_negatives) / cast_float(tf.keras.backend.shape(y_true)[0])


def compile_model(model, lr=0.001):
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=loss, metrics=[map_iou])


def get_callbacks(root_path, model_name, save_checkpoint=True, save_logs=True, early_stop=False):
    exps_dir = os.path.join(root_path, 'segmentation_experiments')
    if not os.path.exists(exps_dir):
        os.makedirs(exps_dir)

    now = datetime.now().strftime('%b%d_%H-%M-%S')

    exp_dir = os.path.join(exps_dir, model_name + '_' + str(now))
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    callbacks = []

    # Model checkpoint
    # ----------------
    if save_checkpoint:
        ckpt_dir = os.path.join(exp_dir, 'ckpts')
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(ckpt_dir, 'cp_{epoch:02d}.ckpt'),
                                                           save_weights_only=True)  # False to save the model directly
        callbacks.append(ckpt_callback)

    # Visualize Learning on Tensorboard
    # ---------------------------------
    if save_logs:
        tb_dir = os.path.join(exp_dir, 'tb_logs')
        if not os.path.exists(tb_dir):
            os.makedirs(tb_dir)

        # By default shows losses and metrics for both training and validation
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_dir,
                                                     profile_batch=0,
                                                     histogram_freq=0)  # if 1 shows weights histograms
        callbacks.append(tb_callback)

    # Early Stopping
    # --------------
    if early_stop:
        es_callback = tf.keras.callback.EarlyStopping(monitor='val_loss', patience=10)
        callbacks.append(es_callback)
