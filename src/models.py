from datetime import datetime

import tensorflow as tf
import os
import numpy as np


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
    true_negatives = (1-true1) * (1 - pred1)  # = 1 -true1 - pred1 + true1*pred1
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
