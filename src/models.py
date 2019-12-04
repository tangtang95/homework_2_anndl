from datetime import datetime

import tensorflow as tf
import os


class NotebookModel(object):
    """
    Lab notebook model adapted for this Kaggle competition
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


def compile_model(model, lr=0.001, use_sigmoid=True):
    def iou(y_true, y_pred):
        # from pobability to predicted class {0, 1}
        y_pred = tf.cast(y_pred > 0.5, tf.float32)  # for sigmoid only

        # A and B
        intersection = tf.reduce_sum(y_true * y_pred)
        # A or B
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
        # IoU
        return intersection / union

    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=loss, metrics=[iou])


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
