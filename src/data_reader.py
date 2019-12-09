import os

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.utils import get_seed

BATCH_SIZE = 16
IMG_H = IMG_W = 256


def prepare_target(x_, y_):
    y_ = tf.cast(y_, tf.int32)
    return x_, y_


def read_test_data(data_path, batch_size=BATCH_SIZE, img_h=IMG_H, img_w=IMG_W, to_rescale=True):
    """
    Read test images data

    :param data_path: path to the folder containing Segmentation Dataset
    :param batch_size: batch size of images taken from directory
    :param img_h: target resized image height
    :param img_w: target resized image width
    :param to_rescale: rescale images by 1/255 if True, otherwise no rescaling at all
    :return: test dataset and test generator
    """
    rescale_factor = 1. / 255 if to_rescale else None
    test_data_gen = ImageDataGenerator(rescale=rescale_factor)

    test_path = os.path.join(data_path, "Segmentation_Dataset", "test", "images")
    test_gen = test_data_gen.flow_from_directory(directory=test_path,
                                                 target_size=(img_w, img_h),
                                                 class_mode=None,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 interpolation='bilinear',
                                                 seed=get_seed())
    test_dataset = tf.data.Dataset.from_generator(lambda: test_gen,
                                                  output_types=tf.float32,
                                                  output_shapes=([None, img_h, img_w, 3]))
    return test_dataset, test_gen


def read_train_valid_data(data_path, batch_size=BATCH_SIZE, img_h=IMG_H, img_w=IMG_W, validation_split=0.15,
                          to_rescale=True,
                          do_augmentation=False):
    """
    Read training and validation image data resized to (img_h, img_w) with a certain batch size

    :param data_path: path to the folder containing Segmentation Dataset
    :param batch_size: batch size of images taken from directory
    :param img_h: target resized image height
    :param img_w: target resized image width
    :param validation_split: percentage to split the data into train and validation
    :param to_rescale: rescale images by 1/255 if True, otherwise no rescaling at all
    :param do_augmentation: if True, the images are augmented, otherwise there is none
    :return: training dataset, validation dataset, training generator and validation generator
    """
    rescale_factor = 1. / 255 if to_rescale else None

    # Define image generator for train images, masks and valid images, masks
    if do_augmentation:
        train_data_img_gen = ImageDataGenerator(rotation_range=45.,
                                                width_shift_range=0.1,
                                                height_shift_range=0.1,
                                                shear_range=0.2,
                                                zoom_range=0.2,
                                                horizontal_flip=True,
                                                vertical_flip=True,
                                                fill_mode='reflect',
                                                rescale=1. / 255,
                                                validation_split=validation_split)

        train_data_mask_gen = ImageDataGenerator(rotation_range=45.,
                                                 width_shift_range=0.1,
                                                 height_shift_range=0.1,
                                                 shear_range=0.2,
                                                 zoom_range=0.2,
                                                 horizontal_flip=True,
                                                 vertical_flip=True,
                                                 fill_mode='reflect',
                                                 rescale=1. / 255,
                                                 validation_split=validation_split)
    else:
        train_data_img_gen = ImageDataGenerator(rescale=rescale_factor,
                                                validation_split=validation_split)

        train_data_mask_gen = ImageDataGenerator(rescale=1. / 255,
                                                 validation_split=validation_split)
    valid_data_img_gen = ImageDataGenerator(rescale=rescale_factor,
                                            validation_split=validation_split)

    valid_data_mask_gen = ImageDataGenerator(rescale=1. / 255,
                                             validation_split=validation_split)

    # Flow images and masks from the directory
    dataset_path = os.path.join(data_path, "Segmentation_Dataset")
    image_dir = os.path.join(dataset_path, "training", "images")
    mask_dir = os.path.join(dataset_path, "training", "masks")
    train_img_gen = train_data_img_gen.flow_from_directory(directory=image_dir,
                                                           target_size=(img_w, img_h),
                                                           class_mode=None,
                                                           batch_size=batch_size,
                                                           shuffle=True,
                                                           interpolation='bilinear',
                                                           seed=get_seed(),
                                                           subset="training")

    train_mask_gen = train_data_mask_gen.flow_from_directory(directory=mask_dir,
                                                             target_size=(img_w, img_h),
                                                             class_mode=None,
                                                             color_mode="grayscale",
                                                             batch_size=batch_size,
                                                             shuffle=True,
                                                             interpolation='bilinear',
                                                             seed=get_seed(),
                                                             subset="training")
    train_gen = zip(train_img_gen, train_mask_gen)

    valid_img_gen = valid_data_img_gen.flow_from_directory(directory=image_dir,
                                                           target_size=(img_w, img_h),
                                                           class_mode=None,
                                                           batch_size=batch_size,
                                                           shuffle=False,
                                                           interpolation='bilinear',
                                                           seed=get_seed(),
                                                           subset="validation")

    valid_mask_gen = valid_data_mask_gen.flow_from_directory(directory=mask_dir,
                                                             target_size=(img_w, img_h),
                                                             class_mode=None,
                                                             color_mode="grayscale",
                                                             batch_size=batch_size,
                                                             shuffle=False,
                                                             interpolation='bilinear',
                                                             seed=get_seed(),
                                                             subset="validation")
    valid_gen = zip(valid_img_gen, valid_mask_gen)

    # Generate datasets
    train_dataset = tf.data.Dataset.from_generator(lambda: train_gen,
                                                   output_types=(tf.float32, tf.float32),
                                                   output_shapes=(
                                                       [None, img_h, img_w, 3], [None, img_h, img_w, 1]))
    valid_dataset = tf.data.Dataset.from_generator(lambda: valid_gen,
                                                   output_types=(tf.float32, tf.float32),
                                                   output_shapes=(
                                                       [None, img_h, img_w, 3], [None, img_h, img_w, 1]))
    train_dataset.map(prepare_target)
    valid_dataset.map(prepare_target)
    train_dataset.repeat()
    valid_dataset.repeat()
    return train_dataset, valid_dataset, train_img_gen, valid_img_gen
