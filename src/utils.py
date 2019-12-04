from datetime import datetime
import numpy as np
import tensorflow as tf

SEED = 69429


def get_seed():
    return SEED


def create_csv(results):
    """
    Create a csv file from the results dictionary output by create_submission_dict function

    :param results: submission dict
    """
    csv_fname = 'results_'
    csv_fname += datetime.now().strftime('%b%d_%H-%M-%S') + '.csv'

    with open(csv_fname, 'w') as f:
        f.write('ImageId,EncodedPixels,Width,Height\n')

        for key, value in results.items():
            f.write(key + ',' + str(value) + ',' + '256' + ',' + '256' + '\n')


def rle_encode(predicted_mask):
    """
    Given a predicted image mask with 0 and 1, it returns the different values to be used to create the csv

    :param predicted_mask: a predicted mask of an image
    :return string containing the value that represents all bounding box for the submission
    """
    # Flatten column-wise
    pixels = predicted_mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def create_submission_dict(image_ids: list, predictions: list):
    """
    Given a list of image id and its prediction list, it returns the submission dict used in create_csv function

    :param image_ids: list of image id
    :param predictions: list of prediction masks
    :return submission dict
    """
    results = {}

    for i in range(len(predictions)):
        image_id = image_ids[i]
        predicted_mask = tf.cast(predictions[i] > 0.5, tf.float32)

        key = image_id
        value = rle_encode(np.array(predicted_mask))
        results[key] = value
    return results


def get_image_ids(filename_list: list):
    """
    Given a list of filenames, returns the list of image id

    :param filename_list: list of filename of images
    :return list of image id
    """
    image_ids = np.apply_along_axis(lambda x: x[0].replace("img/", "").replace(".tif", ""), axis=1,
                                    arr=np.reshape(filename_list, (len(filename_list), 1)))
    return image_ids
