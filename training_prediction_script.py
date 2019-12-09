import os

from src import models
from src.data_reader import read_train_valid_data, read_test_data
from src.models import get_callbacks
from src.utils import get_image_ids, create_submission_dict, create_csv

if __name__ == '__main__':
    cwd = os.getcwd()
    data_path = os.path.join(cwd, "data")

    # Read dataset
    #train_dataset, valid_dataset, train_img_gen, valid_img_gen = read_train_valid_data(data_path)

    # Build model
    model = models.TransposeSkipConn.get_model(start_f=30)
    #callbacks = get_callbacks(cwd, model.name)
    model.summary()


    """
    # Fit model
    model.fit(x=train_dataset,
              epochs=20,
              steps_per_epoch=len(train_img_gen),
              validation_data=valid_dataset,
              validation_steps=len(valid_img_gen),
              callbacks=callbacks)

    # Prediction and create submission file
    test_dataset, test_gen = read_test_data(data_path)
    image_ids = get_image_ids(test_gen.filenames)
    predictions = model.predict(x=test_dataset, steps=len(test_gen), verbose=1)
    results = create_submission_dict(image_ids, predictions)
    create_csv(results)
    """

