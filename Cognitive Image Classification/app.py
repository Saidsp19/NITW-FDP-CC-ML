from __future__ import division, print_function

import os
import gc
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf

# Flask libraries
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

from tqdm import tqdm
from PIL import Image

from tensorflow import keras
import tensorflow_hub as hub

from os.path import abspath

image_path_id = abspath("/test/2216849948.jpg")
image_path = abspath("/test/")
IMAGE_SIZE = (512, 512)

# We used this flag to test combinations using only TF.Keras models
onlykeras = False

used_models_keras = {
                    "mobilenet": abspath('models\cropnet_mobilenetv3\cropnet'),
                    "efficientnetb4": abspath('models\efficientnetb4\efficientnetb4_all_e14.h5')
                     }

stacked_mean = True


def build_mobilenet3(img_size=(224, 224), weights= abspath('models\cropnet_mobilenetv3\cropnet')):

    classifier = hub.KerasLayer(weights)

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=img_size + (3,)),
        hub.KerasLayer(classifier, trainable=False)])

    return model


def image_augmentations(image):

    p_spatial = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_rotate = tf.random.uniform([], 0, 1.0, dtype=tf.float32)

    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    if p_spatial > 0.75:
        image = tf.image.transpose(image)

    if p_rotate > 0.75:
        image = tf.image.rot90(image, k=3)
    elif p_rotate > 0.5:
        image = tf.image.rot90(image, k=2)
    elif p_rotate > 0.25:
        image = tf.image.rot90(image, k=1)

    image = tf.image.resize(image, size=IMAGE_SIZE)
    image = tf.reshape(image, [*IMAGE_SIZE, 3])

    return image


def read_preprocess_file(img_path, normalize=False):
    image = Image.open(img_path)
    if normalize:
        img_scaled = np.array(image) / 255.0
    else:
        img_scaled = np.array(image)
    img_scaled = img_scaled.astype(np.float32)
    return (image.size[0], image.size[1]), img_scaled


def create_image_tiles(origin_dim, processed_img):
    crop_size = 512
    img_list = []
    # Cut image into 4 overlapping patches
    for x in [0, origin_dim[1] - crop_size]:
        for y in [0, origin_dim[0] - crop_size]:
            img_list.append(processed_img[x:x + crop_size, y:y + crop_size, :])
    # Keep one additional center cropped image 
    img_list.append(cv2.resize(processed_img[:, 100:700, :], dsize=(crop_size, crop_size)))
    return np.array(img_list)


def augment_tiles_light(tiles, ttas=2):
    # Copy central croped image to have same ratio to augmented images
    holdout = np.broadcast_to(tiles[-1, :, :, :], (ttas,) + tiles.shape[1:])
    augmented_batch = tf.map_fn(lambda x: image_augmentations(x), tf.concat(
        [tiles[:-1, :, :, :] for _ in range(ttas)], axis=0))
    return tf.concat([augmented_batch, holdout], axis=0)


def cut_crop_image(processed_img):
    image = tf.image.central_crop(processed_img, 0.8)
    image = tf.image.resize(image, (224, 224))
    return np.expand_dims(image, 0)


# CropNet class 6 (unknown) is distributed evenly over all 5 classes to match problem setting
def distribute_unknown(propabilities):
    return propabilities[:, :-1] + np.expand_dims(propabilities[:, -1] / 5, 1)


def multi_predict_tfhublayer(img_path, modelinstance):
    img = cut_crop_image(read_preprocess_file(img_path, True)[1])
    yhat = modelinstance.predict(img)
    return np.mean(distribute_unknown(yhat), axis=0)


def multi_predict_keras(img_path, modelinstance, *args):
    augmented_batch = augment_tiles_light(create_image_tiles(
        *read_preprocess_file(img_path)))
    Yhat = modelinstance.predict(augmented_batch)
    return np.mean(Yhat, axis=0)


def predict_and_vote(image_list, modelinstances, onlykeras):
    predictions = []
    with tqdm(total=len(image_list)) as process_bar:
        for img_path in image_list:
            process_bar.update(1)
            Yhats = np.vstack([func(img_path, modelinstance) for func, modelinstance in modelinstances])
            if onlykeras:
                predictions.append(np.argmax(np.sum(Yhats, axis=0)))
            else:
                predictions.append(Yhats)
    return predictions


def final_predicts(image_path_id1):

    inference_models = []
    submission_df = pd.DataFrame(columns={"image_id", "label"})
    submission_df["image_id"] = abspath('/test')
    submission_df["label"] = 0

    if "mobilenet" in used_models_keras:
        model_mobilenet = build_mobilenet3(weights=used_models_keras["mobilenet"])
        inference_models.append((multi_predict_tfhublayer, model_mobilenet))

    if "efficientnetb4" in used_models_keras:
        model_efficientnetb4 = keras.models.load_model(used_models_keras["efficientnetb4"], compile=False)
        inference_models.append((multi_predict_keras, model_efficientnetb4))

    submission_df["label"] = predict_and_vote([image_path_id1], inference_models, onlykeras)

    mobilenet = submission_df['label'][0][0]
    efficientnetb4 = submission_df['label'][0][1]

    final_result = np.argmax([np.sum(e) for e in zip(mobilenet, efficientnetb4)])

    return int(final_result)


# Define a flask app
app = Flask(__name__)

final_Val = {0: "Cassava Bacterial Blight (CBB)",
             1: "Cassava Brown Streak Disease (CBSD)",
             2: "Cassava Green Mottle (CGM)",
             3: "Cassava Mosaic Disease (CMD)",
             4: "Healthy",
             }


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        submission_result = final_predicts(file_path)
        result_1 = final_Val.get(int(submission_result))
        return result_1

    return None


if __name__ == '__main__':
    app.run(debug=True)
