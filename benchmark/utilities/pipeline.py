import numpy as np
import keras
from keras import utils
import matplotlib.pyplot as plt
import tensorflow as tf

def convert_to_rgb(image):
    return tf.image.grayscale_to_rgb(image)

def calculate_background_color(dataset):
    pixel_values = []
    for images, _ in dataset:
        pixel_values.append(images)
    pixel_values = tf.concat(pixel_values, axis=0)
    return tf.reduce_mean(pixel_values)

def pad_with_background_color(images, mean_background_color):
    padded_images = tf.image.resize_with_crop_or_pad(images, target_height=256, target_width=256)
    mask = tf.reduce_sum(images, axis=-1) > 0
    padded_images = tf.where(mask[..., tf.newaxis], images, mean_background_color)
    return padded_images

def preprocess_image(image, std_dev, mean_background_color):
    image = convert_to_rgb(image)
    padded_image = pad_with_background_color(image, mean_background_color)
    inverted_image = 255 - padded_image
    normalized_image = inverted_image / std_dev
    return normalized_image

def calculate_std(dataset):
    pixel_values = []
    for images, _ in dataset:
        pixel_values.append(images)
    pixel_values = tf.concat(pixel_values, axis=0)
    return tf.math.reduce_std(pixel_values)

def preprocess_dataset(dataset, std_dev, mean_background_color):
    return dataset.map(
        lambda x, y: (
            preprocess_image(x, std_dev=std_dev, mean_background_color=mean_background_color),
            y
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

def display_sample(images, labels):
    plt.figure(figsize=(20, 20))
    for i in range(8):
        ax = plt.subplot(2, 4, i + 1)
        plt.imshow(images[i].numpy().squeeze(), cmap='gray')
        plt.title("Label: {}".format("Forgery" if labels[i] == 1 else "Original"))
        plt.axis("off")
    plt.show()

def get_data_cedar():
    dataset = utils.image_dataset_from_directory(
        "utilities/datasets/cedar",
        labels="inferred",
        label_mode="binary",
        class_names=["forgery", "original"],
        color_mode="grayscale",
        batch_size=24,
        image_size=(256, 256),
        shuffle=True,
        interpolation="bilinear",
        crop_to_aspect_ratio=True,
        pad_to_aspect_ratio=False,
        verbose=True,
    )

    preprocessed_dataset = preprocess_dataset(dataset, calculate_std(dataset), calculate_background_color(dataset))

    return preprocessed_dataset

def get_data_gpds():
    dataset = utils.image_dataset_from_directory(
        "utilities/datasets/gpds",
        labels="inferred",
        label_mode="binary",
        class_names=["forgery", "original"],
        color_mode="grayscale",
        batch_size=24,
        image_size=(256, 256),
        shuffle=True,
        interpolation="bilinear",
        crop_to_aspect_ratio=True,
        pad_to_aspect_ratio=False,
        verbose=True,
    )

    preprocessed_dataset = preprocess_dataset(dataset, calculate_std(dataset), calculate_background_color(dataset))

    return preprocessed_dataset
