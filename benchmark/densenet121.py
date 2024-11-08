import keras
from keras import layers, models, losses

import utilities.pipeline as pipeline
import utilities.viewer as viewer
import utilities.evaluate as evaluate

# For if the certificate verification fails
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def run():
    base_model = keras.applications.DenseNet121(
        name="DenseNet121",
        include_top=False,
        weights="imagenet",
        input_shape=(256, 256, 3),  # Grayscale input
        pooling='avg',  # Use average pooling
    )

    base_model.trainable = False

    # Preprocess the dataset
    preprocessed_dataset = pipeline.get_data()

    # Split dataset into training and validation sets
    train_size = int(0.7 * len(preprocessed_dataset))
    val_size = len(preprocessed_dataset) - int(0.85 * len(preprocessed_dataset))

    train_dataset = preprocessed_dataset.take(train_size)
    val_dataset = preprocessed_dataset.skip(train_size)
    test_dataset = preprocessed_dataset.skip(train_size + val_size)

    # Create the model
    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.summary()

    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )

    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=15,
        batch_size=24
    )

    model.save("utilities/models/densenet121.keras", overwrite=True, zipped=True)

    viewer.plot_training_history(history)

    evaluate.far_ffr_test(test_dataset, "utilities/models/densenet121.keras")
    
    return viewer.get_final_metrics(history)
    