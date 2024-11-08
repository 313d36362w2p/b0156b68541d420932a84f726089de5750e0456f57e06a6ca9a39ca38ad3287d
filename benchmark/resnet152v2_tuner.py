import keras
from keras import layers, models, losses
import keras_tuner as kt

import benchmark.utilities.pipeline as pipeline

# # For if the certificate verification fails
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

def build_model(hp):
    base_model = keras.applications.ResNet152V2(
        include_top=False,
        weights="imagenet",
        input_shape=(256, 256, 3),
        pooling='avg',
    )
    base_model.trainable = False

    # Set Tuner Options
    units = hp.Int('units', min_value=32, max_value=512, step=32)
    dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 5e-3, 1e-3, 5e-4, 1e-4])

    # Set Layers
    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(units, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Tune the optimizer
    optimizer_choice = hp.Choice('optimizer', values=['adam', 'rmsprop', 'sgd'])
    if optimizer_choice == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_choice == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )
    return model

# Preprocess the dataset
preprocessed_dataset = pipeline.get_data()

# Split dataset into training and validation sets
train_size = int(0.7 * len(preprocessed_dataset))
val_size = len(preprocessed_dataset) - int(0.85 * len(preprocessed_dataset))

train_dataset = preprocessed_dataset.take(train_size)
val_dataset = preprocessed_dataset.skip(train_size)

tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    hyperband_iterations=5,
    project_name='densenet121_transfer_learning',
    directory='tests',
    overwrite=True
)

# Stop training early if there’s no improvement
stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)


# Search for the best hyperparameters
tuner.search(
    train_dataset,
    validation_data=val_dataset,
    epochs=15,
    callbacks=[stop_early],
)

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Get the best hyperparameters
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

# Print the best hyperparameters for the best model
print("Best Hyperparameters:")
print(f"Units: {best_hyperparameters.get('units')}")
print(f"Dropout Rate: {best_hyperparameters.get('dropout_rate')}")
print(f"Learning Rate: {best_hyperparameters.get('learning_rate')}")
print(f"Optimizer: {best_hyperparameters.get('optimizer')}")