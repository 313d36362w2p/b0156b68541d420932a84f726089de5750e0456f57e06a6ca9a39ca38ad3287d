import matplotlib.pyplot as plt

# Plot training & validation accuracy and loss
def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')

    plt.show()

def get_final_metrics(history):
    # Get final metrics
    final_training_accuracy = history.history['accuracy'][-1]
    final_training_loss = history.history['loss'][-1]
    final_validation_accuracy = history.history['val_accuracy'][-1]
    final_validation_loss = history.history['val_loss'][-1]

    return final_training_accuracy, final_training_loss, final_validation_accuracy, final_validation_loss