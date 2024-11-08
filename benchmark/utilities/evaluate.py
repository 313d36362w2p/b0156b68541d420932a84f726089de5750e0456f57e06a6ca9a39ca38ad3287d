import numpy as np
from keras.saving import load_model

def far_ffr_test(test_dataset, model_name):
    # Load your model
    model = load_model(model_name, custom_objects=None, compile=True, safe_mode=True)

    # Load test data
    # Assuming test_dataset is a tf.data.Dataset yielding (image, label) pairs
    X_test = []
    y_test = []

    for images, labels in test_dataset:
        X_test.append(images.numpy())
        y_test.append(labels.numpy())

    # Convert to numpy arrays
    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    # Predict probabilities
    y_pred_prob = model.predict(X_test)

    # Convert probabilities to binary outcomes
    threshold = 0.5
    y_pred = (y_pred_prob > threshold).astype(int)

    # Calculate TP, FP, TN, FN manually
    tp = np.sum((y_test == 1) & (y_pred == 1))
    fp = np.sum((y_test == 0) & (y_pred == 1))
    tn = np.sum((y_test == 0) & (y_pred == 0))
    fn = np.sum((y_test == 1) & (y_pred == 0))

    # Calculate FAR and FRR
    FAR = fp / (fp + tn) if (fp + tn) > 0 else 0
    FRR = fn / (fn + tp) if (fn + tp) > 0 else 0

    print(f"False Acceptance Rate (FAR): {FAR:.4f}")
    print(f"False Rejection Rate (FRR): {FRR:.4f}")

