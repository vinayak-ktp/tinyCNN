import time

import numpy as np

from utils.data import create_batches


def train_cnn(
    cnn,
    optimizer,
    criterion,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=10,
    batch_size=32,
    verbose=True
):
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(epochs):
        epoch_start = time.time()

        train_loss = 0
        train_correct = 0
        n_train_batches = 0

        for X_batch, y_batch in create_batches(X_train, y_train, batch_size, shuffle=True):
            cnn.forward(X_batch, training=True)
            loss = criterion.forward(cnn.output, y_batch)

            criterion.backward()
            cnn.backward(criterion.dinputs)

            for i, layer in enumerate(cnn.layers):
                if hasattr(layer, 'W'):
                    optimizer.update_params(layer)

            optimizer.post_update_params()

            train_loss += loss
            predictions = np.argmax(criterion.predictions, axis=1)
            train_correct += np.sum(predictions == y_batch)
            n_train_batches += 1

        train_loss /= n_train_batches
        train_acc = train_correct / len(y_train)

        val_loss, val_acc = evaluate_cnn(cnn, criterion, X_val, y_val, batch_size)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        epoch_time = time.time() - epoch_start

        if verbose:
            print(f"Epoch {epoch+1}/{epochs} - {epoch_time:.2f}s - "
                  f"loss: {train_loss:.4f} - acc: {train_acc:.4f} - "
                  f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")

    return history


def evaluate_cnn(cnn, criterion, X, y, batch_size=32):
    total_loss = 0
    correct = 0
    n_batches = 0

    for X_batch, y_batch in create_batches(X, y, batch_size, shuffle=False):
        cnn.forward(X_batch, training=False)
        loss = criterion.forward(cnn.output, y_batch)

        total_loss += loss
        predictions = np.argmax(criterion.predictions, axis=1)
        correct += np.sum(predictions == y_batch)
        n_batches += 1

    avg_loss = total_loss / n_batches
    accuracy = correct / len(y)

    return avg_loss, accuracy


def predict_cnn(cnn, X, batch_size=32):
    predictions = []

    for i in range(0, len(X), batch_size):
        X_batch = X[i:i+batch_size]
        cnn.forward(X_batch, training=False)

        batch_preds = np.argmax(cnn.output, axis=1)
        predictions.extend(batch_preds)

    return np.array(predictions)


def get_confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for true, pred in zip(y_true, y_pred):
        cm[true, pred] += 1

    return cm


def print_classification_report(y_true, y_pred, class_names):
    n_classes = len(class_names)
    cm = get_confusion_matrix(y_true, y_pred, n_classes)

    print("\nClassification Report:")
    print("-" * 60)
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 60)

    for i, class_name in enumerate(class_names):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"{class_name:<15} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")

    print("-" * 60)
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    print(f"{'Overall Accuracy':<15} {accuracy:.4f}")
    print("-" * 60)

    print("\nConfusion Matrix:")
    print(cm)
