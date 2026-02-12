import matplotlib.pyplot as plt
import numpy as np

from nn.losses import CategoricalCrossEntropyLoss
from nn.optimizers import Adam
from utils.data import generate_synthetic_data, train_val_split
from vision.models import TinyCNN
from vision.training import evaluate_cnn, predict_cnn, train_cnn

# Config
N_SAMPLES = 1000
IMAGE_SIZE = (32, 32)
N_CLASSES = 2
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
SAVE_DIR = '../results/'

print("Generating synthetic data...")
X, y = generate_synthetic_data(n_samples=N_SAMPLES, image_size=IMAGE_SIZE, n_classes=N_CLASSES)

print(f"Data shape: {X.shape}")
print(f"Labels shape: {y.shape}")

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    img = X[i].transpose(1, 2, 0)  # (H, W, C)
    ax.imshow(img)
    ax.set_title(f"Class {y[i]}")
    ax.axis('off')
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/synthetic_samples.png')
plt.show()

# Split data
X_train, y_train, X_val, y_val = train_val_split(X, y, val_split=0.2, shuffle=True, seed=42)

print(f"\nTraining samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")

print("\nCreating TinyCNN model...")
input_shape = (3, IMAGE_SIZE[0], IMAGE_SIZE[1])
model = TinyCNN(input_shape=input_shape, n_classes=N_CLASSES)

criterion = CategoricalCrossEntropyLoss()
optimizer = Adam(learning_rate=LEARNING_RATE)

print("\nTraining model...")
history = train_cnn(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=True
)

print("\nEvaluating model...")
val_loss, val_acc = evaluate_cnn(model, criterion, X_val, y_val, BATCH_SIZE)
print(f"Final validation loss: {val_loss:.4f}")
print(f"Final validation accuracy: {val_acc:.4f}")

# Plot train history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history['train_loss'], label='Train Loss')
ax1.plot(history['val_loss'], label='Val Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Loss')
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(history['train_acc'], label='Train Acc')
ax2.plot(history['val_acc'], label='Val Acc')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Accuracy')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/synthetic_training_history.png')
plt.show()

print("\n" + "="*70)
print("TEST COMPLETE!")
print("="*70)
print(f"Final validation accuracy: {val_acc:.4f}")
