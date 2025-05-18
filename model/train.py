# === 1. Setup ===
print("Importing libraries...")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import MobileNetV2 # Using MobileNetV2

import pathlib
import zipfile
import os
import numpy as np
import matplotlib.pyplot as plt # For plotting training history
import json # For saving class names

# === 2. Dataset Setup ===
zip_file_path = '/content/archive (1).zip' # MAKE SURE THIS PATH IS CORRECT
extract_path = '/content'
dataset_folder_name = 'Dataset' # This should be the name of the folder inside the zip

if os.path.exists(zip_file_path):
    print(f"Extracting {zip_file_path}...")
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Extraction complete.")
else:
    print(f"Warning: Zip file not found at {zip_file_path}. Assuming dataset is already extracted.")

data_dir = pathlib.Path(extract_path) / dataset_folder_name
if not data_dir.exists():
    # Attempt to find the dataset folder if the name is slightly different or nested
    # This is a common issue if the zip extracts into a parent folder first
    extracted_contents = [d for d in pathlib.Path(extract_path).iterdir() if d.is_dir()]
    if len(extracted_contents) == 1 and dataset_folder_name in str(extracted_contents[0]):
        print(f"Dataset folder name might be nested. Trying: {extracted_contents[0]}")
        data_dir = extracted_contents[0]
    elif len(extracted_contents) > 0:
        print(f"Could not find exact folder '{dataset_folder_name}'. Available top-level extracted folders: {[d.name for d in extracted_contents]}")
        print("Please ensure 'dataset_folder_name' matches the main image dataset folder name inside the zip.")

if not data_dir.exists():
    raise FileNotFoundError(f"Dataset directory not found after extraction attempts: {data_dir}. Please check the zip file structure and 'dataset_folder_name'.")
else:
    print(f"Using dataset directory: {data_dir}")


# === 3. Data Preparation ===
print("\nPreparing data loaders...")
batch_size = 32
img_height = 160 # MobileNetV2 default input sizes are often 224, 192, 160, 128
img_width = 160
validation_split = 0.2

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=validation_split,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=validation_split,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"\nFound {num_classes} classes: {class_names}")

# AUTOTUNE for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# === 4. Build the Model using MobileNetV2 ===
print("\nBuilding the model with MobileNetV2 base...")

# Preprocessing layer specific to MobileNetV2
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# Load MobileNetV2 base model
base_model = MobileNetV2(
    input_shape=(img_height, img_width, 3),
    include_top=False,  # Exclude the ImageNet classifier
    weights='imagenet'  # Load pre-trained ImageNet weights
)
base_model.trainable = False # Freeze the base model layers

# Create the new model on top
model = Sequential([
    # Input layer (optional, but good for clarity with input_shape)
    layers.Input(shape=(img_height, img_width, 3), name='input_layer'),

    # Data Augmentation (can be useful, kept simple)
    layers.RandomFlip("horizontal", name='random_flip'),
    layers.RandomRotation(0.1, name='random_rotation'),
    # layers.RandomZoom(0.1, name='random_zoom'), # Optional: add more augmentation

    # MobileNetV2 preprocessing
    layers.Lambda(preprocess_input, name='mobilenet_preprocess'),

    # The pre-trained MobileNetV2 base
    base_model,

    # Classification head
    layers.GlobalAveragePooling2D(name='global_avg_pool'),
    layers.Dense(128, activation='relu', name='dense_intermediate'),
    layers.Dropout(0.3, name='dropout_head'), # Regularization
    layers.Dense(num_classes, activation='softmax', name='output_classifier') # Softmax for multi-class
], name="mobilenetv2_transfer_learning_model")


# === 5. Compile the Model ===
print("\nCompiling the model...")
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001), # Adam is a good default
    loss=keras.losses.SparseCategoricalCrossentropy(), # Use this if labels are integers
    metrics=['accuracy']
)

model.summary() # Display the model's architecture


# === 6. Train the Model ===
print("\nStarting training...")
epochs = 15 # Increased epochs for potentially better convergence
# Early stopping can be useful to prevent overfitting
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
    # callbacks=[early_stopping] # Uncomment to use early stopping
)


# === 7. Evaluate the Model ===
print("\nEvaluating model on validation data...")
loss, accuracy = model.evaluate(val_ds, verbose=1)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

# === 8. Visualize Training History (Basic) ===
print("\nPlotting basic training history...")
if 'accuracy' in history.history and 'val_accuracy' in history.history:
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss_values = history.history['loss']
    val_loss_values = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss_values, label='Training Loss')
    plt.plot(epochs_range, val_loss_values, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.show()
else:
    print("Could not plot history: 'accuracy' or 'val_accuracy' not found in history keys.")
    print("History keys available:", history.history.keys())


# === 9. Convert and Save the Model ===
print("\nConverting and saving the model to TensorFlow Lite format...")

# Define paths for saving
tflite_model_dir = pathlib.Path("/content/tflite_model")
tflite_model_dir.mkdir(parents=True, exist_ok=True) # Create directory if it doesn't exist

tflite_model_path_bin = tflite_model_dir / "model.bin" # Standard extension for TFLite is .tflite
model_json_path = tflite_model_dir / "model.json" # For TFJS Layers format
labels_json_path = tflite_model_dir / "class_names.json" # For class names

# --- 9.1. Convert to TensorFlow Lite (.tflite or .bin) ---
# This is the primary TFLite format, often used with .tflite extension
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT] # Apply default optimizations
tflite_quantized_model = converter.convert()

with open(tflite_model_path_bin, 'wb') as f:
    f.write(tflite_quantized_model)
print(f"TFLite model (quantized) saved to: {tflite_model_path_bin}")
print(f"Size of TFLite model: {os.path.getsize(tflite_model_path_bin) / (1024):.2f} KB")


# --- 9.2. Save class names ---
with open(labels_json_path, 'w') as f:
    json.dump(class_names, f)
print(f"Class names saved to: {labels_json_path}")


# --- 9.3. (Optional) Save for TensorFlow.js (Layers Format: model.json + weights.bin) ---
# This part is for if you intend to use the model directly with TensorFlow.js in a LayersModel format.
# The previous TFLite conversion is generally for mobile/embedded or using TFJS TFLite runtime.
# For direct TFJS use without TFLite runtime, this is often preferred.

# tfjs_target_dir = pathlib.Path("/content/tfjs_model")
# tfjs_target_dir.mkdir(parents=True, exist_ok=True)
# tfjs_model_json_path = tfjs_target_dir / "model.json" # Will create model.json and one or more .bin files for weights

# try:
#     import tensorflowjs as tfjs # Make sure tensorflowjs is installed: pip install tensorflowjs
#     tfjs.converters.save_keras_model(model, tfjs_target_dir)
#     print(f"TensorFlow.js Layers format model saved to directory: {tfjs_target_dir}")
#     # List files in the tfjs_model directory
#     print("Files in TFJS model directory:")
#     for item in tfjs_target_dir.iterdir():
#         print(f"  - {item.name} ({os.path.getsize(item) / 1024:.2f} KB)")
# except ImportError:
#     print("\nSkipping TensorFlow.js Layers format conversion: 'tensorflowjs' library not found.")
#     print("To save in TFJS Layers format, install it: pip install tensorflowjs")
# except Exception as e:
#     print(f"\nError during TensorFlow.js Layers format conversion: {e}")


print("\nScript finished. Model training, evaluation, and TFLite conversion complete.")

# To download the files from Colab:
print(f"\nTo download the TFLite model and class names from Colab:")
print(f"1. Click on the 'Files' icon (folder icon) in the left sidebar of Colab.")
print(f"2. Navigate to the '{tflite_model_dir.name}' directory inside '/content'.")
print(f"3. Right-click on 'model.bin' and 'class_names.json' and select 'Download'.")
# if 'tfjs_target_dir' in locals(): # Check if tfjs_target_dir was defined
# print(f"4. If you also converted to TFJS Layers format, navigate to '{tfjs_target_dir.name}' and download its contents.")
