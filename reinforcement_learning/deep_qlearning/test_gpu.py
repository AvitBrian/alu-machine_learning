import tensorflow as tf

# List physical devices
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs detected: {gpus}")
else:
    print("No GPUs detected. Check your CUDA and cuDNN installation.")