import tensorflow as tf
import tensorflow_hub as hub
import os

# Constants
EFF_NET_URL = "https://tfhub.dev/google/efficientnet/b0/feature-vector/1"
image_size = (224, 224)
ckpt_dir = 'checkpoints/'

# Model Building
def build_model():
    inputs = tf.keras.Input(shape=(224, 224, 3))
    base_model = tf.keras.applications.EfficientNetB0(include_top=False, weights=None, input_tensor=inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs, outputs)

    # Load local checkpoint
    checkpoint_path = 'efficient_net_vehicle.ckpt'
    model.load_weights(checkpoint_path)

    return model

# Load Weights
def load_model_weights(model, checkpoint_dir):
    if os.path.exists(checkpoint_dir):
        model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
        print("Model weights loaded.")
    else:
        print("No checkpoint found.")
    return model

# Preprocess Image
def preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = tf.cast(img, tf.float32) / 255.0
    return tf.expand_dims(img, axis=0)

# Predict
def predict_fraud(img_path):
    model = build_model()
    model = load_model_weights(model, ckpt_dir)
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    print("Fraud likelihood:", prediction[0][0])

# Run
if __name__ == "__main__":
    test_image = "test_image.jpeg"
    if os.path.exists(test_image):
        predict_fraud(test_image)
    else:
        print("Test image not found.")
