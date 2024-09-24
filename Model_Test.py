import tensorflow as tf
from tensorflow.keras import models

# Load model
model = tf.keras.models.load_model('model.keras')

# Test func
def test(model, img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img_arr = tf.expand_dims(img, axis=0) / 255.0
    predict = model.predict(img_arr)
    return tf.argmax(predict, axis=1).numpy()[0]

# Test img dir
test_dir = "test/NORMAL/"                   # changes dir as necessary
test_img = tf.io.gfile.glob(test_dir + '*')

# Predict
for img_path in test_img:
    result = test(model, img_path)
    label = 'Normal' if result == 0 else 'Pneumonia'
    print(f'Image: {img_path}, Prediction: {label}')