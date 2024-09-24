import tensorflow as tf

# Load model
model = tf.keras.preprocessing.image.load_model('model.h5')

# Test func
def test(model, img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img_arr = tf.expand_dims(img, axis=0) / 255.0
    predict = model.predict(img_arr)
    return tf.argmax(predict, axis=1).numpy()[0]

# Test img dir
test_dir = "test"
test_img = tf.io.gfile.glob(test_dir + '*')

# Predict
for img_path in test_img:
    result = test(model, img_path)
    label = 'Normal' if result == 0 else 'Pneumonia'
    print(f'Image: {img_path}, Prediction: {label}')