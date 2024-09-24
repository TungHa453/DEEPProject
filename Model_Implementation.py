import tensorflow as tf

# Train data path
norm_dir = 'train/NORMAL/'
pneu_dir = 'train/PNEUMONIA/'

# Load data
def load_data():
    norm_img = tf.io.gfile.glob(norm_dir + '*')
    pneu_img = tf.io.gfile.glob(pneu_dir + '*')
    images, labels = [], []

    # Norm img
    for img_path in norm_img:
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [224, 224])
        images.append(img)
        labels.append(0)

    # Pneu img
    for img_path in pneu_img:
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [224, 224])
        images.append(img)
        labels.append(1)

    return tf.convert_to_tensor(images) /255.0, tf.convert_to_tensor(labels)

# Load dataset
X, y = load_data()
y = tf.keras.utils.to_categorical(y, 2)

# CvT model
def cvt(input_shape=(224, 224, 3), num_classes=2):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

# Train model / save
model = cvt()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=5, batch_size=16)
model.save('model.h5')