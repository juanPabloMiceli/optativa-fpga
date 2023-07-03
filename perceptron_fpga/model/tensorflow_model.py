import sys
import tensorflow as tf
import numpy as np
from generate_weights_package import write_weights_to_file_from_list 
from sklearn import datasets
from sklearn.model_selection import train_test_split
import cv2

def main():
    if len(sys.argv) == 1:
        print("\nIf you want to train the model run \"python tensorflow_model.py train\"")
        return -1
    if sys.argv[1] == "train":
        train_model()
    elif sys.argv[1] == "test":
        test_model()
    elif sys.argv[1] == "train_test":
        train_model()
        test_model()
    elif sys.argv[1] == "write_vhdl":
        write_vhdl()
    elif sys.argv[1] == "predict":
        predict()
    else:
        print("Nothing done!")
    print("Done!")

def train_model():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = [cv2.resize(img, (8, 8), interpolation = cv2.INTER_AREA) for img in x_train]
    x_test = [cv2.resize(img, (8, 8), interpolation = cv2.INTER_AREA) for img in x_test]


    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(8,8)))
    model.add(tf.keras.layers.Dense(128, activation='relu', use_bias=False))
    model.add(tf.keras.layers.Dense(10, activation='softmax', use_bias=False))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=3)

    model.save('model/tensorflow.model')

def test_model():    
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = [cv2.resize(img, (8, 8), interpolation = cv2.INTER_AREA) for img in x_train]
    x_test = [cv2.resize(img, (8, 8), interpolation = cv2.INTER_AREA) for img in x_test]

    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    model = tf.keras.models.load_model('model/tensorflow.model')

    loss, acc = model.evaluate(x_test, y_test)

    print(f"{loss=}, {acc=}")

def write_vhdl():
    image_index = 0
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (_, _) = mnist.load_data()

    x_train = [cv2.resize(img, (8, 8), interpolation = cv2.INTER_AREA) for img in x_train]
    x_train = tf.keras.utils.normalize(x_train, axis=1)

    image_list = x_train[image_index].flatten().tolist()
    image_label = y_train[image_index]

    model = tf.keras.models.load_model('model/tensorflow.model')
    weights = []
    for layer in model.layers[1:]:
        weights += np.array(layer.weights).flatten().tolist()

    write_weights_to_file_from_list(weights, 'mnist_weights', './tests/mnist_model/tf_mnist_weights.vhdl', 'GetWeights')
    write_weights_to_file_from_list(image_list, f'mnist_inputs_{image_label}', f'./tests/mnist_model/tf_mnist_inputs_{image_label}.vhdl', 'GetInputs')

    prediction = np.argmax(model.predict(x_train[image_index:image_index+1]))
    print(f"{prediction=}: {image_label=}")

def predict():
    image_index = 0
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (_, _) = mnist.load_data()

    x_train = [cv2.resize(img, (8, 8), interpolation = cv2.INTER_AREA) for img in x_train]
    x_train = tf.keras.utils.normalize(x_train, axis=1)

    label = y_train[image_index]

    model = tf.keras.models.load_model('model/tensorflow.model')
    prediction = np.argmax(model.predict(x_train[image_index:image_index+1]))
    print(f"{prediction=}: {label=}")
    


# image_list = x_test[0]
# image_label = y_test[0]





if __name__ == "__main__":
    main()