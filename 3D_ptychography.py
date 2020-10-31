import numpy as np
import tensorflow as tf
from tensorflow import keras
from scipy import fftpack

def measure(object, probe):
    """Measure the modulus squared at distance z_d
    """
    source = probe * object
    x, y, z = np.shape(source)
    dft = fftpack.fftshift(np.fft.fftn(fftpack.ifftshift(source)))
    matrix = np.square(dft)
    return matrix[:, :, 0] / np.square(x * y * z)

def random_probe(shape):
    """Generate random probe with random ampltipude 
    and phase with given shape
    """
    return np.exp(2j*np.pi*np.random.rand(*shape))

def random_object(shape):
    """Generate random object with random ampltipude and phase with given shape
    """
    return np.random.rand(*shape) * np.exp(2j*np.pi*np.random.rand(*shape))

def generate_mnist_data(probe, n_train = 1000, depth = 10, n_test = 100):
    """Generate MNIST training data with 3D complex valued tensor as the object function, 
    real and imaginary part of the data are decoupled
    """
    mnist = keras.datasets.mnist
    (train_ims, y_train), (test_ims, y_test) = mnist.load_data()
    train_ims = np.reshape(train_ims, (np.shape(train_ims)[0] // depth, 28, 28, -1))
    test_ims = np.reshape(test_ims, (np.shape(test_ims)[0] // depth, 28, 28, -1))
    train_ims = (train_ims[:n_train] / 255.0).astype(np.complex64) + 1j * (train_ims[1:n_train + 1] / 255.0).astype(np.complex64)
    test_ims = (test_ims[:n_test] / 255.0).astype(np.complex64) + 1j * (test_ims[1:n_test + 1] / 255.0).astype(np.complex64)

    train_patterns = []
    test_patterns = []
    
    for i in range(n_train):
        patterns = measure(train_ims[i], probe)
        train_patterns.append(patterns)
    
    for i in range(n_test):
        patterns = measure(test_ims[i], probe)
        test_patterns.append(patterns)

    return (train_patterns, train_ims), (test_patterns, test_ims)

if __name__ == '__main__':
    shape = (28,28,10)
    probe = random_probe(shape)
    #object = random_object(shape)
    #print(measure(probe, object, shape[2]//2))
    
    n_train = 500
    n_test = 100
    depth = 10
    (tr_patterns, tr_images), (test_patterns, test_images) = generate_mnist_data(probe, n_train = 1000, depth = shape[2], n_test = 100)