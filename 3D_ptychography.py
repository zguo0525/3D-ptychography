import numpy as np
import tensorflow as tf
from tensorflow import keras
from scipy import fftpack

def fft2d(x):
    """Define 2D FFT function
    """
    return fftpack.fftshift(np.fft.fft2(fftpack.ifftshift(x)))

def ifft2d(x):
    """Define 2D inv FFT function
    """
    return fftpack.fftshift(np.fft.ifft2(fftpack.ifftshift(x)))

def wavenumber(wavelength, index):
    """Define wavenumber function generator
    """
    return 2 * np.pi * index/wavelength

def intensity(x):
    """Return the measured intensity of the field
    """
    return np.square(np.abs(x))

def optical_lens(shape, focal_length, u_inc):
    """Simulate optical focusing lens
    """
    x, y, z = np.shape(shape)
    X, Y = np.meshgrid(x, y)
    Rs = np.sqrt(X**2 + Y**2)
    k = wavenumber(632.8e-9, 1)
    return np.multiply(u_inc, np.exp(-1j * k/2/focal_length * Rs))
                            
def measure(object, probe):
    """Measure the modulus squared at distance z_d
    """
    source = probe * object
    x, y, z = np.shape(source)
    dft = fftpack.fftshift(np.fft.fftn(fftpack.ifftshift(source)))
    matrix = np.square(dft)
    return matrix[:, :, 0] / np.square(x * y * z)

def measure_2D(object, probe):
    """Using multi-slice method to simulate the forward model
    """
    source = probe * object
    x, y, z = np.shape(source)
    V = x * y * z
    k = wavenumber(632.8e-9, 1)
    total_field = 0
    for i in range(z):
        total_field += fft2d((source[:, :, i])) * bmp_operator(k, [x, y], z - i)
    return intensity(ifft2d(total_field) / V)

def bmp_operator(k, shape, layer_thickness):
    """Compute the BPM operator for forward model
    """
    kx = shape[0]
    ky = shape[1]
    Kx, Ky = np.meshgrid(kx, ky)
    K2 = Kx**2 + Ky**2
    return np.exp(-1j * (np.real(np.sqrt(k**2 - K2))) * layer_thickness)

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
    test = measure_2D(tr_images[0], probe)
    print(test)
