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
    X, Y = np.mgrid[:x, :y]
    Rs = np.sqrt(X**2 + Y**2)
    k = wavenumber(632.8e-9, 1)
    return np.multiply(u_inc, np.exp(-1j * k/2/focal_length * Rs))
                            
def measure(object, probe):
    """Measure the modulus squared at distance z_d
    """
    source = probe * object
    x, y, z = np.shape(source)
    dft = fftpack.fftshift(np.fft.fftn(fftpack.ifftshift(source)))
    return intensity(dft[:, :, 0] / (x * y * z))

def measure_2D(object, probe):
    """Using multi-slice method to simulate the forward model
    """
    source = probe * object
    x, y, z = np.shape(source)
    V = x * y * z
    k = wavenumber(632.8e-9, 1)
    total_field = 0
    for z_l in range(z):
        total_field += np.multiply(fft2d(source[:, :, z_l]), bpm_operator(k, [x, y], z - z_l))
        from matplotlib import pyplot as plt
        plt.imshow(np.abs(total_field))
        plt.colorbar()
        plt.figure()
    return intensity(ifft2d(total_field) / V)

def bpm_operator(k, shape, layer_thickness):
    """Compute the BPM operator for forward model
    """
    kx, ky = shape[0], shape[1]
    Kx, Ky = np.mgrid[:kx, :ky]
    K2 = Kx**2 + Ky**2
    return fftpack.fftshift(np.exp(-1j * (np.real(np.sqrt(k**2 - K2))) * layer_thickness))

def propagation_operator(k, shape, depth):
    """Compute the forward propagation operator for BPM
    """
    kx, ky = shape[0], shape[1]
    Kx, Ky = np.mgrid[:kx, :ky]
    K2 = Kx**2 + Ky**2
    return fftpack.fftshift(np.exp(-1j * (k - np.real(np.sqrt(k**2 - K2))) * depth))

def bpm_probe(shape, position):
    """Generate 3D gaussian probe in free space using beam propagation method
    with given shape and focusing at the given position
    """
    beam = np.zeros(shape, dtype=complex)
    beam[:, :, 0] = gaussian_probe(shape, position)[:, :, 0]
    k = wavenumber(632.8e-9, 1)
    for i in range(1, shape[2]):
        beam[:, :, i] = ifft2d(np.multiply(fft2d(beam[:, :, i-1]), 
                                           propagation_operator(k, shape, i)))
    return beam

def gaussian_probe(shape, position):
    """Generate 3D gaussian probe with ampltipude 
    and phase with given shape and center position with 
    amplitude equation in each coordinate via `E = e^{-((x-x_0)/w)^{2P}}`
    """
    mx, my, mz = shape[0]//2 - position[0], shape[1]//2 - position[1], shape[2]//2
    x, y, z = np.mgrid[:shape[0], :shape[1], :shape[2]]
    spot_size = 1 / np.sqrt(2 * np.log(2))
    amplitude = 1. / (2. * np.pi) * np.exp(-(((x - mx)/spot_size)**2 + ((y - my)/spot_size)**2+ ((z - mz)/spot_size)**2))
    phase = np.zeros(shape, dtype=complex)
    for i in range(shape[2]):
        phase[:, :, i] = np.exp(2j * np.pi/100 * i) 
    return np.multiply(amplitude, phase)

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
    x, y, z = shape[0], shape[1], shape[2]
    print(x, y, z)
    #probe = random_probe(shape)
    position = (1,1)
    probe = bpm_probe(shape,position)
    print(probe)
    #object = random_object(shape)
    #print(measure(probe, object, shape[2]//2))
    
    n_train = 500
    n_test = 100
    depth = 10
    (tr_patterns, tr_images), (test_patterns, test_images) = generate_mnist_data(probe, n_train = 1000, depth = shape[2], n_test = 100)
    test = measure_2D(tr_images[0], probe)
    from matplotlib import pyplot as plt
    plt.imshow(np.abs(test))
    plt.colorbar()
    plt.figure()
