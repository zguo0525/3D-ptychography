import numpy as np
import tensorflow as tf
from tensorflow import keras
from scipy import fftpack
from matplotlib import pyplot as plt
from scipy.io import loadmat
from PIL import Image

# physical parameters for the optical system
wavelength = 632.8e-9
n0 = 1.0
n1 = 1.457
k = 2 * np.pi / wavelength
k0 = k
k1 = n1 * k0
# phase contrast of the parameters 
dn = 0.0565
pattern_depth = 575e-9

def rect(x):
    return np.where(np.abs(x)< 0.5, 1, 0)

def fft2d(x):
    """Define 2D FFT function
    """
    return fftpack.fftshift(np.fft.fft2(fftpack.ifftshift(x)))

def ifft2d(x):
    """Define 2D inv FFT function
    """
    return fftpack.fftshift(np.fft.ifft2(fftpack.ifftshift(x)))

def intensity(x):
    """Return the measured intensity of the field
    """
    return np.square(np.abs(x))

def measure(object, probe):
    """Measure the modulus squared at distance z_d
    """
    # element-wise multiplication
    source = np.multiply(object, probe)
    x, y, z = np.shape(source)
    # 3D FFT
    dft = fftpack.fftshift(np.fft.fftn(fftpack.ifftshift(source)))
    # return the intensity
    return intensity(dft[:, :, 0]) / (x * y * z)**2

def real_coordinate(Nx, Ny, Nz, dx, dy, dz):
    """Generate real coordinate for discrete computation
    """
    x, y, z = np.arange(-Nx/2, Nx/2) * dx, np.arange(-Ny/2, Ny/2) * dy, np.arange(-Nz/2, Nz/2) * dz
    return np.meshgrid(x, y, z)

def k_coordinate(Nx, Ny, Nz, dx, dy, dz):
    """Generate k coordinate for discrete computation
    """
    Lx, Ly, Lz = Nx * dx, Ny * dy, Nz * dz
    dkx, dky, dkz = 2 * np.pi/Lx, 2 * np.pi/Ly, 2 * np.pi/Lz
    kx, ky, kz = np.arange(-Nx/2, Nx/2) * dkx, np.arange(-Ny/2, Ny/2) * dky, np.arange(-Nz/2, Nz/2) * dkz
    return np.meshgrid(kx, ky, kz)

def phase_contrast():
    """Calculate the phase contrast
    """
    return k0 * pattern_depth * dn

def ic_layout_object(width, height):
    """Generate ic layout as the object for scattering
    """
    samples = []
    for i in range(0, 4):
        # read IC layout
        img = Image.open('object/layer' + str(i + 5) +'.png')
        # final shape of the image
        size = (2 * width, 2 * height)
        img = np.asarray(img.resize(size))/255 - 1
        # top left img
        samples.append(img[0: width, 
                        0: height])
        # top right img
        samples.append(img[width: 2 * width, 
                        0: height])
        # bottom left img
        samples.append(img[0: width, 
                        height: 2 * height])
        # bottom right img
        samples.append(img[width: 2 * width, 
                        height: 2 * height])
    #convert it to complex value    
    samples = [np.exp(1j * sample * phase_contrast()) for sample in samples]
    return np.transpose(samples)
    
    

if __name__ == '__main__':
    # number of pixels
    Nx, Ny, Nz = 1024, 1024, 16
    # physical spacing between pixels
    dx, dy, dz = 8e-6, 8e-6, 8e-6
    
    # generate coordinates
    X, Y, Z = real_coordinate(Nx, Ny, Nz, dx, dy, dz)
    Kx, Ky, Kz = k_coordinate(Nx, Ny, Nz, dx, dy, dz)
    
    # plane wave propagating in z direction
    plane_wave_z = np.exp(1j * k * Z)
    # generate spherical wave
    spherical_wave = np.exp(1j * k * np.sqrt(X**2 + Y**2 + Z**2))
    # generate object based on ic layout
    object = ic_layout_object(Nx, Ny)
    
    # measure the far field pattern using the object and incident wave
    measured_spherical = measure(object, spherical_wave)
    measured_plane = measure(object, plane_wave_z)
    
    print(measured_plane)