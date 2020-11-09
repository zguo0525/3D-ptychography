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
k = 2 * np.pi / wavelength * n0
k0 = k
k1 = n1 * k0
# phase contrast of the parameters 
dn = 0.0565
pattern_depth = 575e-9

def rect(x):
    """Define rect function
    """
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

def ic_layout_object(width, height, depth):
    """Generate ic layout as the object for scattering
    """
    samples = []
    for i in range(0, depth):
        # read IC layout
        img = Image.open('object/layer' + str(i + 5) +'.png')
        # final shape of the image
        size = (width, height)
        # normalize the image
        img = np.asarray(img.resize(size))/255 - 1
        samples.append(img)

    # convert (depth, width, height) to (width, height, depth) 
    samples = np.transpose(samples)
    #convert it to complex value 
    samples = np.exp(1j * samples * phase_contrast())
    return samples

def propagation_operator(k, probe, dx, dy, dz):
    """BPM propagation operator in z
    """
    Nx, Ny = np.shape(probe)
    Lx, Ly = Nx * dx, Ny * dy
    dkx, dky = 2 * np.pi /Lx, 2 * np.pi /Ly
    kx, ky = np.arange(-Nx/2, Nx/2) * dkx, np.arange(-Ny/2, Ny/2) * dky
    Kx, Ky = np.meshgrid(kx, ky)
    return fftpack.fftshift(np.exp(-1j * (k - np.real(np.sqrt(k**2 - Kx**2 - Ky**2))) * dz))

def bpm_2d_probe_to_3d(probe, depth, propagation_operator_z):
    """BPM propagation the probe in z
    """
    probe_3d = [probe]
    for i in range(depth - 1):
        probe_3d.append(ifft2d(np.multiply(fft2d(probe), propagation_operator_z)))
    return np.transpose(probe_3d)

print(phase_contrast())
    
    
if __name__ == '__main__':
    # number of pixels
    Nx, Ny, Nz = 1024, 1024, 4
    # physical spacing between pixels
    dx, dy, dz = 8e-6, 8e-6, pattern_depth
    
    # generate coordinates
    X, Y, Z = real_coordinate(Nx, Ny, Nz, dx, dy, dz)
    Kx, Ky, Kz = k_coordinate(Nx, Ny, Nz, dx, dy, dz)
    
    # generate plane wave probe in z direction with shape (Nx, Ny, Nz)
    plane_wave_z = np.exp(1j * k * Z)
    # generate spherical wave probe with shape (Nx, Ny, Nz)
    spherical_wave = np.exp(-1j * k * np.sqrt(X**2 + Y**2 + Z**2))
    # get Yudong's probe with shape (256, 256)
    yudong_probe = loadmat('yudong_illum_sample' + '.mat')['illum']
    yudong_shape = np.shape(yudong_probe)
    # get bpm operator 
    propagation_operator_z = propagation_operator(k, yudong_probe, dx, dy, dz)
    # get Yudong's probe in 3D with shape (256, 256, 16)
    yudong_probe = bpm_2d_probe_to_3d(yudong_probe, Nz, propagation_operator_z)
    
    # generate object based on ic layout
    object = ic_layout_object(Nx, Ny, Nz)
    
    # measure the far field pattern using the object and incident wave
    measured_spherical = measure(object, spherical_wave)
    measured_plane = measure(object, plane_wave_z)
    
    # scan yudong_probe to the object
    scan_x = 20
    scan_y = 20
    measured_yudong = []
    for i in range(scan_x):
        for j in range(scan_y):
            measured = measure(object[(Nx - yudong_shape[0])//(scan_x-1) * i : (Nx - yudong_shape[0])//(scan_x-1) * i + yudong_shape[0], 
                                    (Ny - yudong_shape[1])//(scan_y-1) * j : (Ny - yudong_shape[1])//(scan_y-1) * j + yudong_shape[1], 
                                    :], yudong_probe)
            measured_yudong.append(measured)
            print(i, j)
            
    print(np.shape(measured_yudong))