import cupy as np
#import numpy as np
from scipy import fftpack
from matplotlib import pyplot as plt
from scipy.io import loadmat
from PIL import Image

# physical parameters for the optical system
wavelength = 0.14e-9
n0 = 1.0
n1 = 1.0
k = 2 * np.pi / wavelength
k0 = k
k1 = n1 * k0


def rect(x):
    """Define rect function
    """
    return np.where(np.abs(x)< 0.5, 1, 0)

def fft2d(x):
    """Define 2D FFT function
    """
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))

def ifft2d(x):
    """Define 2D inv FFT function
    """
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x)))

def intensity(x):
    """Return the measured intensity of the field
    """
    return np.square(np.abs(x))

def measure(object, probe):
    """Measure the modulus squared at distance z_d
    """
    # element-wise multiplication
    source = np.multiply(object, probe)
    x, y, z = 512, 512, 22
    # 3D FFT
    dft = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(source)))
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

def rgb2gray(rgb):
    """Convert RGB image to greyscale
    """
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def propagation_operator(k, dx, dy, dz):
    """propagation operator in z
    """
    Nx, Ny = 512, 512
    Lx, Ly = Nx * dx, Ny * dy
    dkx, dky = 2 * np.pi /Lx, 2 * np.pi /Ly
    kx, ky = np.arange(-Nx/2, Nx/2) * dkx, np.arange(-Ny/2, Ny/2) * dky
    Kx, Ky = np.meshgrid(kx, ky)
    return np.fft.fftshift(np.exp(-1j * (k - np.real(np.sqrt(k**2 - Kx**2 - Ky**2))) * dz))

def bpm_2d_probe_to_3d(probe, depth, propagation_operator_z):
    """propagation the probe in z
    """
    #probe_3d = [probe]
    probe_3d = np.ones((512, 512, depth)).astype(complex)
    probe_3d[:, :, 0] = probe
    for i in range(1, depth):
        probe_3d[:, :, i] = ifft2d(np.multiply(fft2d(probe_3d[:, :, i-1]), propagation_operator_z))
    return probe_3d

def FourierShift2D(probe, delta):
    N, M = 512, 512
    f_probe = np.fft.fft2(probe)

    x_shift = np.exp(-1j * 2 * np.pi * delta[0] * np.transpose(np.concatenate((np.arange(0, N//2), np.arange(-N//2, 0, 1))))/ N)
    y_shift = np.exp(-1j * 2 * np.pi * delta[1] * np.concatenate((np.arange(0, M//2), np.arange(-M//2, 0, 1)))/ M)
    
    if np.mod(N, 2) == 0:
        x_shift[N//2] = np.real(x_shift[N//2])
    if np.mod(M, 2) == 0:
        y_shift[M//2] = np.real(y_shift[M//2])
    
    tot_shift = np.outer(x_shift, y_shift)
    #print(np.shape(tot_shift))
    Y = np.multiply(f_probe, tot_shift)
    y = np.fft.ifft2(Y)

    return y

import torch as t
from tqdm import tqdm


def complex_to_torch(x):
    """Maps a complex numpy array to a torch tensor

    Pytorch uses tensors with a final dimension of 2 to represent
    complex numbers. This maps a complex type numpy array to a torch
    tensor following this convention
    """
    return t.from_numpy(np.asnumpy(np.stack((np.real(x),np.imag(x)),axis=-1)))

def cmult(a,b):
    """Returns the complex product of two torch tensors

    Pytorch uses tensors with a final dimension of 2 to represent
    complex numbers. This calculates the elementwise product
    of two torch tensors following that standard.
    """
    real = t.mul(a[:, :, :,0], b[:, :, :,0]) - t.mul(a[:, :, :,1], b[:, :, :,1])
    imag = t.mul(a[:, :, :,0], b[:, :, :,1]) + t.mul(a[:, :, :,1], b[:, :, :,0])
    return t.stack((real,imag),dim=-1)

def fftshift(array,dims=None):
    """Drop-in torch replacement for scipy.fftpack.fftshift

    This maps a tensor, assumed to be the output of a fast Fourier
    transform, into a tensor whose zero-frequency element is at the
    center of the tensor instead of the start. It will by default shift
    every dimension in the tensor but the last (which is assumed to
    represent the complex number and be of dimension 2), but can shift
    any arbitrary set of dimensions.

    Parameters
    ----------
    array : torch.Tensor
        An array of data to be fftshifted
    dims : iterable
        A list of all dimensions to shift

    Returns
    -------
    torch.Tensor
        The fftshifted tensor
    """
    if dims is None:
        dims=list(range(array.dim()))[:-1]
    for dim in dims:
        length = array.size()[dim]
        cut_to = (length + 1) // 2
        cut_len = length - cut_to
        array = t.cat((array.narrow(dim,cut_to,cut_len),
                       array.narrow(dim,0,cut_to)), dim)
    return array

def ifftshift(array,dims=None):
    """Drop-in torch replacement for scipy.fftpack.iftshift

    This maps a tensor, assumed to be the shifted output of a fast
    Fourier transform, into a tensor whose zero-frequency element is
    back at the start of the tensor instead of the center. It is the
    inverse of the fftshift operator. It will by default shift
    every dimension in the tensor but the last (which is assumed to
    represent the complex number and be of dimension 2), but can shift
    any arbitrary set of dimensions.

    Parameters
    ----------
    array : torch.Tensor
        An array of data to be ifftshifted
    dims : list(int)
        A list of all dimensions to shift

    Returns
    -------
    torch.Tensor 
        The ifftshifted tensor
    """
    if dims is None:
        dims=list(range(array.dim()))[:-1]
    for dim in dims:
        length = array.size()[dim]
        cut_to = length // 2
        cut_len = length - cut_to

        array = t.cat((array.narrow(dim,cut_to,cut_len),
                       array.narrow(dim,0,cut_to)), dim)
    return array

def cabssq(x):
    """Returns the square of the absolute value of a complex torch tensor

    Pytorch uses tensors with a final dimension of 2 to represent
    complex numbers. This calculates the elementwise absolute value
    squared of any toch tensor following that standard.

    Parameters
    ----------
    x : torch.Tensor
        An input tensor

    Returns
    -------
    torch.Tensor
        A tensor storing the elementwise absolute value squared

    """
    return x[...,0]**2 + x[...,1]**2

def amplitude_mse(simulated, measured, mask=None):
    """loss function normalized amplitude mse for simulated and measured pattern
    """
    if mask is None:
        return t.sum((t.sqrt(simulated) - t.sqrt(measured))**2) / t.sum(measured) #t.sum(simulated) #
    else:
        masked_measured = measured.masked_select(mask)
        return t.sum((t.sqrt(simulated.masked_select(mask)) - t.sqrt(masked_measured))**2)

def measure_torch(object, probe):
    """Measure the modulus squared at distance z_d using torch (Good)
    """
    # element-wise multiplication
    #print(object.size())
    source = cmult(object, probe)
    #print(source.size())
    x, y, z, c = [*source.size()]
    # 3D FFT
    dft = fftshift(t.fft(ifftshift(source), signal_ndim = 3, normalized = False))
    # return the intensity
    return cabssq(dft[:, :, 0, :]) / (x * y * z)**2

if __name__ == '__main__':
    
    import sys
    
    scan_pos = loadmat('scan' + '.mat')['scan_pos']
    layer_thickness = np.asarray([0.016, 0.03, 0.04, 0.054, 0.027, 0.07, 0.06, 0.07, 0.06, 0.07, 0.063, 0.07, 0.063, 0.08, 0.063, 0.08, 0.068, 0.08, 0.8, 0.85, 0.8, 0.85]) * 1e-6

    dn = np.zeros((512,512,22))
    dn[:, :, 1] = -6e-6
    dn[:, :, 2] = -7e-6
    dn[:, :, 3:4] = -4e-5
    dn[:, :, 5:22] = -2e-5

    import h5py
    from tqdm import tqdm
    print('Enter starting layer:')
    starting_layer_number = int(input()) + 1
    print('Enter ending layer:')
    ending_layer_numer = int(input()) + 1

    # measured sets of diffraction patterns for different layers
    #measured_layers = []

    for l in tqdm(range(starting_layer_number, ending_layer_numer)):
        
        # defining the coordinates
        Nx, Ny, Nz = 512, 512, 22
        
        # physical spacing between pixels
        dx, dy, dz = 1.409e-8, 1.409e-8, 2e-7

        # generate coordinates
        X, Y, Z = real_coordinate(Nx, Ny, Nz, dx, dy, dz)
        Kx, Ky, Kz = k_coordinate(Nx, Ny, Nz, dx, dy, dz)

        probes = np.load('probes.npy')

        # get object with shape (512, 512, 22)
        filename = 'layers/layers_' + str(l) + '.mat'
        with h5py.File(filename, "r") as f:
            a_group_key = list(f.keys())[0]
            object = list(f[a_group_key])
            object = np.asarray(object)
            object = np.transpose(object)
            
        object = np.exp(1j * object * dn * layer_thickness * k0)
        object = complex_to_torch(object)
        object = object.to(device='cuda:0')

        # create list for summing the four diffractions
        #measured_sum = np.zeros((400, 100, 100))
        #measured_sum = np.ones((400, 512, 512))
        measured_sum = []
        # create list for the four probes measurements
        for scan in range(400):
            measured_list = []
            for p in range(4):        
                probe = probes[:, :, p]
                #probe_shape = np.shape(probe)

                delta = scan_pos[scan, :]
                probe = FourierShift2D(probe, delta)
                
                probe_3d = np.ones((512, 512, 22)).astype(complex)
                probe_3d[:, :, 0] = probe
                for i in range(1, 22):
                    propagation_operator_z = propagation_operator(k, 7.0446e-9, 7.0446e-9, layer_thickness[i-1])
                    probe_3d[:, :, i] = ifft2d(np.multiply(fft2d(probe_3d[:, :, i-1]), propagation_operator_z))

                probe = complex_to_torch(probe_3d)

                probe = probe.to(device='cuda:0')

                measured = measure_torch(object, probe)[206:306, 206:306]
                measured = measured.cpu().detach().numpy()
                
                measured_list.append(measured)
            
            #print(measured_list[3])
            #print(scan)
            measured_sum.append(measured_list[0] + measured_list[1] + measured_list[2] + measured_list[3])
        
        measured_sum = measured_sum

        np.save('diffraction_patterns/measured_patterns_layout_' + str(l), measured_sum)
