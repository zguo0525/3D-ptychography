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
        #print(dims)
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
    #print(x.size())
    return x[:, :, 0]**2 + x[:, :, 1]**2

def amplitude_mse(simulated, measured, mask=None):
    """loss function normalized amplitude mse for simulated and measured pattern
    """
    #simulated = simulated / t.sqrt(t.sum(simulated**2))
    #measured = measured / t.sqrt(t.sum(measured**2))
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
    #print(x, y, z, c)
    #print(ifftshift(source).size())
    dft = fftshift(t.fft(ifftshift(source), signal_ndim = 3, normalized = False))
    # return the intensity
    #print(dft.size())
    return cabssq(dft[:, :, 0, :]) / (x * y * z)**2

import matplotlib.pyplot as plt

def reconstruct(patterns, probes, scan_pos, resolution, scan_x, scan_y, layers, lr, iterations, 
                background = None, mask = None, optimizer = 'Adam',
                GPU = False, schedule = True, convergence_threshold = 1e-9, lr_threshold = 1e-18):
    """reconstruction using Wirtinger derivative and Adam optimizer
    """
    min_loss = None
    
    # generate inital random guess
    layer_thickness = np.asarray([0.016, 0.03, 0.04, 0.054, 0.027, 0.07, 0.06, 0.07, 0.06, 0.07, 0.063, 0.07, 0.063, 0.08, 0.063, 0.08, 0.068, 0.08, 0.8, 0.85, 0.8, 0.85]) * 1e-6
    wavelength = 0.14e-9
    k0 = 2 * np.pi / wavelength
    dn = np.zeros((512,512,22))
    dn[:, :, 1] = -6e-6
    dn[:, :, 2] = -7e-6
    dn[:, :, 3:4] = -4e-5
    dn[:, :, 5:22] = -2e-5
    samples = np.ones((resolution, resolution, layers))
    samples = np.exp(1j * samples * dn * layer_thickness * k0)
    #samples = np.ones((resolution, resolution, layers)) + 1j * np.ones((resolution, resolution, layers))
    im = complex_to_torch(samples)


    if GPU:
        im = im.to(device='cuda:0')
        if mask is not None:
            mask = mask.to(device='cuda:0')
            if background is not None:
                background = background.to(device='cuda:0')

    def closure():
        t_optimizer.zero_grad()
        
        # calculate four probes iteract with the object and sum them up
        simulated = measure_torch(im, probes_new[:, :, :, 0, :])[206:306, 206:306]
        for q in range(3):
            simulated += measure_torch(im, probes_new[:, :, :, q, :])[206:306, 206:306]

        if background is not None:
            simulated = simulated + background
        
        l = amplitude_mse(simulated, pattern, mask = mask)
        
        #plt.rcParams['figure.figsize'] = [10, 7]
        #plt.imshow(simulated.detach().cpu().numpy())
        #plt.colorbar(shrink=0.9)
        #plt.show()
        #plt.imshow(pattern.detach().cpu().numpy())
        #plt.colorbar(shrink=0.9)
        #plt.show()
        #print(l)
        
        l.backward()
        #for m in range(22):
        #    if m != layer: 
        #        im.grad.data[:,:,m,:].fill_(0)
        #print(im.grad.data)
        return l
    

    loss = []
    for k in range(iterations):

        # random order to update
        scan_list = np.arange(0, 400, 1)
        np.random.shuffle(scan_list)
        layers = np.arange(0,22,1)
        np.random.shuffle(layers)
       
        for i in scan_list:
            
            #with t.no_grad():
            #    im.detach().cpu()
            #    print(im)
            #    im = np.where(im <= 1, im, 1)
            #    im = im.to(device='cuda:0')

            temp = int(i)
            pattern = patterns[temp]
            pattern = pattern.to(device='cuda:0')

            probes_new = np.zeros((512, 512, 22, 4)).astype(complex)
            for p in range(4):
                probe = probes[:, :, p]
                delta = scan_pos[temp, :]
                #plt.imshow(np.real(probes[:, :, p]).get())
                #plt.show()
                probe = FourierShift2D(probe, delta)

                probe_3d = np.ones((512, 512, 22)).astype(complex)
                probe_3d[:, :, 0] = probe
                #print(probe)
                for i in range(1, 22):
                    propagation_operator_z = propagation_operator(k0, 7.0446e-9, 7.0446e-9, layer_thickness[i-1])
                    probe_3d[:, :, i] = ifft2d(np.multiply(fft2d(probe_3d[:, :, i-1]), propagation_operator_z))
                    
                probes_new[:, :, :, p] = probe_3d
            
            probes_new = complex_to_torch(probes_new).to(dtype=t.float32)
            probes_new = probes_new.to(device='cuda:0')

            #for layer in layers:
            im = t.nn.Parameter(im)
            t_optimizer = t.optim.Adam([im], lr=lr)
            #scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(t_optimizer)

            tensor = t_optimizer.step(closure)
            tensor = tensor.detach().cpu().numpy()
            loss.append(tensor[()])
            #if schedule:
            #    scheduler.step(loss[-1])
            lr = t_optimizer.param_groups[0]['lr']
            #if lr < lr_threshold or loss[-1] < convergence_threshold:
            #    break


    scaling_factor = 1

    # We scale the image by the scaling factor to match the given probe
    # intensity
    this_result = im.detach() * scaling_factor
    
    if GPU:
        this_result = this_result.to(device='cpu')
        # must be done to reset for the next attempt
        pattern = pattern.to(device='cpu')

    if min_loss is None or loss[-1] < min_loss[-1]:
        min_loss = loss
        result = this_result

    return result, min_loss

if __name__ == '__main__':
    
    scan_pos = loadmat('scan' + '.mat')['scan_pos']
    
    print('Enter starting layer:')
    starting_layer_number = int(input()) + 1
    print('Enter ending layer:')
    ending_layer_numer = int(input()) + 1
    
    for a in tqdm(range(starting_layer_number, ending_layer_numer)):
    
        measured_numpy = np.load('diffraction_patterns/measured_patterns_layout_' +str(a) + '.npy')

        result, min_loss = reconstruct(patterns = t.tensor(measured_numpy), 
                                        probes = np.load('probes.npy'),  
                                        scan_pos = scan_pos,
                                        resolution = 512,
                                        scan_x = 20,
                                        scan_y = 20,                           
                                        layers = 22,
                                        lr = 4e-4,
                                        iterations = 1,
                                        GPU = True)


        np.save('approximates/result' + str(a), result)
        np.save('approximates/loss' + str(a), min_loss)
