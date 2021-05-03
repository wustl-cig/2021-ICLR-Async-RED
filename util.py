import numpy as np
import scipy.io as sio
import scipy.misc as smisc
from scipy.optimize import fminbound


def to_rgb(img):
    img = np.atleast_3d(img)
    channels = img.shape[2]
    if channels < 3:
        img = np.tile(img, 3)

    img[np.isnan(img)] = 0
    img -= np.amin(img)
    img /= np.amax(img)
    img *= 255
    return img


def to_double(img):
    img = np.atleast_3d(img)
    channels = img.shape[2]
    if channels < 3:
        img = np.tile(img, 3)

    img[np.isnan(img)] = 0
    img -= np.amin(img)
    img /= np.amax(img)
    return img


def save_mat(img, path):
    sio.savemat(path, {'img': img})


def save_img(img, path):
    img = to_rgb(img)
    smisc.imsave(path, img.round().astype(np.uint8))


def addwgn(z, inputSnr):
    shape = z.shape
    z = z.flatten('F')
    noiseNorm = np.linalg.norm(z.flatten('F')) * 10 ** (-inputSnr / 20)
    xBool = np.isreal(z)
    real = True
    for e in np.nditer(xBool):
        if e == False:
            real = False
    if (real == True):
        noise = np.random.randn(z.size)
    else:
        noise = np.random.randn(z.size) + 1j * np.random.randn(z.size)

    noise = noise / np.linalg.norm(noise.flatten('F')) * noiseNorm
    y = z + noise
    return y.reshape(shape,order='F'), noise.reshape(shape,order='F')


def optimizeTau(x, algoHandle, taurange, maxfun=20):
    # maxfun ~ number of iterations for optimization

    evaluateSNR = lambda x, xhat: 20 * np.log10(
        np.linalg.norm(x.flatten('F')) / np.linalg.norm(x.flatten('F') - xhat.flatten('F')))
    fun = lambda tau: -evaluateSNR(x, algoHandle(tau)[0])
    tau = fminbound(fun, taurange[0], taurange[1], xtol=1e-6, maxfun=maxfun, disp=3)
    return tau


def powerIter(A, imgSize, iter=100, tol=1e-6, verbose=False):
    # compute singular value for A'*A
    # A should be a function (lambda:x)

    x = np.random.randn(imgSize[0], imgSize[1])
    x = x / np.linalg.norm(x.flatten('F'))

    lam = 1

    for i in range(iter):
        # apply Ax
        xnext = A(x)

        # xnext' * x / norm(x)^2
        lamNext = np.dot(xnext.flatten('F'), x.flatten('F')) / np.linalg.norm(x.flatten('F')) ** 2
        # only take the real part
        lamNext = lamNext.real

        # normalize xnext 
        xnext = xnext / np.linalg.norm(xnext.flatten('F'))

        # compute relative difference
        relDiff = np.abs(lamNext - lam) / np.abs(lam)

        x = xnext
        lam = lamNext

        # verbose
        if verbose:
            print('[{}/{}] lam = {}, relative Diff = {:0.4f}'.format(i, iter, lam, relDiff))

        # stopping criterion
        if relDiff < tol:
            break

    return lam


def u_mult(block, block_idx, num_blocks, block_size):
    # inject one block into one image
    patches = np.zeros([num_blocks, block_size, block_size])
    patches[block_idx, ...] = block
    x = putback_nonoverlap_patches(patches)
    return x


def u_tran(x, block_idx, num_blocks, block_size, extend_p=0, pad_mode='reflect'):
    padded_patches = extract_padded_patches(x, num_blocks, block_size, extend_p=extend_p, pad_mode=pad_mode)
    return padded_patches[block_idx, ...]


def extract_padded_patches(x, num_blocks, block_size, extend_p=0, pad_mode='reflect'):
    # extract padded patches from one image
    x_padded = np.pad(x, ((extend_p,), (extend_p,)), pad_mode)
    x_shape0, x_shape1 = (x_padded.shape[0], x_padded.shape[1])
    patch_size = block_size + 2 * extend_p
    h_idx_list = list(range(0, x_shape0 - patch_size, block_size)) + [x_shape0 - patch_size]
    w_idx_list = list(range(0, x_shape1 - patch_size, block_size)) + [x_shape1 - patch_size]
    padded_patches = np.zeros([num_blocks, patch_size, patch_size])
    count = 0
    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            padded_patches[count, ...] = x_padded[h_idx:h_idx + patch_size, w_idx:w_idx + patch_size]
            count = count + 1
    return padded_patches


def extract_nonoverlap_patches(x, num_blocks, block_size):
    # extract distinct patches from one image
    patches = np.zeros([num_blocks, block_size, block_size])
    nx, ny = x.shape
    count = 0
    for i in range(0, nx - block_size + 1, block_size):
        for j in range(0, ny - block_size + 1, block_size):
            patches[count, :] = x[i:i + block_size, j:j + block_size]
            count = count + 1
    return patches


def putback_padded_patches(patches, extend_p=0):
    # put back blocks into one image
    num_blocks, padded_size, _ = patches.shape
    block_size = padded_size-2*extend_p
    nx = ny = int(np.sqrt(num_blocks) * block_size)
    x = np.zeros([nx, ny])
    count = 0
    for i in range(0, nx - block_size + 1, block_size):
        for j in range(0, ny - block_size + 1, block_size):
            x[i:i + block_size, j:j + block_size] = patches[count]
            count = count + 1
    return x    


def putback_nonoverlap_patches(patches):
    # put back blocks into one image
    num_blocks, block_size, _ = patches.shape
    nx = ny = int(np.sqrt(num_blocks) * block_size)
    x = np.zeros([nx, ny])
    count = 0
    for i in range(0, nx - block_size + 1, block_size):
        for j in range(0, ny - block_size + 1, block_size):
            x[i:i + block_size, j:j + block_size] = patches[count]
            count = count + 1
    return x
