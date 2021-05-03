import matplotlib.pyplot as plt
from DataFidelities.BlockRandomClass import BlockRandomClass
from Regularizers.DnCNNstarClass import DnCNNstarClass
from asyncIterAlgs import asyncRED_solver
import util
import time
import scipy.io as sio
import numpy as np
import os
import shutil

####################################################
####              HYPER-PARAMETERS               ###
####################################################
# optimal tau values for residual DnCNN with Random Matrix with 30 dB input SNR noise
DnCNN_taus = [0.800695550810627,1.499111076510110,1.08665481586030,
            0.931244454946573,0.885731609281081,1.31972073454657]

np.random.seed(1)

####################################################
####              DATA PREPARATION               ###
####################################################

# here we load all 10 test images
data_name = 'set12_240'
data = sio.loadmat('data/{}.mat'.format(data_name), squeeze_me=True)
imgs = np.squeeze(data['imgs'])

# prepare for the data info
recon_shape = np.array(imgs[..., 0].shape)
num_blocks = 9
block_size = 80

downsample_rate = 0.8367  # downsample_rate^2 = 0.7
noise_level = 30  # two noise levels {30,40} are available for the simulation.

# save note
save_note = 'dr={}_noise={}'.format(downsample_rate, noise_level)

# generate random measurement matrix used for all 10 images.
print()
print('Generating random measurement matrix . . .')
try:
    data = sio.loadmat('matrix/{}_{}_{}.mat'.format(num_blocks, block_size, downsample_rate))
    A_blocks = data['matrix']
except:
    A_blocks = BlockRandomClass.generate_A(num_blocks, block_size, downsample_rate=downsample_rate)
    sio.savemat('matrix/{}_{}_{}.mat'.format(num_blocks, block_size, downsample_rate), {'matrix': A_blocks})
print('. . . Done')
print()

####################################################
####                Key Arguments                ###
####################################################

data_kargs = {
    'nx': recon_shape[0],
    'ny': recon_shape[1],
    'ic': 1,
    'oc': 1
}

# indicate the GPU index if available. If not, just leave it
# unchanged. Note that GPU significantly accelerates the
# computation of DnCNN.
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

alg_args = {
    'cpu_offset': 0,
    'processes_per_gpu': 4,
    'pad': 10,
    'minibatch_size': 'full',
    'num_iter': 1000,
    'is_noise': True,
    'mode': 'uniform',
    'logging': True,
    'verbose': True,
    'is_save': False,
    'save_every': 90,
    'is_conv': False,
    'conv_every': 4,
    'save_note': save_note,
}

####################################################
####            NETWORK INITIALIZATION           ###
####################################################

# -- Network Setup --#
# select the DnCNNstar model
model_path = 'models/DnCNN_lipschitz=1_sigma=5/model'

####################################################
####                LOOP IMAGES                  ###
####################################################

numImgs = imgs.shape[-1]
asyncRED_output = {}

# number of processes
pnum = 4

# select which image you want to reconstruct. By default we use the sixth image.
startIndex = 0
endIndex = 1

for i in range(startIndex, endIndex):
    # extract truth
    x = imgs[..., i]
    alg_args['xtrue'] = x
    recon_shape = np.array(x.shape)

    # reform x into blocks
    x_blocks = util.extract_nonoverlap_patches(x, num_blocks, block_size)

    # measure
    y_blocks = BlockRandomClass.generate_y(x_blocks, A_blocks, noise_level=noise_level)

    # dncnn
    tau = DnCNN_taus[i]

    # -- Reconstruction --#
    dObj = BlockRandomClass(y_blocks, recon_shape, A_blocks)
    rObj = DnCNNstarClass(model_path, tau, data_kargs, gpu_ratio=0.2)

    print()
    print('###################')
    print('#### Async-RED ####')
    print('###################')
    print()

    # - To try out direct DnCNN, set useNoise to False.
    # - To denoise with full denoiser, set pad to None.
    # - To denoise with block-wise denoiser, set pad to some scalar (5 by default).
    # We set the step-size to be 1/(L+2*tau)
    alg_args['step'] = 1 / (2 + 2 * tau)
    alg_args['num_processes'] = pnum
    time_start = time.time()
    asyncRED_recon, asyncRED_out, path = asyncRED_solver(dObj, rObj, **alg_args)
    time_end = time.time() - time_start
    print(f"Total time used: {time_end:.{4}}")
    asyncRED_out['recon'] = asyncRED_recon
    asyncRED_out['tau'] = tau

    # save out info
    asyncRED_output['img_{}'.format(i)] = asyncRED_out
    if not os.path.exists('results'):
        os.makdir('results')
    sio.savemat('results/AsyncRED_{}-{}_proc={}.mat'.format(startIndex,endIndex,pnum), asyncRED_output)


    ####################################################
    ####              PlOT CONVERGENCE               ###
    ####################################################

    # asyncRED_dist = asyncRED_out['dist']
    asyncRED_snr  = asyncRED_out['snr']

    # compute the averaged distance to fixed points
    avgSnrAsyncRED = np.squeeze(asyncRED_snr)
    xRange = np.linspace(0, alg_args['num_iter'], alg_args['num_iter'])
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # SNR Plot
    ax1.plot(xRange, avgSnrAsyncRED, label='Async-RED')
    ax1.set_xlim(0, alg_args['num_iter'])
    ax1.set_ylim(0, 30)
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('SNR (dB)')
    ax1.set_title('SNR plot for Async-RED')

    # the image
    ax2.imshow(asyncRED_out['recon'])

    plt.show()
