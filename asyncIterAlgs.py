# library
import os
import shutil
import numpy as np
import time
import multiprocessing as mp
from multiprocessing.sharedctypes import Value, Array, RawArray
# scripts
import util

############################
##### Helper Functions #####
############################

def mp2np(mp_arr):
    return np.frombuffer(mp_arr.get_obj())

def read_x(x_blocks, blocks_shape):
    xtilde_blocks = mp2np(x_blocks).reshape(blocks_shape, order='C')  # reshape by rows
    return xtilde_blocks

def read_x_list(x_blocks):
    xtilde_blocks = np.copy(x_blocks)
    return xtilde_blocks

def evaluateSnr(xref, xin):
    return 20 * np.log10(np.linalg.norm(xref.flatten('F')) / np.linalg.norm(xref.flatten('F') - xin.flatten('F')))


##########################
##### Define Process #####
##########################

def uniform_block_process(x_blocks, dist, timer, snr, global_count,              # share_able variables
                          cpu_idx, gpu_idx, data_obj, rglr_obj,                  # cpu, gpu & objects
                          step, is_noise, pad, minibatch_size, max_global_iter,  # algorithmic parameters
                          logging, verbose, is_save, save_every,                 # logging parameters
                          is_conv, conv_every, save_path, xtrue) -> None:
    #### CPU Core Assigment ####
    process = mp.current_process()
    os.system("taskset -p -c %d %d" % (cpu_idx, process.pid))

    #### Random Seed ####
    np.random.seed(cpu_idx)

    ##### Main Loop ####
    # initialize some variables
    rglr_obj.init(gpu_idx=gpu_idx)

    while global_count.value < max_global_iter:

        # select block randomly at uniform
        block_idx = np.random.randint(data_obj.num_blocks)

        # 1st read
        xtilde_blocks = read_x_list(x_blocks)
        xtilde_block = xtilde_blocks[block_idx,:,:]

        # get the block gradient of data-fit
        g_data_block = data_obj.gradStoc_block(xtilde_block, block_idx, minibatch_size)
        if pad is 'full':
            xtilde = util.putback_nonoverlap_patches(xtilde_blocks)
            g_rglr = rglr_obj.red(xtilde, is_noise=is_noise, extend_p=0)
            g_rglr_block = util.extract_nonoverlap_patches(g_rglr, 
                    data_obj.num_blocks, data_obj.block_size)[block_idx,:,:]
        else:
            g_rglr_block = rglr_obj.red(xtilde_block, is_noise=is_noise, extend_p=pad)  # pad removed

        # compute the overall gradient g_tot
        g_tot_block = g_data_block + g_rglr_block

        # update the selected block & update global memory
        # upload new x to the global memory
        x_blocks[block_idx] = x_blocks[block_idx]-step*g_tot_block
        if logging:
            xlog = util.putback_nonoverlap_patches(read_x_list(x_blocks))
        with global_count.get_lock():
            local_count = np.copy(global_count.value)  # record local count for logging
            global_count.value += 1  # update global count

        # record final finishing time
        if local_count < max_global_iter:
            timer[local_count] = time.time() - timer[local_count]

        #### Log Info ####
        if logging and local_count < max_global_iter:
            if is_conv and (local_count+1) % conv_every == 0:
                # calculate full gradient (g = Sx)
                g_full_data = data_obj.grad(xtilde_blocks)
                g_full_rglr = rglr_obj.red(xtilde_blocks, is_noise=is_noise)
                g_full_tot = g_full_data + g_full_rglr
                dist[local_count] = np.linalg.norm(g_full_tot.flatten('F')) ** 2

            if snr is not None:
                snr[local_count] = evaluateSnr(xtrue, xlog)

            # save & print
            if is_save and (local_count+1) % save_every == 0:
                util.save_mat(xlog, save_path + '/iter_{}_mat.mat'.format(local_count + 1))
                util.save_img(xlog, save_path + '/iter_{}_img.tif'.format(local_count + 1))

            np.set_printoptions(precision=3)
            if verbose and snr is not None:
                print(
                    f"[uniform_block_async: {local_count + 1}/{max_global_iter}] [Process: {process.name} {process.pid}] "
                    f"[Step-size: {step:.{4}}] [||Gx_k||^2: {dist[local_count]:.{4}}] [SNR: {snr[local_count]:.{4}}] "
                    f"[Time: {timer[local_count]:.{4}}]", flush=True)
            elif verbose:
                print(
                    f"[uniform_block_async: {local_count + 1}/{max_global_iter}] [Process: {process.name} {process.pid}] "
                    f"[Step-size: {step:.{4}}] [||Gx_k||^2: {dist[local_count]:.{4}}] "
                    f"[Timer: {timer[local_count]:.{4}}]", flush=True)


def epoch_block_process(x_blocks, dist, timer, snr, global_count, block_queue, # share_able variables
                        cpu_idx, gpu_idx, data_obj, rglr_obj,                  # cpu, gpu & objects
                        step, is_noise, pad, minibatch_size, max_global_iter,  # algorithmic parameters
                        logging, verbose, is_save, save_every,                 # logging parameters
                        is_conv, conv_every, save_path, xtrue) -> None:
    #### CPU Core Assigment ####
    process = mp.current_process()
    os.system("taskset -p -c %d %d" % (cpu_idx, process.pid))

    #### Random Seed ####
    np.random.seed(cpu_idx)

    ##### Main Loop ####
    # initialize some variables
    rglr_obj.init(gpu_idx=gpu_idx)

    ##### Parameters #####
    blocks_shape = [data_obj.num_blocks, data_obj.block_size, data_obj.block_size]

    while global_count.value < max_global_iter:

        # get the block index from the FIFO queue
        block_idx = block_queue.get()
        # put back the index immediately to prevent empty queue
        block_queue.put(block_idx)

        # 1st read
        xtilde_blocks = read_x(x_blocks, blocks_shape)
        xtilde_block = xtilde_blocks[block_idx,:,:]

        # get the block gradient of data-fit
        g_data_block = data_obj.gradStoc_block(xtilde_block, block_idx, minibatch_size)
        if pad is 'full':
            xtilde = util.putback_nonoverlap_patches(xtilde_blocks)
            g_rglr = rglr_obj.red(xtilde, is_noise=is_noise, extend_p=0)
            g_rglr_block = util.extract_nonoverlap_patches(g_rglr, 
                    data_obj.num_blocks, data_obj.block_size)[block_idx,:,:]
        else:
            g_rglr_block = rglr_obj.red(xtilde_block, is_noise=is_noise, extend_p=pad)  # pad removed

        # compute the overall gradient g_tot
        g_tot_block = g_data_block + g_rglr_block

        # update the selected block & update global memory
        # upload new x to the global memory
        start_idx = int(block_idx * data_obj.block_size**2)
        end_idx = int((block_idx+1) * data_obj.block_size**2)
        x_blocks[start_idx:end_idx] = x_blocks[start_idx:end_idx] - step * g_tot_block.flatten('C')
        if logging:
            xlog = util.putback_nonoverlap_patches(read_x(x_blocks, blocks_shape))
        # update the global count
        with global_count.get_lock():
            local_count = np.copy(global_count.value)  # record local count for logging
            global_count.value += 1  # update global count

        # record final finishing time
        if local_count < max_global_iter:
            timer[local_count] = time.time() - timer[local_count]

        #### Log Info ####
        if logging and local_count < max_global_iter:
            if is_conv and (local_count+1) % conv_every == 0:
                # calculate full gradient (g = Sx)
                g_full_data = data_obj.grad(xtilde_blocks)
                g_full_rglr = rglr_obj.red(xtilde_blocks, is_noise=is_noise)
                g_full_tot = g_full_data + g_full_rglr
                dist[local_count] = np.linalg.norm(g_full_tot.flatten('F')) ** 2

            if snr is not None:
                snr[local_count] = evaluateSnr(xtrue, xlog)

            # save & print
            if is_save and (local_count+1) % save_every == 0:
                util.save_mat(xlog, save_path + '/iter_{}_mat.mat'.format(local_count + 1))
                util.save_img(xlog, save_path + '/iter_{}_img.tif'.format(local_count + 1))

            np.set_printoptions(precision=3)
            if verbose and snr is not None:
                print(
                    f"[epoch_block_async: {local_count + 1}/{max_global_iter}] [Process: {process.name} {process.pid}] "
                    f"[Step-size: {step:.{4}}] [||Gx_k||^2: {dist[local_count]:.{4}}] [SNR: {snr[local_count]:.{4}}] "
                    f"[Time: {timer[local_count]:.{4}}]", flush=True)
            elif verbose:
                print(
                    f"[epoch_block_async: {local_count + 1}/{max_global_iter}] [Process: {process.name} {process.pid}] "
                    f"[Step-size: {step:.{4}}] [||Gx_k||^2: {dist[local_count]:.{4}}] "
                    f"[Timer: {timer[local_count]:.{4}}]", flush=True)



#############################
##### Iterative Methods #####
#############################

def asyncRED_solver(data_obj, rglr_obj, 
                    num_processes=4, cpu_offset=0, processes_per_gpu=4,
                    pad=10, minibatch_size=500, num_iter=500, 
                    step=0.1, is_noise=True, mode='epoch',
                    logging=False, verbose=True, 
                    is_save=True, save_every=1,
                    is_conv=True, conv_every=1, 
                    save_note=None, xtrue=None, xinit=None) -> [np.ndarray, dict]:
    """
    Asynchoronous Regularization by denoising (Async-RED)

    ### INPUT:
    data_obj   ~ the data fidelity term, measurement/forward model.
    rglr_obj   ~ the regularizer term.
    num_processes ~ total number of processes to be launched. *num_processes* cpus will be occupied.
    cpu_offset ~ CPU starting index.
    processes_per_gpu ~ number of processes running on 1 GPU.
    pad        ~ the pad size for block-wise denoising / set to 'Full' if you want to use the full denoiser.
    num_iter   ~ the total (global) number of iterations.
    step       ~ the step-size.
    logging    ~ if logging is true (general control).
    verbose    ~ if true print info of each iteration.
    is_save    ~ if true save the reconstruction of each iteration.
    save_every ~ save image every *save_every* iteration.
    is_conv    ~ if true compute convergece.
    conv_every ~ compute convergence measure every *conv_every* iterations.
    save_note  ~ the save path for is_save.
    xtrue      ~ the ground truth of the image, for tracking purpose.
    xinit      ~ the initial value of x.

    ### OUTPUT:
    x     ~ final reconstructed signal.
    outs  ~ detailed information including convergence measure, SNR, step-size, and time cost of each iteration.
    """

    ####################################################
    ####                 SAVE ROOTS                  ###
    ####################################################

    # you can change the save path here
    # save_root = 'results/Async_DnCNNstar_Random/'
    save_path = 'results/Async_{}_{}_{}'.format(save_note, 
                    data_obj.__class__.__name__[0:-5], rglr_obj.__class__.__name__[0:-5]) + \
                '_numprocess={}_iters={}_step={:f}_numblock={}'.format(
                    num_processes,num_iter, step, data_obj.num_blocks) + \
                '_blocksize={}_pad={}_minibatch={}_mode={}_tau={}'.format(
                    data_obj.block_size,pad, minibatch_size, mode, rglr_obj.tau)

    abs_save_path = os.path.abspath(os.path.join(save_path,'log'))
    if is_save:
        if os.path.exists(abs_save_path):
            print("Removing '{:}'".format(abs_save_path))
            shutil.rmtree(abs_save_path, ignore_errors=True)
        # make new path
        print("Allocating '{:}'".format(abs_save_path))
        os.makedirs(abs_save_path)

    # initialize variables
    if xinit is None:
        xinit_blocks = np.zeros([data_obj.num_blocks, data_obj.block_size, data_obj.block_size], dtype=np.float32)
    else:
        xinit_blocks = util.extract_nonoverlap_patches(data_obj.num_blocks, data_obj.block_size,)
    global_count = Value('i', 0)
    if mode == 'uniform':
        man = mp.Manager()
        x_blocks = man.list(xinit_blocks)
    else:
        x_blocks = Array('d', xinit_blocks.flatten('C'))  # flatten by rows (Only for storing the shared vector) !!!

    # logging variables & locker for logging
    dist = RawArray('d', np.zeros(num_iter))
    snr = RawArray('d', np.zeros(num_iter))
    timer = RawArray('d', time.time()*np.ones(num_iter))

    # launch asynchronous block process
    p_list = []
    if mode == 'uniform':
        for i in range(num_processes):
            process_args = (x_blocks, dist, timer, snr, global_count,  # share_able variables
                            i+cpu_offset, int(i / processes_per_gpu), data_obj, rglr_obj, 
                            step, is_noise, pad, minibatch_size, num_iter,
                            logging, verbose, is_save, save_every,
                            is_conv, conv_every, abs_save_path, xtrue)
            p = mp.Process(name=f"worker {i}", target=uniform_block_process, args=process_args)
            p.start()
            p_list.append(p)
    elif mode == 'epoch':
        # create a block queue
        block_queue = mp.Queue()
        for j in np.random.permutation(data_obj.num_blocks):
            block_queue.put(j)
        for i in range(num_processes):
            process_args = (x_blocks, dist, timer, snr, global_count, block_queue,  # share_able variables
                            i+cpu_offset, int(i / processes_per_gpu), data_obj, rglr_obj, 
                            step, is_noise, pad, minibatch_size, num_iter,
                            logging, verbose, is_save, save_every,
                            is_conv, conv_every, abs_save_path, xtrue)
            p = mp.Process(name=f"worker {i}", target=epoch_block_process, args=process_args)
            p.start()
            p_list.append(p)
    else:
        print(f"mode {mode} is not available, aborted")
        exit()

    # Terminate all processes
    for p in p_list:
        p.join()

    # collect results
    outs = {
        'snr': np.array(snr),
        'dist': np.array(dist),
        'timer': np.array(timer),
    }

    if mode is 'uniform':
        recon_blocks = read_x_list(x_blocks)
    else:
        recon_blocks = read_x(x_blocks, [data_obj.num_blocks,data_obj.block_size,data_obj.block_size])
    recon = util.putback_nonoverlap_patches(recon_blocks)  # reform blocks into one image

    return recon, outs, os.path.abspath(save_path)