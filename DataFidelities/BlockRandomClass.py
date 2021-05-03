import numpy as np
import util
from DataFidelities.DataClass import DataClass


class BlockRandomClass(DataClass):

    def __init__(self, y_blocks, recon_shape, A_blocks):
        self.y_blocks = y_blocks
        self.A_blocks = A_blocks
        self.recon_shape = recon_shape
        self.num_blocks, self.num_meas, block_size_sq = A_blocks.shape
        self.block_size = int(np.sqrt(block_size_sq))

    def resStoc_block(self, x_block, block_idx, idx_set):
        res = self.fmult(x_block, self.A_blocks[block_idx, idx_set, :]) \
                    - self.y_blocks[block_idx, idx_set]
        return res

    def res_block(self, x_block, block_idx):
        res = self.fmult(x_block, self.A_blocks[block_idx, :, :]) \
                    - self.y_blocks[block_idx]
        return res

    def res(self, x_blocks):
        res_blocks = []
        for i in range(self.num_blocks):
            z_block = self.res_block(x_blocks[i], i)
            res_blocks.append(z_block)
        return np.array(res_blocks)

    def gradStoc_block(self, x_block, block_idx, minibatch_size):
        if minibatch_size is 'full':
            idx_set = np.array(range(self.num_meas))
        else:
            idx_set = np.random.permutation(self.num_meas)[0:minibatch_size]
        res = self.resStoc_block(x_block, block_idx, idx_set)
        g = self.ftran(res, self.A_blocks[block_idx, idx_set, :])
        return g

    def grad_block(self, x_block, block_idx):
        res = self.res_block(x_block, block_idx)
        g = self.ftran(res, self.A_blocks[block_idx, :, :])
        return g

    def grad(self, x_blocks):
        grad_blocks = []
        for i in range(self.num_blocks):
            grad_block = self.grad_block(x_blocks[i], i)
            grad_blocks.append(grad_block)
        g = util.putback_nonoverlap_patches(np.array(grad_blocks))
        return g

    @staticmethod
    def generate_A(num_blocks, block_size, downsample_rate=1):
        # downsample rate is the downsample rate of nx and ny
        # size of image should be the power of 2
        # num of blocks better to be power of 2
        d_size = int(downsample_rate * block_size)  # downsample size of width
        A_blocks = np.random.randn(num_blocks, d_size**2, block_size**2) / np.sqrt(d_size**2)
        return np.array(A_blocks)

    @staticmethod
    def generate_y(x_blocks, A_blocks, noise_level=30):
        num_blocks, d_size_sq, block_size_sq = A_blocks.shape
        z_blocks = []
        for i in range(num_blocks):
            z_block = np.dot(A_blocks[i], x_blocks[i].flatten('F'))
            z_blocks.append(z_block)
        y_blocks, _ = util.addwgn(np.array(z_blocks), noise_level)
        return y_blocks

    @staticmethod
    def fmult(x, A):
        meas_size_sq = A.shape[0]
        z = np.dot(A, x.flatten('F'))
        return z

    @staticmethod
    def ftran(z, A):
        block_size_sq = A.shape[1]
        block_size = int(np.sqrt(block_size_sq))
        x = np.dot(A.T, z.flatten('F'))
        return x.reshape([block_size, block_size], order='F')
