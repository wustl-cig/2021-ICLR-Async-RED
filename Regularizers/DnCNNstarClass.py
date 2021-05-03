from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
import util

from Regularizers.RegularizerClass import RegularizerClass


class DnCNNstarClass(RegularizerClass):
    """
    data_kargs:
        nx, ny, (nz) ~ 2D/3D spatial size of the image
        ic ~ input data channel size
        oc ~ ground truth channel size

    net_kargs:
        layer_num ~ the number of layers (dynamically adapted to the architecture)
        filter_size ~ the size of the convolutional filter
        feature_root ~ the starting number of feature maps

    train_kargs:
        batch_size ~ the size of training batch
        valid_size ~ the size of valid batch
        learning_rate ~ could be a list of learning rate corresponding to differetent epoches
        epoches ~ number of epoches
        is_restore ~ True / False
        prediction_path ~ where to save predicted results. No saves if set to None. (also used to save validation)
        save_epoch ~ save model every save_epochs

    """

    def __init__(self, model_path, tau, data_kargs, gpu_ratio=0.1):
        # data args
        self.data_kargs = data_kargs
        # regularization parameter
        self.tau = tau
        # model path
        self.model_path = model_path
        # gpu ratio
        self.gpu_ratio = gpu_ratio
        # will be initialized in init()
        self.x = None
        self.xhat = None
        self.input_shape_of_conv_layer = None
        self.sess = None


    def net(self):
        # input layer
        input_shape_of_conv_layer = []
        with tf.variable_scope('DnCNN'):
            with tf.variable_scope('layer_1'):
                input_shape_of_conv_layer.append([self.x.shape[1], self.x.shape[2]])
                in_node = tf.layers.conv2d(self.x, 64, 3, padding='same', activation=tf.nn.relu)
            for layer in range(2, 6 + 1):
                with tf.variable_scope('layer_{}'.format(layer)):
                    input_shape_of_conv_layer.append([self.x.shape[1], self.x.shape[2]])
                    in_node = tf.layers.conv2d(in_node, 64, 3, padding='same', name='conv2d_{}'.format(layer), use_bias=True)
                    in_node = tf.nn.relu(in_node)
            with tf.variable_scope('layer_7'):
                input_shape_of_conv_layer.append([self.x.shape[1], self.x.shape[2]])
                output = tf.layers.conv2d(in_node, self.data_kargs['oc'], 3, padding='same')
        return output, input_shape_of_conv_layer

    def restore(self, sess, model_path):
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        tf.logging.info("Model restored from file: %s" % model_path)

    def lipschitz_upper(self):
        lipschitz_net = 1
        convolutional_operators = [v for v in self._get_vars() if 'kernel:' in v.name]
        for i in range(len(convolutional_operators)):
            conv_operator = convolutional_operators[i]
            input_shape = self.input_shape_of_conv_layer[i]
            lipschitz_layer_ = self._compute_singular_values(conv_operator, input_shape)
            lipschitz_layer = self.sess.run(lipschitz_layer_)
            print('The estimate of the upper bound of the lipschitz constant of Conv {}: {}'.format(i, lipschitz_layer))
            lipschitz_net = lipschitz_net * lipschitz_layer
        return lipschitz_net

    def init(self, **kwargs):
        tf.reset_default_graph()
        # define graph
        self.x = tf.placeholder(tf.float32, shape=[None, None, None, self.data_kargs['ic']], name='input')
        self.xhat, self.input_shape_of_conv_layer = self.net()

        # define sess
        config = tf.ConfigProto()
        config.log_device_placement = True
        config.gpu_options.per_process_gpu_memory_fraction = self.gpu_ratio
        config.gpu_options.visible_device_list = str(kwargs['gpu_idx'])
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.restore(self.sess, self.model_path)
        ## dummy return for tv completion
        # return 0, np.zeros([kwargs['num_blocks'], 1])

    def red(self, s, is_noise=False, extend_p=None, pad_mode='reflect'):
        extend_p = 0 if extend_p is None else extend_p
        size  = s.shape[-1]

        if len(s.shape) == 2:
            sfull = np.pad(s, ((extend_p,), (extend_p,)), pad_mode)
            stemp = np.expand_dims(np.expand_dims(sfull, axis=-1), axis=0)
            xtemp = self.sess.run(self.xhat, feed_dict={self.x: stemp})
        elif len(s.shape) == 3:
            sfull = util.putback_nonoverlap_patches(s)
            size  = sfull.shape[1]
            stemp = np.expand_dims(np.expand_dims(sfull, axis=-1), axis=0)
            xtemp = self.sess.run(self.xhat, feed_dict={self.x: stemp})
        else:
            print('Incorrect s.shape:{}'.format(s.shape))
            exit()

        if is_noise:
            noise = self.tau * xtemp.squeeze()
        else:
            noise = self.tau * (sfull - xtemp.squeeze())

        return noise[extend_p:extend_p+size, extend_p:extend_p+size]

    def prox(self, s, step):
        if len(s.shape) == 2:
            # reshape
            s = np.expand_dims(np.expand_dims(s, axis=-1), axis=0)
            xtemp = self.sess.run(self.xhat, feed_dict={self.x: s})
        else:
            print('Incorrect s.shape')
            exit()

        return xtemp.squeeze()

    def eval(self, x):
        return 0

    @staticmethod
    def _compute_singular_values(conv, inp_shape):
        """ Find the singular values of the linear transformation
        corresponding to the convolution represented by conv on
        an n x n x depth input. """
        conv_tr = tf.cast(tf.transpose(conv, perm=[2, 3, 0, 1]), tf.complex64)
        conv_shape = conv.get_shape().as_list()
        a1 = int(inp_shape[0] - conv_shape[0])
        a2 = int(inp_shape[1] - conv_shape[1])
        padding = tf.constant([[0, 0], [0, 0],
                               [0, a1],
                               [0, a2]])
        transform_coeff = tf.fft2d(tf.pad(conv_tr, padding))
        singular_values = tf.svd(tf.transpose(transform_coeff, perm=[2, 3, 0, 1]), compute_uv=False)

        return tf.reduce_max(singular_values)

    @staticmethod
    def _get_vars():
        lst_vars = []
        for v in tf.global_variables():
            lst_vars.append(v)
        return lst_vars
