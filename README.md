# [Async-RED: A Provably Convergent Asynchronous Block Parallel Stochastic Method using Deep Denoising Priors](https://openreview.net/forum?id=9EsrXMzlFQY)

Regularization by denoising (RED) is a recently developed framework for solving inverse problems by integrating advanced denoisers as image priors. Recent work has shown its state-of-the-art performance when combined with pre-trained deep denoisers. However, current RED algorithms are inadequate for parallel processing on multicore systems. We address this issue by proposing a new{asynchronous RED (Async-RED) algorithm that enables asynchronous parallel processing of data, making it significantly faster than its serial counterparts for large-scale inverse problems. The computational complexity of Async-RED is further reduced by using a random subset of measurements at every iteration. We present a complete theoretical analysis of the algorithm by establishing its convergence under explicit assumptions on the data-fidelity and the denoiser. We validate Async-RED on image recovery using pre-trained deep denoisers as priors.

## How to run the code

### Prerequisites
```
python 3.6  
tensorflow 1.12  
scipy 1.2.1  
numpy v1.17  
matplotlib v3.3.4
```
It is better to use Conda for installation of all dependecies.

### Run the Demo
We provide the script
```
demo_asyncRED.py
```
to demonstrate the performance of Async-RED with block-diagnal compressive sensing matrix. One can run the code by simply typing

```
$ python demo_asyncRED.py
```

To try with different settings, please open the script and follow the instruction inside.

## Citation
Y. Sun, J. Liu, Y. Sun, B. Wohlberg, and U. S. Kamilov, “Async-RED: A Provably Convergent Asynchronous Block Parallel Stochastic Method using Deep Denoising Priors,” Proc. Int. Conf. Learn. Represent. (ICLR 2021) (Vienna, Austria, May 4-8).
```
@inproceedings{
sun2021asyncred,
title={Async-{\{}RED{\}}: A Provably Convergent Asynchronous Block Parallel Stochastic Method using Deep Denoising Priors},
author={Yu Sun and Jiaming Liu and Yiran Sun and Brendt Wohlberg and Ulugbek Kamilov},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=9EsrXMzlFQY}
}
```
