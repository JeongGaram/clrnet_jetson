import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import numpy as np

def process1():
    # create a new CUDA context
    cuda_ctx = cuda.Device(0).make_context()

    # create a CUDA stream
    stream = cuda.Stream()

    # allocate GPU memory
    a_gpu = gpuarray.to_gpu(np.ones((100,100)).astype(np.float32))

    # process 1 CUDA code here
    a_gpu *= 2

    # synchronize with the stream
    stream.synchronize()

    # release the CUDA context
    cuda_ctx.pop()

def process2():
    # create a new CUDA context
    cuda_ctx = cuda.Device(0).make_context()

    # create a CUDA stream
    stream = cuda.Stream()

    # allocate GPU memory
    b_gpu = gpuarray.to_gpu(np.ones((100,100)).astype(np.float32))

    # process 2 CUDA code here
    b_gpu *= 3

    # synchronize with the stream
    stream.synchronize()

    # release the CUDA context
    cuda_ctx.pop()

if __name__ == '__main__':
    process1()
    process2()
