# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import subprocess
from logger import get_logger

log = get_logger("amd_hipify")

HIPIFY_PERL = '/opt/rocm/bin/hipify-perl'

def hipify(src_file_path, dst_file_path):
    log.debug('Hipifying: "{}" -> "{}"'.format(src_file_path, dst_file_path))

    dst_file_path = dst_file_path.replace('cuda', 'rocm')
    dir_name = os.path.dirname(dst_file_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
    with open(dst_file_path, 'w') as f:
        subprocess.run([HIPIFY_PERL, src_file_path], stdout=f)
    with open(dst_file_path) as f:
        s = f.read().replace('kCudaExecutionProvider', 'kRocmExecutionProvider')
        s = s.replace('CublasHandle', 'RocblasHandle')
        s = s.replace('cublas_handle', 'rocblas_handle_var')
        s = s.replace('hipblasHandle_t', 'rocblas_handle')
        s = s.replace('CudaAsyncBuffer', 'RocmAsyncBuffer')
        s = s.replace('CudaKernel', 'RocmKernel')
        s = s.replace('ToCudaType', 'ToHipType')
        s = s.replace('CudaT', 'HipT')
        s = s.replace('CUDA_LONG', 'HIP_LONG')
        s = s.replace('CUDA_RETURN_IF_ERROR', 'HIP_RETURN_IF_ERROR')
        s = s.replace('CUDA_KERNEL_ASSERT', 'HIP_KERNEL_ASSERT')
        s = s.replace('CUDA_CALL', 'HIP_CALL')
        s = s.replace('SliceCuda', 'SliceRocm')
        s = s.replace('thrust::cuda', 'thrust::hip')
        s = s.replace('cuda', 'rocm')
        # s = s.replace('Cuda', 'Rocm')
        s = s.replace('CUDA', 'ROCM')

        s = s.replace('GPU_WARP_SIZE = 32', 'GPU_WARP_SIZE = 64')
        s = s.replace('std::exp', 'expf')
        s = s.replace('std::log', 'logf')
        s = s.replace('#include <cub/device/device_radix_sort.cuh>',
                      '#include <hipcub/hipcub.hpp>\n#include <hipcub/backend/rocprim/device/device_radix_sort.hpp>')
        s = s.replace('#include <cub/device/device_reduce.cuh>',
                      '#include <hipcub/backend/rocprim/device/device_reduce.hpp>')
        s = s.replace('#include <cub/device/device_run_length_encode.cuh>',
                      '#include <hipcub/backend/rocprim/device/device_run_length_encode.hpp>')
        s = s.replace('#include <cub/device/device_scan.cuh>',
                      '#include <hipcub/backend/rocprim/device/device_scan.hpp>')
        s = s.replace('#include <cub/iterator/counting_input_iterator.cuh>',
                      '#include <hipcub/backend/rocprim/iterator/counting_input_iterator.hpp>')
        s = s.replace('#include <cub/iterator/discard_output_iterator.cuh>',
                      '#include <hipcub/backend/rocprim/iterator/discard_output_iterator.hpp>')
        s = s.replace('typedef half MappedType', 'typedef __half MappedType')
        # CUBLAS -> ROCBLAS
        # s = s.replace('CUBLAS', 'HIPBLAS')
        # s = s.replace('Cublas', 'Hipblas')
        # s = s.replace('cublas', 'hipblas')

        # CURAND -> HIPRAND
        s = s.replace('CURAND', 'HIPRAND')
        s = s.replace('Curand', 'Hiprand')
        s = s.replace('curand', 'hiprand')

        # NCCL -> RCCL
        # s = s.replace('NCCL_CALL', 'RCCL_CALL')
        s = s.replace('#include <nccl.h>', '#include <rccl.h>')

        # CUDNN -> MIOpen
        s = s.replace('CUDNN', 'MIOPEN')
        s = s.replace('Cudnn', 'Miopen')
        s = s.replace('cudnn', 'miopen')
        # hipify seems to have a bug for MIOpen, cudnn.h -> hipDNN.h, cudnn -> hipdnn
        s = s.replace('#include <hipDNN.h>', '#include <miopen/miopen.h>')
        s = s.replace('hipdnn', 'miopen')
        s = s.replace('HIPDNN_STATUS_SUCCESS', 'miopenStatusSuccess')
        s = s.replace('HIPDNN', 'MIOPEN')

        # CUSPARSE -> HIPSPARSE
        s = s.replace('CUSPARSE', 'HIPSPARSE')

        # CUFFT -> HIPFFT
        s = s.replace('CUFFT', 'HIPFFT')

        # Undo where above hipify steps went too far.
        s = s.replace('ROCM_VERSION', 'CUDA_VERSION')  # semantically different meanings, cannot hipify

    with open(dst_file_path, 'w') as f:
        f.write(s)


hipify(sys.argv[1], sys.argv[2])
