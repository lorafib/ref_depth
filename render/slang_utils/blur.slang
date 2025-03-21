import utils;

/////////////////////////////////////////////////////////////////////////////////
// Cuda kernels
/////////////////////////////////////////////////////////////////////////////////


[Differentiable]
float atrous(DiffTensorView<float> input, no_diff int step_size, uint2 idx){

    const float kernel[3] = { 1.0, 2.0 / 3.0, 1.0 / 6.0 };

    float center = input[idx];
    float accum_col = 0.f, accum_w = 0.f;
    for (int fy = -2; fy <= 2; fy++) {
        const int y = int(idx.x) + fy * step_size;
        if (y < 0 || y >= input.size(0)) continue;
        for (int fx = -2; fx <= 2; fx++) {
            const int x = int(idx.y) + fx * step_size;
            if (x < 0 || x >= input.size(1)) continue;
            const float tap = input[y, x];
            // compute bilateral filter weight
            const float w_kernel = kernel[abs(fx)] * kernel[abs(fy)];
            const float w_color = exp(-abs(center - tap));
            const float w = w_kernel * w_color;
            accum_col += w * tap;
            accum_w += w;
        }
    }

    return accum_col / (accum_w);
    
    
}




[CudaKernel]
void atrous_fwd_kernel(DiffTensorView<float> input, no_diff int step_size, TensorView<float> output) {
    uint2 idx = cudaBlockIdx().xy * cudaBlockDim().xy + cudaThreadIdx().xy;
    if (idx.x >= input.size(0) || idx.y >= input.size(1)) return;
    const float center = atrous(input, step_size, idx);

    output[idx] = center; //;taccum_col / accum_w;
}


[TorchEntryPoint]
TorchTensor<float> atrous_fwd(
    TorchTensor<float> input,  
    int step_size
) {
    var output = TorchTensor<float>.alloc(input.size(0), input.size(1));
    var d_input = DiffTensorView<float, AtomicAdd>(input);
    let groupSize = uint3(32, 32, 1);
    let blockCount = uint3((input.size(0) + 31) / 32, (input.size(1) + 31) / 32, 1);
    __dispatch_kernel(atrous_fwd_kernel, blockCount, groupSize)(d_input, step_size, output);
    return output;
}

// ---------------------------------------------------
// Backward pass

[CudaKernel]
void atrous_bwd_kernel(
    DiffTensorView<float> input,  // [h,w, 1]
    int step_size,              // [1]
    TensorView<float> output_grad
) {
    uint2 idx = cudaBlockIdx().xy * cudaBlockDim().xy + cudaThreadIdx().xy;
    if (idx.x >= input.size(0) || idx.y >= input.size(1)) return;
    __bwd_diff(atrous)(input, step_size, idx, output_grad[idx]);
}

[TorchEntryPoint]
TorchTensor<float> atrous_bwd(
    TorchTensor<float> input,  // [h,w, 1]
    int step_size,              // [1]
    TorchTensor<float> output_grad
) {
    var input_grad = TorchTensor<float>.zerosLike(input);
    var d_input = DiffTensorView<float, AtomicAdd>(input, {input_grad});
    let groupSize = uint3(32, 32, 1);
    let blockCount = uint3((input.size(0) + 31) / 32, (input.size(1) + 31) / 32, 1);
    __dispatch_kernel(atrous_bwd_kernel, blockCount, groupSize)(d_input, step_size, output_grad);
    return input_grad;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// masked atrous


[Differentiable]
float masked_atrous(
    DiffTensorView<float> input, 
    no_diff TensorView<float> mask, 
    no_diff TensorView<float> weight, 
    no_diff int step_size, 
    uint2 idx
){

    const float kernel[3] = { 1.0, 2.0 / 3.0, 1.0 / 6.0 };

    float center = input[idx];
    float accum_col = 0.f, accum_w = 0.f;
    for (int fy = -2; fy <= 2; fy++) {
        const int y = int(idx.x) + fy * step_size;
        if (y < 0 || y >= input.size(0)) continue;
        for (int fx = -2; fx <= 2; fx++) {
            const int x = int(idx.y) + fx * step_size;
            if (x < 0 || x >= input.size(1)) continue;
            const float tap_mask = no_diff mask[y, x];
            if (tap_mask < 1e-8) continue; 
            const float tap = input[y, x];
            // compute bilateral filter weight
            const float w_kernel = kernel[abs(fx)] * kernel[abs(fy)];
            const float w_from_input = no_diff weight[y, x];
            const float w = w_kernel * w_from_input;
            accum_col += w * tap;
            accum_w += w;
        }
    } 
    if (accum_w > 1e-8)
        return accum_col / accum_w;
    else
        return 0.;    
}




[CudaKernel]
void masked_atrous_fwd_kernel(
    DiffTensorView<float> input, 
    TensorView<float> mask, 
    TensorView<float> weight, 
    int step_size, 
    TensorView<float> output
) {
    uint2 idx = cudaBlockIdx().xy * cudaBlockDim().xy + cudaThreadIdx().xy;
    if (idx.x >= input.size(0) || idx.y >= input.size(1)) return;
    const float center = masked_atrous(input, mask, weight, step_size, idx);

    output[idx] = center; //;taccum_col / accum_w;
}


[TorchEntryPoint]
TorchTensor<float> masked_atrous_fwd(
    TorchTensor<float> input,  
    TorchTensor<float> mask,  
    TorchTensor<float> weight,  
    int step_size
) {
    var output = TorchTensor<float>.alloc(input.size(0), input.size(1));
    var d_input = DiffTensorView<float, AtomicAdd>(input);
    let groupSize = uint3(32, 32, 1);
    let blockCount = uint3((input.size(0) + 31) / 32, (input.size(1) + 31) / 32, 1);
    __dispatch_kernel(masked_atrous_fwd_kernel, blockCount, groupSize)(d_input, mask, weight, step_size, output);
    return output;
}

// ---------------------------------------------------
// Backward pass

[CudaKernel]
void masked_atrous_bwd_kernel(
    DiffTensorView<float> input,  // [h,w, 1]
    TensorView<float> mask,  
    TensorView<float> weight,  
    int step_size,              // [1]
    TensorView<float> output_grad
) {
    uint2 idx = cudaBlockIdx().xy * cudaBlockDim().xy + cudaThreadIdx().xy;
    if (idx.x >= input.size(0) || idx.y >= input.size(1)) return;
    __bwd_diff(masked_atrous)(input, mask, weight, step_size, idx, output_grad[idx]);
}

[TorchEntryPoint]
TorchTensor<float> masked_atrous_bwd(
    TorchTensor<float> input,  // [h,w, 1]
    TorchTensor<float> mask,  
    TorchTensor<float> weight,  
    int step_size,              // [1]
    TorchTensor<float> output_grad
) {
    var input_grad = TorchTensor<float>.zerosLike(input);
    var d_input = DiffTensorView<float, AtomicAdd>(input, {input_grad});
    let groupSize = uint3(32, 32, 1);
    let blockCount = uint3((input.size(0) + 31) / 32, (input.size(1) + 31) / 32, 1);
    __dispatch_kernel(masked_atrous_bwd_kernel, blockCount, groupSize)(d_input, mask, weight, step_size, output_grad);
    return input_grad;
}



// [CudaKernel]
// [AutoPyBindCUDA]
// [Differentiable]
// void masked_atrous_kernel(DiffTensorView<float> input, DiffTensorView<float> mask, DiffTensorView<float> in_weight, no_diff int step_size, DiffTensorView<float> output) {
//     uint3 idx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
//     if (idx.x >= output.size(0) || idx.y >= output.size(1) || idx.z >= output.size(2)) return;
//     const float kernel[3] = { 1.0, 2.0 / 3.0, 1.0 / 6.0 };
//     const float center = input[idx];
//     float accum_col = 0.f, accum_w = 0.f;
//     for (int fy = -2; fy <= 2; fy++) {
//         const int y = int(idx.x) + fy * step_size;
//         if (y < 0 || y >= output.size(0)) continue;
//         for (int fx = -2; fx <= 2; fx++) {
//             const int x = int(idx.y) + fx * step_size;
//             if (x < 0 || x >= output.size(1)) continue;
//             const float tap_mask = mask[y, x, idx.z];
//             if (tap_mask < 1e-8) continue; 
//             const float tap = input[y, x, idx.z];
//             // compute bilateral filter weight
//             const float w_kernel = kernel[abs(fx)] * kernel[abs(fy)];
//             const float w_from_input = 1.; //TODO   //in_weight[y, x, idx.z];
//             const float w = w_kernel * w_from_input;
//             accum_col += w * tap;
//             accum_w += w;
//         }
//     } 
//     output[idx] = accum_col / (accum_w+1);
//     // TOOD
//     // if (accum_w > 1e-6)
//     //     output[idx] = accum_col / (accum_w+1);
//     // else
//     //     output[idx] = 0.;

// }

