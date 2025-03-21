import utils;

/////////////////////////////////////////////////////////////////////////////////
// Cuda kernels
/////////////////////////////////////////////////////////////////////////////////

bool is_valid_tri(float3 p0, float3 p1, float3 p2, float near, float far) {
    bool valid_depth = all(abs(float3(p0.z, p1.z, p2.z)) < float3(far));
    valid_depth &=     all(abs(float3(p0.z, p1.z, p2.z)) > float3(near));
    bool valid_values =  !(any(isnan(p0)) || any(isnan(p1)) || any(isnan(p2)));
    valid_values     &=  !(any(isinf(p0)) || any(isinf(p1)) || any(isinf(p2)));

    return valid_depth && valid_values;
}


[CudaKernel]
[AutoPyBindCUDA]
void calc_tri_indices_kernel(
    TensorView<float3> pos, 
    int width, int height, 
    float near, float far,
    TensorView<int3> output)
{
    uint3 idx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
    if (idx.x >= width-1 || idx.y >= height-1 || idx.z >= 1)
        return;
    int pos_idx = idx.y * width + idx.x;
    int tri_idx = 2 * (idx.y * (width - 1) + idx.x);

    bool tri_diag =  bool(idx.x % 2) ^ bool(idx.y % 2);

    if (tri_diag){
    int3 tri1 = int3(pos_idx, pos_idx + 1, pos_idx + width);
    if (is_valid_tri(pos[tri1.x], pos[tri1.y], pos[tri1.z], near, far)) 
        output[tri_idx] = tri1;
    else
        output[tri_idx] = int3(-1);

    int3 tri2 = int3(pos_idx + 1, pos_idx + 1 + width, pos_idx + width);
    if (is_valid_tri(pos[tri2.x], pos[tri2.y], pos[tri2.z], near, far))
        output[tri_idx + 1] = tri2;
    else
        output[tri_idx + 1] = int3(-1);

    } else{
        int3 tri1 = int3(pos_idx, pos_idx + 1 + width, pos_idx + width);
        if (is_valid_tri(pos[tri1.x], pos[tri1.y], pos[tri1.z], near, far)) 
            output[tri_idx] = tri1;
        else
            output[tri_idx] = int3(-1);

        int3 tri2 = int3(pos_idx + 1, pos_idx + 1 + width, pos_idx);
        if (is_valid_tri(pos[tri2.x], pos[tri2.y], pos[tri2.z], near, far))
            output[tri_idx + 1] = tri2;
        else
            output[tri_idx + 1] = int3(-1);
    }

}



// /////////////////////////////////////////////////////////////////////////////////
// // Torch entry point
// /////////////////////////////////////////////////////////////////////////////////

// [TorchEntryPoint]
// TorchTensor<int3> calc_tri_indices(
//     TorchTensor<float3> pos, int width, int height)
// {
//     var num_tris = 2 * (width - 1) * (height - 1);
//     var result = TorchTensor<int3>.alloc(num_tris);
//     uint3 dims = uint3(result.size(2), result.size(1), result.size(0));
//     let blockSize = getLaunchBlockSize(8, 8, dims);
//     let blockCount = getLaunchGridSize(blockSize, dims);

//     __dispatch_kernel(calc_tri_indices_kernel, blockCount, blockSize)(pos, width, height, result);
//     return result;
// }