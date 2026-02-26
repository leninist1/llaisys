#include "rand_sample_nvidia.hpp"

#include "../../../core/llaisys_core.hpp"
#include "../../../utils.hpp"
#include "../../nvidia_utils.cuh"
#include "../cpu/rand_sample_cpu.hpp"

namespace llaisys::ops::nvidia {
void rand_sample(std::byte *sample_idx, std::byte *sample_val, const std::byte *vals, llaisysDataType_t type, size_t numel,
                 int64_t batch_size, float temperature, size_t topK, float topP, int64_t seed) {
    auto &runtime = llaisys::core::context().runtime();
    size_t vals_bytes = numel * static_cast<size_t>(batch_size) * llaisys::utils::dsize(type);
    size_t idx_bytes = static_cast<size_t>(batch_size) * sizeof(int64_t);
    size_t val_bytes = static_cast<size_t>(batch_size) * llaisys::utils::dsize(type);

    auto host_vals = static_cast<std::byte *>(runtime.api()->malloc_host(vals_bytes));
    auto host_idx = static_cast<std::byte *>(runtime.api()->malloc_host(idx_bytes));
    auto host_val = static_cast<std::byte *>(runtime.api()->malloc_host(val_bytes));

    runtime.api()->memcpy_sync(host_vals, vals, vals_bytes, LLAISYS_MEMCPY_D2H);
    cpu::rand_sample(host_idx, host_val, host_vals, type, numel, batch_size, temperature, topK, topP, seed);
    runtime.api()->memcpy_sync(sample_idx, host_idx, idx_bytes, LLAISYS_MEMCPY_H2D);
    runtime.api()->memcpy_sync(sample_val, host_val, val_bytes, LLAISYS_MEMCPY_H2D);

    runtime.api()->free_host(host_vals);
    runtime.api()->free_host(host_idx);
    runtime.api()->free_host(host_val);
}
}
