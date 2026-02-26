#include "self_attention_nvidia.hpp"

#include "../../../core/llaisys_core.hpp"
#include "../../../utils.hpp"
#include "../cpu/self_attention_cpu.hpp"

namespace llaisys::ops::nvidia {
void self_attention(std::byte *out, const std::byte *q, const std::byte *k, const std::byte *v, float scale,
                    llaisysDataType_t type, size_t qlen, size_t kvlen, size_t nh, size_t nkvh, size_t hd, size_t dv) {
    auto &runtime = llaisys::core::context().runtime();
    size_t elem_size = llaisys::utils::dsize(type);
    size_t q_bytes = qlen * nh * hd * elem_size;
    size_t k_bytes = kvlen * nkvh * hd * elem_size;
    size_t v_bytes = kvlen * nkvh * dv * elem_size;
    size_t out_bytes = qlen * nh * dv * elem_size;

    auto host_q = static_cast<std::byte *>(runtime.api()->malloc_host(q_bytes));
    auto host_k = static_cast<std::byte *>(runtime.api()->malloc_host(k_bytes));
    auto host_v = static_cast<std::byte *>(runtime.api()->malloc_host(v_bytes));
    auto host_out = static_cast<std::byte *>(runtime.api()->malloc_host(out_bytes));

    runtime.api()->memcpy_sync(host_q, q, q_bytes, LLAISYS_MEMCPY_D2H);
    runtime.api()->memcpy_sync(host_k, k, k_bytes, LLAISYS_MEMCPY_D2H);
    runtime.api()->memcpy_sync(host_v, v, v_bytes, LLAISYS_MEMCPY_D2H);

    cpu::self_attention(host_out, host_q, host_k, host_v, scale, type, qlen, kvlen, nh, nkvh, hd, dv);
    runtime.api()->memcpy_sync(out, host_out, out_bytes, LLAISYS_MEMCPY_H2D);

    runtime.api()->free_host(host_q);
    runtime.api()->free_host(host_k);
    runtime.api()->free_host(host_v);
    runtime.api()->free_host(host_out);
}
}
