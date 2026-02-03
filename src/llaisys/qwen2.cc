#include "llaisys/models/qwen2.h"
#include "llaisys_tensor.hpp"

#include "../tensor/tensor.hpp"
#include "../ops/add/op.hpp"
#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rearrange/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"
#include "../utils.hpp"

#include <cmath>
#include <cstring>
#include <vector>

// wrap c++ tensor to external handle
namespace {

// based on modelMetaInfo create tensor
llaisys::tensor_t make_tensor(
    const LlaisysQwen2Meta &meta,
    llaisysDeviceType_t device,
    int device_id,
    const std::vector<size_t> &shape) {
    return llaisys::Tensor::create(shape, meta.dtype, device, device_id);
}

// based on dtype create tensor
llaisys::tensor_t make_tensor_dtype(
    llaisysDataType_t dtype,
    llaisysDeviceType_t device,
    int device_id,
    const std::vector<size_t> &shape) {
    return llaisys::Tensor::create(shape, dtype, device, device_id);
}

// set tensor data to zero
void zero_tensor(const llaisys::tensor_t &t) {
    CHECK_ARGUMENT(t->deviceType() == LLAISYS_DEVICE_CPU, "Zero: only CPU is supported.");
    std::memset(t->data(), 0, t->numel() * t->elementSize());
}

// wrap c++ tensor to external handle and record to handles list
llaisysTensor_t wrap_tensor(
    const llaisys::tensor_t &t,
    std::vector<LlaisysTensor *> &handles) {
    auto *h = new LlaisysTensor{t};
    handles.push_back(h);
    return h;
}

}

// Qwen2 Model Instance Structure
struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta;              // model meta info
    llaisysDeviceType_t device;         // device type 
    int device_id;                      // device id

    LlaisysQwen2Weights weights;        // all weight tensors handle
    std::vector<LlaisysTensor *> handles; // handles list for unified release

    // attn out bias
    std::vector<llaisys::tensor_t> attn_o_bias;   
    // MLP gate bias
    std::vector<llaisys::tensor_t> mlp_gate_bias; 
    // MLP up bias
    std::vector<llaisys::tensor_t> mlp_up_bias;   
    // MLP down bias
    std::vector<llaisys::tensor_t> mlp_down_bias; 
    // out bias
    llaisys::tensor_t out_bias;                  

    // KV cache
    std::vector<llaisys::tensor_t> k_cache;
    std::vector<llaisys::tensor_t> v_cache;

    size_t cache_len;
};

// init model weights, allocate memory and zero bias
static void init_weights(LlaisysQwen2Model *model) {
    const auto &m = model->meta;
    // input embedding
    model->weights.in_embed = wrap_tensor(
        make_tensor(m, model->device, model->device_id, {m.voc, m.hs}),
        model->handles);
    // output embedding
    model->weights.out_embed = wrap_tensor(
        make_tensor(m, model->device, model->device_id, {m.voc, m.hs}),
        model->handles);
    // output layer norm
    model->weights.out_norm_w = wrap_tensor(
        make_tensor(m, model->device, model->device_id, {m.hs}),
        model->handles);

    // allocate ptr array for each layer
    model->weights.attn_norm_w = new llaisysTensor_t[m.nlayer];
    model->weights.attn_q_w = new llaisysTensor_t[m.nlayer];
    model->weights.attn_q_b = new llaisysTensor_t[m.nlayer];
    model->weights.attn_k_w = new llaisysTensor_t[m.nlayer];
    model->weights.attn_k_b = new llaisysTensor_t[m.nlayer];
    model->weights.attn_v_w = new llaisysTensor_t[m.nlayer];
    model->weights.attn_v_b = new llaisysTensor_t[m.nlayer];
    model->weights.attn_o_w = new llaisysTensor_t[m.nlayer];
    model->weights.mlp_norm_w = new llaisysTensor_t[m.nlayer];
    model->weights.mlp_gate_w = new llaisysTensor_t[m.nlayer];
    model->weights.mlp_up_w = new llaisysTensor_t[m.nlayer];
    model->weights.mlp_down_w = new llaisysTensor_t[m.nlayer];

    // bias
    model->attn_o_bias.resize(m.nlayer);
    model->mlp_gate_bias.resize(m.nlayer);
    model->mlp_up_bias.resize(m.nlayer);
    model->mlp_down_bias.resize(m.nlayer);

    // for each layer, create its weight and bias tensor
    // first weight tensor
    for (size_t i = 0; i < m.nlayer; ++i) {
        // attn layer norm
        model->weights.attn_norm_w[i] = wrap_tensor(
            make_tensor(m, model->device, model->device_id, {m.hs}),
            model->handles);
        // Q/K/V/O proj weight and bias
        model->weights.attn_q_w[i] = wrap_tensor(
            make_tensor(m, model->device, model->device_id, {m.nh * m.dh, m.hs}),
            model->handles);
        model->weights.attn_q_b[i] = wrap_tensor(
            make_tensor(m, model->device, model->device_id, {m.nh * m.dh}),
            model->handles);
        model->weights.attn_k_w[i] = wrap_tensor(
            make_tensor(m, model->device, model->device_id, {m.nkvh * m.dh, m.hs}),
            model->handles);
        model->weights.attn_k_b[i] = wrap_tensor(
            make_tensor(m, model->device, model->device_id, {m.nkvh * m.dh}),
            model->handles);
        model->weights.attn_v_w[i] = wrap_tensor(
            make_tensor(m, model->device, model->device_id, {m.nkvh * m.dh, m.hs}),
            model->handles);
        model->weights.attn_v_b[i] = wrap_tensor(
            make_tensor(m, model->device, model->device_id, {m.nkvh * m.dh}),
            model->handles);
        model->weights.attn_o_w[i] = wrap_tensor(
            make_tensor(m, model->device, model->device_id, {m.hs, m.nh * m.dh}),
            model->handles);

        // MLP layer norm and gate/up/down weight
        model->weights.mlp_norm_w[i] = wrap_tensor(
            make_tensor(m, model->device, model->device_id, {m.hs}),
            model->handles);
        model->weights.mlp_gate_w[i] = wrap_tensor(
            make_tensor(m, model->device, model->device_id, {m.di, m.hs}),
            model->handles);
        model->weights.mlp_up_w[i] = wrap_tensor(
            make_tensor(m, model->device, model->device_id, {m.di, m.hs}),
            model->handles);
        model->weights.mlp_down_w[i] = wrap_tensor(
            make_tensor(m, model->device, model->device_id, {m.hs, m.di}),
            model->handles);

        // bias tensor
        model->attn_o_bias[i] = make_tensor(m, model->device, model->device_id, {m.hs});
        model->mlp_gate_bias[i] = make_tensor(m, model->device, model->device_id, {m.di});
        model->mlp_up_bias[i] = make_tensor(m, model->device, model->device_id, {m.di});
        model->mlp_down_bias[i] = make_tensor(m, model->device, model->device_id, {m.hs});

        // initialize bias tensor to 0
        zero_tensor(model->weights.attn_q_b[i]->tensor);
        zero_tensor(model->weights.attn_k_b[i]->tensor);
        zero_tensor(model->weights.attn_v_b[i]->tensor);
        zero_tensor(model->attn_o_bias[i]);
        zero_tensor(model->mlp_gate_bias[i]);
        zero_tensor(model->mlp_up_bias[i]);
        zero_tensor(model->mlp_down_bias[i]);
    }

    // output layerbias
    model->out_bias = make_tensor(m, model->device, model->device_id, {m.voc});
    zero_tensor(model->out_bias);
}

// initialize kv cache tensor
static void init_cache(LlaisysQwen2Model *model) {
    const auto &m = model->meta;
    model->k_cache.resize(m.nlayer);
    model->v_cache.resize(m.nlayer);
    for (size_t i = 0; i < m.nlayer; ++i) {
        // cache shape: [maxseq, num_heads, head_dim]
        model->k_cache[i] = make_tensor(m, model->device, model->device_id, {m.maxseq, m.nkvh, m.dh});
        model->v_cache[i] = make_tensor(m, model->device, model->device_id, {m.maxseq, m.nkvh, m.dh});
    }
    model->cache_len = 0;
}

// inference implementation
static int64_t infer_impl(LlaisysQwen2Model *model, const int64_t *token_ids, size_t ntoken) {
    CHECK_ARGUMENT(model != nullptr, "model is null");
    CHECK_ARGUMENT(token_ids != nullptr || ntoken == 0, "token_ids is null");
    CHECK_ARGUMENT(model->device == LLAISYS_DEVICE_CPU, "Only CPU device is supported.");

    if (ntoken == 0) {
        return model->meta.end_token;
    }

    // prefill
    if (ntoken > 1 || model->cache_len == 0) {
        model->cache_len = 0;
    }

    size_t seqlen = ntoken;
    size_t pos_offset = model->cache_len;

    // position ID [pos_offset, pos_offset + seqlen)
    std::vector<int64_t> pos_ids(seqlen);
    for (size_t i = 0; i < seqlen; ++i) {
        pos_ids[i] = static_cast<int64_t>(pos_offset + i);
    }

    // input token and position ID tensors
    auto input_ids = make_tensor_dtype(LLAISYS_DTYPE_I64, model->device, model->device_id, {seqlen});
    input_ids->load(token_ids);

    auto pos_tensor = make_tensor_dtype(LLAISYS_DTYPE_I64, model->device, model->device_id, {seqlen});
    pos_tensor->load(pos_ids.data());

    // embedding lookup
    auto x = make_tensor(model->meta, model->device, model->device_id, {seqlen, model->meta.hs});
    llaisys::ops::embedding(x, input_ids, model->weights.in_embed->tensor);

    // attn scale factor
    float scale = 1.0f / std::sqrt(static_cast<float>(model->meta.dh));

    // layer forward
    for (size_t i = 0; i < model->meta.nlayer; ++i) {
        // attn input norm
        auto x_norm = make_tensor(model->meta, model->device, model->device_id, {seqlen, model->meta.hs});
        llaisys::ops::rms_norm(x_norm, x, model->weights.attn_norm_w[i]->tensor, model->meta.epsilon);

        // Q/K/V linear proj
        auto q = make_tensor(model->meta, model->device, model->device_id, {seqlen, model->meta.nh * model->meta.dh});
        auto k = make_tensor(model->meta, model->device, model->device_id, {seqlen, model->meta.nkvh * model->meta.dh});
        auto v = make_tensor(model->meta, model->device, model->device_id, {seqlen, model->meta.nkvh * model->meta.dh});

        llaisys::ops::linear(q, x_norm, model->weights.attn_q_w[i]->tensor, model->weights.attn_q_b[i]->tensor);
        llaisys::ops::linear(k, x_norm, model->weights.attn_k_w[i]->tensor, model->weights.attn_k_b[i]->tensor);
        llaisys::ops::linear(v, x_norm, model->weights.attn_v_w[i]->tensor, model->weights.attn_v_b[i]->tensor);

        // transform to multi-head dim
        auto q_view = q->view({seqlen, model->meta.nh, model->meta.dh});
        auto k_view = k->view({seqlen, model->meta.nkvh, model->meta.dh});
        auto v_view = v->view({seqlen, model->meta.nkvh, model->meta.dh});

        // RoPE
        auto q_rope = make_tensor(model->meta, model->device, model->device_id, {seqlen, model->meta.nh, model->meta.dh});
        auto k_rope = make_tensor(model->meta, model->device, model->device_id, {seqlen, model->meta.nkvh, model->meta.dh});

        llaisys::ops::rope(q_rope, q_view, pos_tensor, model->meta.theta);
        llaisys::ops::rope(k_rope, k_view, pos_tensor, model->meta.theta);

        // write new k/v to cache
        auto k_cache_slice = model->k_cache[i]->slice(0, model->cache_len, model->cache_len + seqlen);
        auto v_cache_slice = model->v_cache[i]->slice(0, model->cache_len, model->cache_len + seqlen);

        llaisys::ops::rearrange(k_cache_slice, k_rope);
        llaisys::ops::rearrange(v_cache_slice, v_view);

        // get all history k/v
        size_t total_len = model->cache_len + seqlen;
        auto k_total = model->k_cache[i]->slice(0, 0, total_len);
        auto v_total = model->v_cache[i]->slice(0, 0, total_len);

        // self attn
        auto attn = make_tensor(model->meta, model->device, model->device_id, {seqlen, model->meta.nh, model->meta.dh});
        llaisys::ops::self_attention(attn, q_rope, k_total, v_total, scale);

        // attn out proj
        auto attn_flat = attn->view({seqlen, model->meta.nh * model->meta.dh});
        auto attn_proj = make_tensor(model->meta, model->device, model->device_id, {seqlen, model->meta.hs});
        llaisys::ops::linear(attn_proj, attn_flat, model->weights.attn_o_w[i]->tensor, model->attn_o_bias[i]);

        // first residual conn
        auto res1 = make_tensor(model->meta, model->device, model->device_id, {seqlen, model->meta.hs});
        llaisys::ops::add(res1, x, attn_proj);

        // MLP input norm
        auto x_norm2 = make_tensor(model->meta, model->device, model->device_id, {seqlen, model->meta.hs});
        llaisys::ops::rms_norm(x_norm2, res1, model->weights.mlp_norm_w[i]->tensor, model->meta.epsilon);

        // MLP gate and up proj
        auto gate = make_tensor(model->meta, model->device, model->device_id, {seqlen, model->meta.di});
        auto up = make_tensor(model->meta, model->device, model->device_id, {seqlen, model->meta.di});

        llaisys::ops::linear(gate, x_norm2, model->weights.mlp_gate_w[i]->tensor, model->mlp_gate_bias[i]);
        llaisys::ops::linear(up, x_norm2, model->weights.mlp_up_w[i]->tensor, model->mlp_up_bias[i]);

        // SwiGLU activate
        auto swiglu_out = make_tensor(model->meta, model->device, model->device_id, {seqlen, model->meta.di});
        llaisys::ops::swiglu(swiglu_out, gate, up);

        // MLP down proj
        auto down = make_tensor(model->meta, model->device, model->device_id, {seqlen, model->meta.hs});
        llaisys::ops::linear(down, swiglu_out, model->weights.mlp_down_w[i]->tensor, model->mlp_down_bias[i]);

        // second residual conn
        auto res2 = make_tensor(model->meta, model->device, model->device_id, {seqlen, model->meta.hs});
        llaisys::ops::add(res2, res1, down);

        x = res2;
    }

    // update cache len
    model->cache_len += seqlen;

    // final norm
    auto x_norm = make_tensor(model->meta, model->device, model->device_id, {seqlen, model->meta.hs});
    llaisys::ops::rms_norm(x_norm, x, model->weights.out_norm_w->tensor, model->meta.epsilon);

    // output proj to vocab
    auto logits = make_tensor(model->meta, model->device, model->device_id, {seqlen, model->meta.voc});
    llaisys::ops::linear(logits, x_norm, model->weights.out_embed->tensor, model->out_bias);

    // get last token logits and argmax
    auto last = logits->slice(0, seqlen - 1, seqlen)->view({model->meta.voc});
    auto max_idx = make_tensor_dtype(LLAISYS_DTYPE_I64, model->device, model->device_id, {1});
    auto max_val = make_tensor(model->meta, model->device, model->device_id, {1});
    llaisys::ops::argmax(max_idx, max_val, last);

    return reinterpret_cast<int64_t *>(max_idx->data())[0];
}

// C API wrapper
__C {

// ModelCreate: Check args, initialize weights and cache
struct LlaisysQwen2Model *llaisysQwen2ModelCreate(
    const LlaisysQwen2Meta *meta,
    llaisysDeviceType_t device,
    int *device_ids,
    int ndevice) {
    CHECK_ARGUMENT(meta != nullptr, "meta is null");
    CHECK_ARGUMENT(device == LLAISYS_DEVICE_CPU, "Only CPU device is supported.");
    CHECK_ARGUMENT(ndevice >= 1, "Invalid device count.");

    auto *model = new LlaisysQwen2Model();
    model->meta = *meta;
    model->device = device;
    model->device_id = device_ids ? device_ids[0] : 0;
    init_weights(model);
    init_cache(model);
    return model;
}

// ModelDestroy: Free weights and cache, all handles
void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
    if (!model) {
        return;
    }
    delete[] model->weights.attn_norm_w;
    delete[] model->weights.attn_q_w;
    delete[] model->weights.attn_q_b;
    delete[] model->weights.attn_k_w;
    delete[] model->weights.attn_k_b;
    delete[] model->weights.attn_v_w;
    delete[] model->weights.attn_v_b;
    delete[] model->weights.attn_o_w;
    delete[] model->weights.mlp_norm_w;
    delete[] model->weights.mlp_gate_w;
    delete[] model->weights.mlp_up_w;
    delete[] model->weights.mlp_down_w;

    for (auto *h : model->handles) {
        delete h;
    }
    delete model;
}

// ModelWeights: Get weights pointer for loading pretrained params
struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
    CHECK_ARGUMENT(model != nullptr, "model is null");
    return &model->weights;
}

// ModelInfer: Single interface for single token prediction
int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
    return infer_impl(model, token_ids, ntoken);
}

}
