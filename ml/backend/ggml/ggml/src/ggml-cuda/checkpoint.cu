#if !defined(GGML_USE_HIP)
#include "checkpoint.cuh"
#include "common.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <unordered_map>

// Checkpoint storage for gradient checkpointing
struct CheckpointStorage {
    std::unordered_map<int, void*> checkpoints;
    std::unordered_map<int, size_t> checkpoint_sizes;
    std::vector<int> checkpoint_layers;
    size_t total_memory_used;
    size_t memory_limit;

    CheckpointStorage() : total_memory_used(0), memory_limit(0) {}

    ~CheckpointStorage() {
        clear();
    }

    void clear() {
        for (auto& pair : checkpoints) {
            if (pair.second) {
                cudaFree(pair.second);
            }
        }
        checkpoints.clear();
        checkpoint_sizes.clear();
        checkpoint_layers.clear();
        total_memory_used = 0;
    }
};

static CheckpointStorage g_checkpoint_storage;

// Initialize checkpointing with memory budget
extern "C" void ggml_cuda_checkpoint_init(size_t memory_limit_bytes, int num_layers, int strategy) {
    g_checkpoint_storage.clear();
    g_checkpoint_storage.memory_limit = memory_limit_bytes;

    // Select checkpoint layers based on strategy
    // 0 = none, 1 = sqrt(n), 2 = adaptive, 3 = minimal
    switch (strategy) {
        case 1: { // sqrt(n) checkpointing
            int sqrt_n = (int)sqrt((double)num_layers);
            if (sqrt_n < 1) sqrt_n = 1;
            for (int i = 0; i < num_layers; i += sqrt_n) {
                g_checkpoint_storage.checkpoint_layers.push_back(i);
            }
            break;
        }
        case 2: { // adaptive - checkpoint every log(n) layers
            int log_n = (int)(log((double)num_layers) / log(2.0));
            if (log_n < 2) log_n = 2;
            for (int i = 0; i < num_layers; i += log_n) {
                g_checkpoint_storage.checkpoint_layers.push_back(i);
            }
            break;
        }
        case 3: { // minimal - only checkpoint every 10th layer
            for (int i = 0; i < num_layers; i += 10) {
                g_checkpoint_storage.checkpoint_layers.push_back(i);
            }
            break;
        }
        default:
            break; // No checkpointing
    }
}

// Check if a layer should be checkpointed
extern "C" bool ggml_cuda_should_checkpoint(int layer_idx) {
    for (int cp_layer : g_checkpoint_storage.checkpoint_layers) {
        if (cp_layer == layer_idx) {
            return true;
        }
    }
    return false;
}

// Save checkpoint for a layer
extern "C" bool ggml_cuda_save_checkpoint(int layer_idx, const void* src, size_t size_bytes) {
    // Check memory limit
    if (g_checkpoint_storage.total_memory_used + size_bytes > g_checkpoint_storage.memory_limit) {
        // Need to evict old checkpoints
        // Simple strategy: evict oldest checkpoint not in current window
        if (!g_checkpoint_storage.checkpoints.empty()) {
            auto it = g_checkpoint_storage.checkpoints.begin();
            if (it->second) {
                cudaFree(it->second);
                g_checkpoint_storage.total_memory_used -= g_checkpoint_storage.checkpoint_sizes[it->first];
            }
            g_checkpoint_storage.checkpoints.erase(it);
        }
    }

    // Allocate checkpoint memory
    void* checkpoint_ptr = nullptr;
    cudaError_t err = cudaMalloc(&checkpoint_ptr, size_bytes);
    if (err != cudaSuccess || !checkpoint_ptr) {
        return false;
    }

    // Copy data to checkpoint
    err = cudaMemcpy(checkpoint_ptr, src, size_bytes, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        cudaFree(checkpoint_ptr);
        return false;
    }

    // Store checkpoint
    if (g_checkpoint_storage.checkpoints.count(layer_idx)) {
        // Free old checkpoint if exists
        cudaFree(g_checkpoint_storage.checkpoints[layer_idx]);
        g_checkpoint_storage.total_memory_used -= g_checkpoint_storage.checkpoint_sizes[layer_idx];
    }

    g_checkpoint_storage.checkpoints[layer_idx] = checkpoint_ptr;
    g_checkpoint_storage.checkpoint_sizes[layer_idx] = size_bytes;
    g_checkpoint_storage.total_memory_used += size_bytes;

    return true;
}

// Restore checkpoint for a layer
extern "C" bool ggml_cuda_restore_checkpoint(int layer_idx, void* dst, size_t size_bytes) {
    auto it = g_checkpoint_storage.checkpoints.find(layer_idx);
    if (it == g_checkpoint_storage.checkpoints.end()) {
        return false; // Checkpoint not found
    }

    size_t checkpoint_size = g_checkpoint_storage.checkpoint_sizes[layer_idx];
    if (checkpoint_size != size_bytes) {
        return false; // Size mismatch
    }

    // Copy checkpoint data back
    cudaError_t err = cudaMemcpy(dst, it->second, size_bytes, cudaMemcpyDeviceToDevice);
    return (err == cudaSuccess);
}

// Free checkpoint for a layer (no longer needed)
extern "C" void ggml_cuda_free_checkpoint(int layer_idx) {
    auto it = g_checkpoint_storage.checkpoints.find(layer_idx);
    if (it != g_checkpoint_storage.checkpoints.end()) {
        if (it->second) {
            cudaFree(it->second);
            g_checkpoint_storage.total_memory_used -= g_checkpoint_storage.checkpoint_sizes[layer_idx];
        }
        g_checkpoint_storage.checkpoints.erase(it);
        g_checkpoint_storage.checkpoint_sizes.erase(layer_idx);
    }
}

// Get current checkpoint memory usage
extern "C" size_t ggml_cuda_checkpoint_memory_used() {
    return g_checkpoint_storage.total_memory_used;
}

// Get number of active checkpoints
extern "C" int ggml_cuda_checkpoint_count() {
    return (int)g_checkpoint_storage.checkpoints.size();
}

// Cleanup all checkpoints
extern "C" void ggml_cuda_checkpoint_cleanup() {
    g_checkpoint_storage.clear();
}

#endif // !GGML_USE_HIP