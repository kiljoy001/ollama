#pragma once

#include <cstddef>
#include <cmath>

// Initialize checkpointing with memory budget and strategy
// strategy: 0=none, 1=sqrt(n), 2=adaptive, 3=minimal
extern "C" void ggml_cuda_checkpoint_init(size_t memory_limit_bytes, int num_layers, int strategy);

// Check if a layer should be checkpointed
extern "C" bool ggml_cuda_should_checkpoint(int layer_idx);

// Save checkpoint for a layer
extern "C" bool ggml_cuda_save_checkpoint(int layer_idx, const void* src, size_t size_bytes);

// Restore checkpoint for a layer
extern "C" bool ggml_cuda_restore_checkpoint(int layer_idx, void* dst, size_t size_bytes);

// Free checkpoint for a layer
extern "C" void ggml_cuda_free_checkpoint(int layer_idx);

// Get checkpoint statistics
extern "C" size_t ggml_cuda_checkpoint_memory_used();
extern "C" int ggml_cuda_checkpoint_count();

// Cleanup
extern "C" void ggml_cuda_checkpoint_cleanup();