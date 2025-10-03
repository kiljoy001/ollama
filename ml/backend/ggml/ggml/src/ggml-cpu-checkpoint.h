#pragma once

#include <stddef.h>
#include <stdbool.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

void ggml_cpu_checkpoint_init(size_t memory_limit_bytes, int num_layers, int strategy);
bool ggml_cpu_should_checkpoint(int layer_idx);
bool ggml_cpu_save_checkpoint(int layer_idx, const void* src, size_t size_bytes);
bool ggml_cpu_restore_checkpoint(int layer_idx, void* dst, size_t size_bytes);
size_t ggml_cpu_checkpoint_memory_used(void);
int ggml_cpu_checkpoint_count(void);
void ggml_cpu_checkpoint_cleanup(void);

#ifdef __cplusplus
}
#endif
