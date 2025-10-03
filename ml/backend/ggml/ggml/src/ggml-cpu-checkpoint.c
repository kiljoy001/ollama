#include "ggml-cpu-checkpoint.h"
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

// Checkpoint storage for CPU backend
typedef struct {
    void* data;
    size_t size;
} checkpoint_entry_t;

typedef struct {
    checkpoint_entry_t* checkpoints;
    int* checkpoint_layers;
    int num_checkpoints;
    int max_checkpoints;
    size_t total_memory;
    size_t memory_limit;
    int* lru_order;
} checkpoint_storage_t;

static checkpoint_storage_t g_checkpoint_storage = {0};

// Initialize checkpoint system
void ggml_cpu_checkpoint_init(size_t memory_limit_bytes, int num_layers, int strategy) {
    g_checkpoint_storage.memory_limit = memory_limit_bytes;
    g_checkpoint_storage.max_checkpoints = num_layers;
    g_checkpoint_storage.num_checkpoints = 0;
    g_checkpoint_storage.total_memory = 0;

    g_checkpoint_storage.checkpoints = (checkpoint_entry_t*)calloc(num_layers, sizeof(checkpoint_entry_t));
    g_checkpoint_storage.checkpoint_layers = (int*)calloc(num_layers, sizeof(int));
    g_checkpoint_storage.lru_order = (int*)calloc(num_layers, sizeof(int));

    for (int i = 0; i < num_layers; i++) {
        g_checkpoint_storage.checkpoint_layers[i] = -1;
        g_checkpoint_storage.lru_order[i] = i;
    }
}

// Check if layer should be checkpointed
bool ggml_cpu_should_checkpoint(int layer_idx) {
    // For CPU, checkpoint every sqrt(n) layers
    if (g_checkpoint_storage.max_checkpoints == 0) return false;

    int interval = (int)sqrt((double)g_checkpoint_storage.max_checkpoints);
    if (interval < 1) interval = 1;

    return (layer_idx % interval) == 0;
}

// Evict least recently used checkpoint
static void evict_lru_checkpoint(void) {
    if (g_checkpoint_storage.num_checkpoints == 0) return;

    int lru_idx = g_checkpoint_storage.lru_order[0];

    if (g_checkpoint_storage.checkpoints[lru_idx].data != NULL) {
        free(g_checkpoint_storage.checkpoints[lru_idx].data);
        g_checkpoint_storage.total_memory -= g_checkpoint_storage.checkpoints[lru_idx].size;
        g_checkpoint_storage.checkpoints[lru_idx].data = NULL;
        g_checkpoint_storage.checkpoints[lru_idx].size = 0;
        g_checkpoint_storage.checkpoint_layers[lru_idx] = -1;
        g_checkpoint_storage.num_checkpoints--;
    }

    // Shift LRU order
    for (int i = 0; i < g_checkpoint_storage.max_checkpoints - 1; i++) {
        g_checkpoint_storage.lru_order[i] = g_checkpoint_storage.lru_order[i + 1];
    }
}

// Save checkpoint
bool ggml_cpu_save_checkpoint(int layer_idx, const void* src, size_t size_bytes) {
    if (layer_idx >= g_checkpoint_storage.max_checkpoints) return false;

    // Evict if needed
    while (g_checkpoint_storage.total_memory + size_bytes > g_checkpoint_storage.memory_limit &&
           g_checkpoint_storage.num_checkpoints > 0) {
        evict_lru_checkpoint();
    }

    // Allocate and copy
    void* checkpoint_data = malloc(size_bytes);
    if (checkpoint_data == NULL) return false;

    memcpy(checkpoint_data, src, size_bytes);

    g_checkpoint_storage.checkpoints[layer_idx].data = checkpoint_data;
    g_checkpoint_storage.checkpoints[layer_idx].size = size_bytes;
    g_checkpoint_storage.checkpoint_layers[layer_idx] = layer_idx;
    g_checkpoint_storage.total_memory += size_bytes;
    g_checkpoint_storage.num_checkpoints++;

    // Update LRU (move to end)
    g_checkpoint_storage.lru_order[g_checkpoint_storage.max_checkpoints - 1] = layer_idx;

    return true;
}

// Restore checkpoint
bool ggml_cpu_restore_checkpoint(int layer_idx, void* dst, size_t size_bytes) {
    if (layer_idx >= g_checkpoint_storage.max_checkpoints) return false;
    if (g_checkpoint_storage.checkpoints[layer_idx].data == NULL) return false;
    if (g_checkpoint_storage.checkpoints[layer_idx].size != size_bytes) return false;

    memcpy(dst, g_checkpoint_storage.checkpoints[layer_idx].data, size_bytes);

    // Update LRU (move to end)
    for (int i = 0; i < g_checkpoint_storage.max_checkpoints; i++) {
        if (g_checkpoint_storage.lru_order[i] == layer_idx) {
            for (int j = i; j < g_checkpoint_storage.max_checkpoints - 1; j++) {
                g_checkpoint_storage.lru_order[j] = g_checkpoint_storage.lru_order[j + 1];
            }
            g_checkpoint_storage.lru_order[g_checkpoint_storage.max_checkpoints - 1] = layer_idx;
            break;
        }
    }

    return true;
}

// Get memory stats
size_t ggml_cpu_checkpoint_memory_used(void) {
    return g_checkpoint_storage.total_memory;
}

int ggml_cpu_checkpoint_count(void) {
    return g_checkpoint_storage.num_checkpoints;
}

// Cleanup
void ggml_cpu_checkpoint_cleanup(void) {
    for (int i = 0; i < g_checkpoint_storage.max_checkpoints; i++) {
        if (g_checkpoint_storage.checkpoints[i].data != NULL) {
            free(g_checkpoint_storage.checkpoints[i].data);
        }
    }

    free(g_checkpoint_storage.checkpoints);
    free(g_checkpoint_storage.checkpoint_layers);
    free(g_checkpoint_storage.lru_order);

    memset(&g_checkpoint_storage, 0, sizeof(checkpoint_storage_t));
}
