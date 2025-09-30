// +build cuda

package llm

/*
#cgo CFLAGS: -I${SRCDIR}/../ml/backend/ggml/ggml/include
#cgo LDFLAGS: -L${SRCDIR}/../ml/backend/ggml/ggml/build/src -lggml -lcudart

// Forward declarations for checkpoint functions
extern void ggml_cuda_checkpoint_init(size_t memory_limit_bytes, int num_layers, int strategy);
extern int ggml_cuda_should_checkpoint(int layer_idx);
extern int ggml_cuda_save_checkpoint(int layer_idx, const void* src, size_t size_bytes);
extern int ggml_cuda_restore_checkpoint(int layer_idx, void* dst, size_t size_bytes);
extern void ggml_cuda_free_checkpoint(int layer_idx);
extern size_t ggml_cuda_checkpoint_memory_used();
extern int ggml_cuda_checkpoint_count();
extern void ggml_cuda_checkpoint_cleanup();
*/
import "C"
import "unsafe"

// CheckpointManager manages CUDA gradient checkpointing
type CheckpointManager struct {
	numLayers     int
	strategy      PebblingStrategy
	memoryLimit   uint64
	enabled       bool
	checkpointMap map[int]bool
}

// NewCheckpointManager creates a new checkpoint manager
func NewCheckpointManager(numLayers int, strategy PebblingStrategy, memoryLimit uint64) *CheckpointManager {
	cm := &CheckpointManager{
		numLayers:     numLayers,
		strategy:      strategy,
		memoryLimit:   memoryLimit,
		enabled:       true,
		checkpointMap: make(map[int]bool),
	}

	// Convert strategy to C int
	var cStrategy C.int
	switch strategy {
	case CheckpointSqrtN:
		cStrategy = 1
	case AdaptivePebbling:
		cStrategy = 2
	case MemoryPoor:
		cStrategy = 3
	default:
		cStrategy = 0
	}

	// Initialize CUDA checkpointing
	C.ggml_cuda_checkpoint_init(C.size_t(memoryLimit), C.int(numLayers), cStrategy)

	return cm
}

// ShouldCheckpoint checks if a layer should be checkpointed
func (cm *CheckpointManager) ShouldCheckpoint(layerIdx int) bool {
	if !cm.enabled {
		return false
	}
	return C.ggml_cuda_should_checkpoint(C.int(layerIdx)) != 0
}

// SaveCheckpoint saves a checkpoint for a layer
func (cm *CheckpointManager) SaveCheckpoint(layerIdx int, data unsafe.Pointer, sizeBytes uint64) bool {
	if !cm.enabled {
		return false
	}
	result := C.ggml_cuda_save_checkpoint(C.int(layerIdx), data, C.size_t(sizeBytes))
	if result != 0 {
		cm.checkpointMap[layerIdx] = true
		return true
	}
	return false
}

// RestoreCheckpoint restores a checkpoint for a layer
func (cm *CheckpointManager) RestoreCheckpoint(layerIdx int, data unsafe.Pointer, sizeBytes uint64) bool {
	if !cm.enabled {
		return false
	}
	return C.ggml_cuda_restore_checkpoint(C.int(layerIdx), data, C.size_t(sizeBytes)) != 0
}

// FreeCheckpoint frees a checkpoint for a layer
func (cm *CheckpointManager) FreeCheckpoint(layerIdx int) {
	if !cm.enabled {
		return
	}
	C.ggml_cuda_free_checkpoint(C.int(layerIdx))
	delete(cm.checkpointMap, layerIdx)
}

// MemoryUsed returns current checkpoint memory usage
func (cm *CheckpointManager) MemoryUsed() uint64 {
	return uint64(C.ggml_cuda_checkpoint_memory_used())
}

// CheckpointCount returns number of active checkpoints
func (cm *CheckpointManager) CheckpointCount() int {
	return int(C.ggml_cuda_checkpoint_count())
}

// Cleanup frees all checkpoints
func (cm *CheckpointManager) Cleanup() {
	C.ggml_cuda_checkpoint_cleanup()
	cm.checkpointMap = make(map[int]bool)
}

// SetEnabled enables or disables checkpointing
func (cm *CheckpointManager) SetEnabled(enabled bool) {
	cm.enabled = enabled
	if !enabled {
		cm.Cleanup()
	}
}