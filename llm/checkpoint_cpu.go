// +build !cuda

package llm

/*
#cgo CFLAGS: -I${SRCDIR}/../ml/backend/ggml/ggml/src
#cgo LDFLAGS: -lm

#include "ggml-cpu-checkpoint.h"
*/
import "C"
import "unsafe"

// CPU checkpoint manager (works for Intel GPU via CPU backend)
type CPUCheckpointManager struct {
	numLayers   int
	strategy    PebblingStrategy
	memoryLimit uint64
	enabled     bool
}

func NewCPUCheckpointManager(numLayers int, strategy PebblingStrategy, memoryLimit uint64) *CPUCheckpointManager {
	cm := &CPUCheckpointManager{
		numLayers:   numLayers,
		strategy:    strategy,
		memoryLimit: memoryLimit,
		enabled:     true,
	}

	C.ggml_cpu_checkpoint_init(C.size_t(memoryLimit), C.int(numLayers), C.int(strategy))
	return cm
}

func (cm *CPUCheckpointManager) ShouldCheckpoint(layerIdx int) bool {
	if !cm.enabled {
		return false
	}
	return bool(C.ggml_cpu_should_checkpoint(C.int(layerIdx)))
}

func (cm *CPUCheckpointManager) SaveCheckpoint(layerIdx int, data []byte) bool {
	if !cm.enabled || len(data) == 0 {
		return false
	}
	return bool(C.ggml_cpu_save_checkpoint(
		C.int(layerIdx),
		unsafe.Pointer(&data[0]),
		C.size_t(len(data)),
	))
}

func (cm *CPUCheckpointManager) RestoreCheckpoint(layerIdx int, data []byte) bool {
	if !cm.enabled || len(data) == 0 {
		return false
	}
	return bool(C.ggml_cpu_restore_checkpoint(
		C.int(layerIdx),
		unsafe.Pointer(&data[0]),
		C.size_t(len(data)),
	))
}

func (cm *CPUCheckpointManager) MemoryUsed() uint64 {
	return uint64(C.ggml_cpu_checkpoint_memory_used())
}

func (cm *CPUCheckpointManager) CheckpointCount() int {
	return int(C.ggml_cpu_checkpoint_count())
}

func (cm *CPUCheckpointManager) Cleanup() {
	if cm.enabled {
		C.ggml_cpu_checkpoint_cleanup()
		cm.enabled = false
	}
}

func (cm *CPUCheckpointManager) Enable() {
	cm.enabled = true
}

func (cm *CPUCheckpointManager) Disable() {
	cm.enabled = false
}
