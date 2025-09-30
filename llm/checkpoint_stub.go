// +build !cuda

package llm

import "unsafe"

// CheckpointManager stub for non-CUDA builds
type CheckpointManager struct {
	numLayers   int
	strategy    PebblingStrategy
	memoryLimit uint64
	enabled     bool
}

// NewCheckpointManager creates a no-op checkpoint manager
func NewCheckpointManager(numLayers int, strategy PebblingStrategy, memoryLimit uint64) *CheckpointManager {
	return &CheckpointManager{
		numLayers:   numLayers,
		strategy:    strategy,
		memoryLimit: memoryLimit,
		enabled:     false,
	}
}

// ShouldCheckpoint always returns false in stub
func (cm *CheckpointManager) ShouldCheckpoint(layerIdx int) bool {
	return false
}

// SaveCheckpoint no-op
func (cm *CheckpointManager) SaveCheckpoint(layerIdx int, data unsafe.Pointer, sizeBytes uint64) bool {
	return false
}

// RestoreCheckpoint no-op
func (cm *CheckpointManager) RestoreCheckpoint(layerIdx int, data unsafe.Pointer, sizeBytes uint64) bool {
	return false
}

// FreeCheckpoint no-op
func (cm *CheckpointManager) FreeCheckpoint(layerIdx int) {
}

// MemoryUsed returns 0
func (cm *CheckpointManager) MemoryUsed() uint64 {
	return 0
}

// CheckpointCount returns 0
func (cm *CheckpointManager) CheckpointCount() int {
	return 0
}

// Cleanup no-op
func (cm *CheckpointManager) Cleanup() {
}

// SetEnabled no-op
func (cm *CheckpointManager) SetEnabled(enabled bool) {
}