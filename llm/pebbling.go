package llm

import (
	"log/slog"
	"math"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/fs/ggml"
)

// PebblingStrategy defines different memory optimization strategies
type PebblingStrategy int

const (
	StandardBackprop PebblingStrategy = iota
	CheckpointSqrtN                   // sqrt(n) checkpointing
	AdaptivePebbling                  // Williams' sqrt(T log T) algorithm
	MemoryPoor                        // Aggressive O(log n) reduction
)

// String returns string representation of pebbling strategy
func (ps PebblingStrategy) String() string {
	switch ps {
	case StandardBackprop:
		return "standard"
	case CheckpointSqrtN:
		return "sqrt_n_checkpoint"
	case AdaptivePebbling:
		return "williams_adaptive"
	case MemoryPoor:
		return "memory_poor"
	default:
		return "unknown"
	}
}

// ParsePebblingStrategy converts string to PebblingStrategy enum
func ParsePebblingStrategy(strategy string) PebblingStrategy {
	switch strategy {
	case "sqrt", "checkpoint":
		return CheckpointSqrtN
	case "adaptive", "williams":
		return AdaptivePebbling
	case "poor", "minimal":
		return MemoryPoor
	case "standard", "none", "":
		return StandardBackprop
	default:
		slog.Warn("Unknown pebbling strategy, using standard", "strategy", strategy)
		return StandardBackprop
	}
}

// PebblingMemoryEstimate extends MemoryEstimate with pebbling optimization
type PebblingMemoryEstimate struct {
	MemoryEstimate

	// Pebbling-specific fields
	Strategy           PebblingStrategy
	CheckpointLayers   []int
	TheoreticalMemory  uint64            // Theoretical memory bound
	PracticalMemory    uint64            // Actual memory after constraints
	RecomputationCost  float64           // Estimated extra computation
	CheckpointManager  *CheckpointManager // Active checkpoint manager
}

// EstimateWithPebbling calculates memory estimate with pebbling optimization
func EstimateWithPebbling(
	gpus []discover.GpuInfo,
	f *ggml.GGML,
	projectors []string,
	opts api.Options,
	numParallel int,
	strategy PebblingStrategy,
) PebblingMemoryEstimate {
	// Start with standard estimation
	baseEstimate := estimateGPULayers(gpus, f, projectors, opts, numParallel)

	// Enhance with pebbling optimization
	pebblingEstimate := PebblingMemoryEstimate{
		MemoryEstimate: baseEstimate,
		Strategy:       strategy,
	}

	layerCount := int(f.KV().BlockCount())
	if layerCount == 0 {
		return pebblingEstimate
	}

	// Calculate layer size from base estimate
	layerSize := baseEstimate.TotalSize / uint64(layerCount)
	if layerSize == 0 {
		layerSize = 1024 * 1024 * 100 // Default 100MB per layer
	}

	// Calculate theoretical memory bounds and checkpoints
	switch strategy {
	case CheckpointSqrtN:
		// Standard sqrt(n) checkpointing
		sqrtN := uint64(math.Sqrt(float64(layerCount)))
		pebblingEstimate.TheoreticalMemory = sqrtN * layerSize * 2 // Checkpoints + computation window
		pebblingEstimate.CheckpointLayers = selectSqrtNCheckpoints(layerCount)
		pebblingEstimate.RecomputationCost = 1.0 // One extra forward pass

	case AdaptivePebbling:
		// Williams' sqrt(T log T) optimization
		logT := math.Log2(float64(layerCount))
		pebblingEstimate.TheoreticalMemory = uint64(math.Sqrt(float64(layerCount)*logT)) * layerSize
		pebblingEstimate.CheckpointLayers = selectAdaptiveCheckpoints(layerCount, gpus[0].FreeMemory)
		pebblingEstimate.RecomputationCost = logT // Log(n) recomputations worst case

	case MemoryPoor:
		// Extreme memory saving, high computation
		pebblingEstimate.TheoreticalMemory = layerSize * 2 // Only 2 layers at a time
		pebblingEstimate.CheckpointLayers = []int{}        // No checkpoints
		pebblingEstimate.RecomputationCost = float64(layerCount) // O(n) recomputations

	default:
		// Standard (no pebbling optimization)
		pebblingEstimate.TheoreticalMemory = baseEstimate.TotalSize
		pebblingEstimate.RecomputationCost = 0
	}

	// Adjust for practical constraints
	pebblingEstimate.PracticalMemory = adjustForPracticalConstraints(
		pebblingEstimate.TheoreticalMemory,
		gpus[0].FreeMemory,
	)

	// Recalculate layer distribution with pebbling
	pebblingEstimate.Layers = calculatePebblingLayerCount(
		pebblingEstimate.PracticalMemory,
		layerSize,
		layerCount,
		strategy,
	)

	// Update VRAM size to reflect pebbling savings
	pebblingEstimate.VRAMSize = pebblingEstimate.PracticalMemory

	// Create checkpoint manager if not using standard backprop
	if strategy != StandardBackprop && gpus[0].Library != "cpu" {
		pebblingEstimate.CheckpointManager = NewCheckpointManager(
			layerCount,
			strategy,
			pebblingEstimate.PracticalMemory,
		)
	}

	slog.Info("Pebbling memory estimate",
		"strategy", strategy.String(),
		"theoretical", format.HumanBytes2(pebblingEstimate.TheoreticalMemory),
		"practical", format.HumanBytes2(pebblingEstimate.PracticalMemory),
		"original", format.HumanBytes2(baseEstimate.TotalSize),
		"savings", format.HumanBytes2(baseEstimate.TotalSize-pebblingEstimate.PracticalMemory),
		"recompute_cost", pebblingEstimate.RecomputationCost,
		"checkpoints", len(pebblingEstimate.CheckpointLayers))

	return pebblingEstimate
}

// selectSqrtNCheckpoints selects checkpoint layers for sqrt(n) strategy
func selectSqrtNCheckpoints(layerCount int) []int {
	sqrtN := int(math.Sqrt(float64(layerCount)))
	if sqrtN < 1 {
		sqrtN = 1
	}

	checkpoints := []int{}
	for i := 0; i < layerCount; i += sqrtN {
		checkpoints = append(checkpoints, i)
	}

	return checkpoints
}

// selectAdaptiveCheckpoints selects checkpoints based on available memory
func selectAdaptiveCheckpoints(layerCount int, availableMemory uint64) []int {
	// Use memory pressure to determine checkpoint density
	memoryPressure := float64(layerCount) * 1e9 / float64(availableMemory)

	var checkpointInterval int
	if memoryPressure > 10 {
		// High pressure: more checkpoints
		checkpointInterval = int(math.Sqrt(float64(layerCount)) / 2)
	} else if memoryPressure > 5 {
		// Medium pressure: sqrt(n) checkpoints
		checkpointInterval = int(math.Sqrt(float64(layerCount)))
	} else {
		// Low pressure: fewer checkpoints
		checkpointInterval = int(math.Sqrt(float64(layerCount)) * 2)
	}

	if checkpointInterval < 1 {
		checkpointInterval = 1
	}

	checkpoints := []int{}
	for i := 0; i < layerCount; i += checkpointInterval {
		checkpoints = append(checkpoints, i)
	}

	return checkpoints
}

// adjustForPracticalConstraints adjusts theoretical memory for real constraints
func adjustForPracticalConstraints(theoretical, available uint64) uint64 {
	// Can't use more than available
	practical := min(theoretical, available*9/10) // Leave 10% buffer

	// Ensure minimum for operation
	minRequired := uint64(1024 * 1024 * 100) // 100MB minimum
	if practical < minRequired {
		practical = minRequired
	}

	// Align to GPU memory boundaries (typically 256MB chunks)
	alignment := uint64(256 * 1024 * 1024)
	practical = (practical / alignment) * alignment

	return practical
}

// calculatePebblingLayerCount calculates layers that fit with pebbling
func calculatePebblingLayerCount(memory, layerSize uint64, totalLayers int, strategy PebblingStrategy) int {
	switch strategy {
	case CheckpointSqrtN:
		// Memory holds sqrt(n) checkpoints + computation window
		sqrtN := int(math.Sqrt(float64(totalLayers)))
		checkpointMem := uint64(sqrtN) * layerSize
		if memory > checkpointMem {
			return totalLayers // Can fit all with checkpointing
		}
		return int(memory / layerSize)

	case AdaptivePebbling:
		// Williams' bound: can always fit if memory >= sqrt(n * log n) * layer_size
		logN := math.Log2(float64(totalLayers))
		bound := uint64(math.Sqrt(float64(totalLayers)*logN)) * layerSize
		if memory >= bound {
			return totalLayers
		}
		return int(memory / layerSize)

	case MemoryPoor:
		// Can process all layers with minimal memory
		if memory >= layerSize*2 {
			return totalLayers
		}
		return 0

	default:
		return int(memory / layerSize)
	}
}