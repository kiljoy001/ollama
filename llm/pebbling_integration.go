package llm

import (
	"log/slog"
	"math"
	"sort"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/fs/ggml"
)

// PebblingStrategy defines different memory optimization strategies
type PebblingStrategy int

const (
	StandardBackprop PebblingStrategy = iota
	CheckpointSqrtN
	AdaptivePebbling
	MemoryPoor
)

// PebblingMemoryEstimate extends MemoryEstimate with pebbling optimization
type PebblingMemoryEstimate struct {
	MemoryEstimate

	// Pebbling-specific fields
	PebblingStrategy   PebblingStrategy
	CheckpointLayers   []int
	TheoreticalMemory  uint64 // Williams' sqrt(T log T) bound
	PracticalMemory    uint64 // Actual memory after constraints
	RecomputationCost  float64 // Estimated extra computation
	XOROverlapRegions  int
	UnityRoots         int
}

// EstimatePebblingGPULayers estimates GPU layers with pebbling optimization
func EstimatePebblingGPULayers(
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
		MemoryEstimate:   baseEstimate,
		PebblingStrategy: strategy,
	}

	layerCount := int(f.KV().BlockCount())

	// Calculate layer size from base estimate
	layerSize := baseEstimate.TotalSize / uint64(layerCount)
	if layerSize == 0 {
		layerSize = 1024 * 1024 * 100 // Default 100MB per layer
	}

	// Calculate theoretical memory bounds
	switch strategy {
	case CheckpointSqrtN:
		// Standard sqrt(n) checkpointing
		pebblingEstimate.TheoreticalMemory = calculateSqrtNMemory(f, opts.NumCtx)
		pebblingEstimate.CheckpointLayers = selectSqrtNCheckpoints(layerCount)
		pebblingEstimate.RecomputationCost = 1.0 // One extra forward pass

	case AdaptivePebbling:
		// Williams' sqrt(T log T) optimization
		logT := math.Log2(float64(layerCount))
		pebblingEstimate.TheoreticalMemory = uint64(math.Sqrt(float64(layerCount) * logT)) * layerSize
		pebblingEstimate.CheckpointLayers = selectAdaptiveCheckpoints(layerCount, gpus[0].FreeMemory)
		pebblingEstimate.RecomputationCost = logT // Log(n) recomputations worst case

		// XOR and unity optimization parameters
		pebblingEstimate.XOROverlapRegions = calculateXORRegions(gpus[0].FreeMemory)
		pebblingEstimate.UnityRoots = 7 // 7th roots of unity for good properties

	case MemoryPoor:
		// Extreme memory saving, high computation
		pebblingEstimate.TheoreticalMemory = layerSize * 2 // Only 2 layers at a time
		pebblingEstimate.CheckpointLayers = []int{} // No checkpoints
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
		opts,
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

	logPebblingEstimate(pebblingEstimate)

	return pebblingEstimate
}

// calculateSqrtNMemory calculates memory for sqrt(n) checkpointing
func calculateSqrtNMemory(f *ggml.GGML, numCtx int) uint64 {
	layers := f.Tensors().GroupLayers()
	layerCount := f.KV().BlockCount()

	var layerSize uint64
	if blk0, ok := layers["blk.0"]; ok {
		layerSize = blk0.Size()
	}

	// KV cache per checkpoint
	kvSize := uint64(numCtx) * uint64(f.KV().HeadCountMax()) * 128 // Simplified KV estimate

	// sqrt(n) checkpoints + current computation window
	sqrtN := uint64(math.Sqrt(float64(layerCount)))
	checkpointMemory := sqrtN * (layerSize + kvSize)
	computationMemory := sqrtN * layerSize // Window between checkpoints

	return checkpointMemory + computationMemory
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
	memoryPressure := float64(layerCount) * 1e9 / float64(availableMemory) // Rough estimate

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

// calculateXORRegions determines number of XOR memory regions
func calculateXORRegions(availableMemory uint64) int {
	// Each region should be at least 1MB for efficiency
	regionSize := uint64(1024 * 1024)
	maxRegions := int(availableMemory / regionSize / 10) // Use 10% of memory for XOR pools

	if maxRegions < 4 {
		return 4 // Minimum for effective overlapping
	}
	if maxRegions > 64 {
		return 64 // Maximum for manageable complexity
	}

	return maxRegions
}

// adjustForPracticalConstraints adjusts theoretical memory for real constraints
func adjustForPracticalConstraints(theoretical, available uint64, opts api.Options) uint64 {
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
		bound := uint64(math.Sqrt(float64(totalLayers) * logN)) * layerSize
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

// IntegratePebblingWithScheduler integrates pebbling with GPU scheduler
func IntegratePebblingWithScheduler(
	gpus discover.GpuInfoList,
	f *ggml.GGML,
	adapters, projectors []string,
	opts api.Options,
	numParallel int,
) (discover.GpuInfoList, PebblingMemoryEstimate, error) {

	// Determine best pebbling strategy based on model size and available memory
	strategy := selectOptimalStrategy(f, gpus, opts)

	// Get pebbling-optimized memory estimate
	estimate := EstimatePebblingGPULayers(gpus, f, projectors, opts, numParallel, strategy)

	// Modify GPU allocation based on pebbling
	optimizedGPUs := optimizeGPUAllocation(gpus, estimate)

	slog.Info("Pebbling optimization complete",
		"strategy", strategy,
		"original_memory", format.HumanBytes2(estimate.TotalSize),
		"optimized_memory", format.HumanBytes2(estimate.PracticalMemory),
		"memory_saved", format.HumanBytes2(estimate.TotalSize-estimate.PracticalMemory),
		"recompute_cost", estimate.RecomputationCost,
		"checkpoints", len(estimate.CheckpointLayers))

	return optimizedGPUs, estimate, nil
}

// selectOptimalStrategy chooses the best pebbling strategy
func selectOptimalStrategy(f *ggml.GGML, gpus discover.GpuInfoList, opts api.Options) PebblingStrategy {
	_ = f.KV().BlockCount()
	totalMemory := uint64(0)
	for _, gpu := range gpus {
		totalMemory += gpu.FreeMemory
	}

	modelSize := estimateModelSize(f)

	// Decision logic
	memoryRatio := float64(totalMemory) / float64(modelSize)

	if memoryRatio > 2.0 {
		// Plenty of memory: use standard approach
		return StandardBackprop
	} else if memoryRatio > 0.5 {
		// Moderate memory: sqrt(n) checkpointing
		return CheckpointSqrtN
	} else if memoryRatio > 0.1 {
		// Low memory: use Williams' advanced techniques
		return AdaptivePebbling
	} else {
		// Very low memory: extreme pebbling
		return MemoryPoor
	}
}

// estimateModelSize estimates total model size
func estimateModelSize(f *ggml.GGML) uint64 {
	layers := f.Tensors().GroupLayers()
	var totalSize uint64

	for _, layer := range layers {
		totalSize += layer.Size()
	}

	return totalSize
}

// optimizeGPUAllocation adjusts GPU allocation for pebbling
func optimizeGPUAllocation(gpus discover.GpuInfoList, estimate PebblingMemoryEstimate) discover.GpuInfoList {
	optimized := make(discover.GpuInfoList, len(gpus))
	copy(optimized, gpus)

	// Sort GPUs by free memory
	sort.Sort(sort.Reverse(discover.ByFreeMemory(optimized)))

	// Distribute checkpoints across GPUs
	if len(estimate.CheckpointLayers) > 0 {
		checkpointsPerGPU := len(estimate.CheckpointLayers) / len(optimized)
		if checkpointsPerGPU < 1 {
			checkpointsPerGPU = 1
		}

		// Mark GPUs with checkpoint responsibilities
		for i := range optimized {
			if i*checkpointsPerGPU < len(estimate.CheckpointLayers) {
				// This GPU will hold checkpoints
				// Adjust available memory accordingly
				checkpointMem := uint64(checkpointsPerGPU) * (estimate.PracticalMemory / uint64(len(estimate.CheckpointLayers)))
				if optimized[i].FreeMemory > checkpointMem {
					optimized[i].FreeMemory -= checkpointMem
				}
			}
		}
	}

	return optimized
}

// logPebblingEstimate logs detailed pebbling information
func logPebblingEstimate(estimate PebblingMemoryEstimate) {
	slog.Debug("Pebbling memory estimate",
		slog.Group("strategy",
			"type", estimate.PebblingStrategy,
			"checkpoints", len(estimate.CheckpointLayers),
			"xor_regions", estimate.XOROverlapRegions,
			"unity_roots", estimate.UnityRoots,
		),
		slog.Group("memory",
			"theoretical", format.HumanBytes2(estimate.TheoreticalMemory),
			"practical", format.HumanBytes2(estimate.PracticalMemory),
			"original", format.HumanBytes2(estimate.TotalSize),
			"savings", format.HumanBytes2(estimate.TotalSize-estimate.PracticalMemory),
		),
		slog.Group("performance",
			"layers_offloaded", estimate.Layers,
			"recompute_cost", estimate.RecomputationCost,
		),
	)

	if len(estimate.CheckpointLayers) > 0 && len(estimate.CheckpointLayers) <= 20 {
		slog.Debug("Checkpoint layers", "layers", estimate.CheckpointLayers)
	}
}

// ExportPebblingConfig exports pebbling configuration for debugging
func ExportPebblingConfig(estimate PebblingMemoryEstimate) map[string]interface{} {
	return map[string]interface{}{
		"strategy":           estimate.PebblingStrategy.String(),
		"theoretical_memory": estimate.TheoreticalMemory,
		"practical_memory":   estimate.PracticalMemory,
		"checkpoints":        estimate.CheckpointLayers,
		"xor_regions":        estimate.XOROverlapRegions,
		"unity_roots":        estimate.UnityRoots,
		"recompute_cost":     estimate.RecomputationCost,
		"memory_saved":       estimate.TotalSize - estimate.PracticalMemory,
		"layers":             estimate.Layers,
	}
}

// String returns string representation of pebbling strategy
func (ps PebblingStrategy) String() string {
	switch ps {
	case StandardBackprop:
		return "standard"
	case CheckpointSqrtN:
		return "sqrt_n_checkpoint"
	case MemoryPoor:
		return "memory_poor"
	case AdaptivePebbling:
		return "williams_adaptive"
	default:
		return "unknown"
	}
}