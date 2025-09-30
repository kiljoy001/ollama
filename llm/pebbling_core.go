package llm

import (
	"fmt"
	"log/slog"
	"math"
)

// ComputationDAG represents a directed acyclic graph of computations
type ComputationDAG struct {
	Nodes           []ComputationNode
	LayerCount      int
	TotalMemory     uint64
	CheckpointNodes []int
}

// ComputationNode represents a single computation in the DAG
type ComputationNode struct {
	ID           int
	Name         string
	MemorySize   uint64
	ComputeCost  uint64
	Dependencies []int
	Layer        int
	Computed     bool
	Value        []byte
}

// PebblingMemoryManager handles memory optimization during computation
type PebblingMemoryManager struct {
	strategy      PebblingStrategy
	maxMemory     uint64
	currentMemory uint64
	dag           *ComputationDAG
	pebbledNodes  map[int]bool
	checkpoints   map[int]bool
	nodeData      map[int][]byte
}

// NewPebblingMemoryManager creates a new memory manager
func NewPebblingMemoryManager(strategy PebblingStrategy, maxMemory uint64) *PebblingMemoryManager {
	return &PebblingMemoryManager{
		strategy:     strategy,
		maxMemory:    maxMemory,
		pebbledNodes: make(map[int]bool),
		checkpoints:  make(map[int]bool),
		nodeData:     make(map[int][]byte),
	}
}

// SelectCheckpoints chooses which nodes to checkpoint based on strategy
func (pmm *PebblingMemoryManager) SelectCheckpoints(dag *ComputationDAG) {
	switch pmm.strategy {
	case CheckpointSqrtN:
		pmm.selectSqrtNCheckpoints(dag)
	case AdaptivePebbling:
		pmm.selectAdaptiveCheckpoints(dag)
	case MemoryPoor:
		// No checkpoints for memory-poor strategy
		dag.CheckpointNodes = []int{}
	default:
		pmm.selectSqrtNCheckpoints(dag)
	}
}

// selectSqrtNCheckpoints implements sqrt(n) checkpointing
func (pmm *PebblingMemoryManager) selectSqrtNCheckpoints(dag *ComputationDAG) {
	sqrtN := int(math.Sqrt(float64(dag.LayerCount)))
	if sqrtN < 1 {
		sqrtN = 1
	}

	checkpoints := []int{}
	for i := 0; i < dag.LayerCount; i += sqrtN {
		if i < len(dag.Nodes) {
			checkpoints = append(checkpoints, i)
			pmm.checkpoints[i] = true
		}
	}

	dag.CheckpointNodes = checkpoints
	slog.Debug("Selected sqrt(n) checkpoints", "count", len(checkpoints), "interval", sqrtN)
}

// selectAdaptiveCheckpoints implements adaptive checkpointing based on memory pressure
func (pmm *PebblingMemoryManager) selectAdaptiveCheckpoints(dag *ComputationDAG) {
	// Estimate memory pressure
	totalMemoryNeeded := uint64(0)
	for _, node := range dag.Nodes {
		totalMemoryNeeded += node.MemorySize
	}

	memoryPressure := float64(totalMemoryNeeded) / float64(pmm.maxMemory)

	var checkpointInterval int
	if memoryPressure > 10 {
		// High pressure: more frequent checkpoints
		checkpointInterval = int(math.Sqrt(float64(dag.LayerCount)) / 2)
	} else if memoryPressure > 2 {
		// Medium pressure: sqrt(n) checkpoints
		checkpointInterval = int(math.Sqrt(float64(dag.LayerCount)))
	} else {
		// Low pressure: fewer checkpoints
		checkpointInterval = int(math.Sqrt(float64(dag.LayerCount)) * 2)
	}

	if checkpointInterval < 1 {
		checkpointInterval = 1
	}

	checkpoints := []int{}
	for i := 0; i < dag.LayerCount; i += checkpointInterval {
		if i < len(dag.Nodes) {
			checkpoints = append(checkpoints, i)
			pmm.checkpoints[i] = true
		}
	}

	dag.CheckpointNodes = checkpoints
	slog.Debug("Selected adaptive checkpoints",
		"count", len(checkpoints),
		"interval", checkpointInterval,
		"memory_pressure", memoryPressure)
}

// PebbleNode marks a node as pebbled (computed and stored)
func (pmm *PebblingMemoryManager) PebbleNode(nodeID int) error {
	if nodeID >= len(pmm.dag.Nodes) {
		return fmt.Errorf("node ID %d out of range", nodeID)
	}

	node := &pmm.dag.Nodes[nodeID]

	// Check if dependencies are satisfied
	for _, depID := range node.Dependencies {
		if !pmm.pebbledNodes[depID] && !pmm.checkpoints[depID] {
			return fmt.Errorf("dependency %d not satisfied for node %d", depID, nodeID)
		}
	}

	// Simulate computation
	node.Value = pmm.computeNodeValue(node)
	node.Computed = true
	pmm.pebbledNodes[nodeID] = true
	pmm.nodeData[nodeID] = node.Value

	// Update memory usage
	pmm.currentMemory += node.MemorySize

	slog.Debug("Pebbled node", "id", nodeID, "memory_used", node.MemorySize, "total_memory", pmm.currentMemory)
	return nil
}

// ComputeNode computes a node value, using checkpoints and recomputation as needed
func (pmm *PebblingMemoryManager) ComputeNode(nodeID int) error {
	if nodeID >= len(pmm.dag.Nodes) {
		return fmt.Errorf("node ID %d out of range", nodeID)
	}

	node := &pmm.dag.Nodes[nodeID]

	// If already computed, return
	if node.Computed {
		return nil
	}

	// Ensure dependencies are computed
	for _, depID := range node.Dependencies {
		if err := pmm.ComputeNode(depID); err != nil {
			return fmt.Errorf("failed to compute dependency %d: %w", depID, err)
		}
	}

	// Compute the node
	node.Value = pmm.computeNodeValue(node)
	node.Computed = true
	pmm.currentMemory += node.MemorySize

	// If this is a checkpoint, keep it in memory
	if pmm.checkpoints[nodeID] {
		pmm.nodeData[nodeID] = node.Value
		slog.Debug("Computed checkpoint node", "id", nodeID)
	} else if pmm.strategy == MemoryPoor {
		// For memory-poor strategy, immediately discard non-checkpoint nodes
		if nodeID > 0 { // Keep the last node
			delete(pmm.nodeData, nodeID)
			pmm.currentMemory -= node.MemorySize
		}
	}

	return nil
}

// computeNodeValue simulates actual node computation
func (pmm *PebblingMemoryManager) computeNodeValue(node *ComputationNode) []byte {
	// Simplified computation: combine dependency values
	result := make([]byte, 1024) // 1KB result

	if len(node.Dependencies) == 0 {
		// Leaf node: deterministic value based on ID
		for i := range result {
			result[i] = byte((node.ID + i) % 256)
		}
	} else {
		// Combine dependencies
		for _, depID := range node.Dependencies {
			if depData, exists := pmm.nodeData[depID]; exists {
				for i := range result {
					if i < len(depData) {
						result[i] ^= depData[i]
					}
				}
			}
		}
	}

	return result
}

// GetMemoryStats returns current memory usage statistics
func (pmm *PebblingMemoryManager) GetMemoryStats() PebblingMemoryStats {
	pebbledCount := len(pmm.pebbledNodes)
	checkpointCount := len(pmm.checkpoints)

	return PebblingMemoryStats{
		CurrentMemory:   pmm.currentMemory,
		MaxMemory:       pmm.maxMemory,
		PebbledNodes:    pebbledCount,
		CheckpointNodes: checkpointCount,
		MemoryRatio:     float64(pmm.currentMemory) / float64(pmm.maxMemory),
	}
}

// PebblingMemoryStats holds memory usage statistics for pebbling
type PebblingMemoryStats struct {
	CurrentMemory   uint64
	MaxMemory       uint64
	PebbledNodes    int
	CheckpointNodes int
	MemoryRatio     float64
}

// EstimatePebblingMemory estimates memory usage for a given strategy
func EstimatePebblingMemory(f interface{}, strategy PebblingStrategy, layerCount uint64) uint64 {
	layerSize := uint64(1024 * 1024) // 1MB per layer assumption

	switch strategy {
	case StandardBackprop:
		return layerCount * layerSize
	case CheckpointSqrtN:
		sqrtN := uint64(math.Sqrt(float64(layerCount)))
		return (sqrtN + sqrtN) * layerSize // Checkpoints + computation window
	case AdaptivePebbling:
		logN := math.Log2(float64(layerCount))
		return uint64(math.Sqrt(float64(layerCount)*logN)) * layerSize
	case MemoryPoor:
		return 2 * layerSize // Just 2 layers at most
	default:
		return layerCount * layerSize
	}
}