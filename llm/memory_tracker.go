package llm

import (
	"runtime"
	"sync"
	"time"

	"github.com/ollama/ollama/discover"
)

// MemoryTracker monitors memory usage during pebbling operations
type MemoryTracker struct {
	isMonitoring bool
	startTime    time.Time
	samples      []MemorySample
	peakMemory   uint64
	mutex        sync.RWMutex
	stopChan     chan bool
}

// MemorySample represents a point-in-time memory measurement
type MemorySample struct {
	TimestampMS   int64  `json:"timestamp_ms"`
	AllocMB       uint64 `json:"alloc_mb"`
	SysMB         uint64 `json:"sys_mb"`
	HeapAllocMB   uint64 `json:"heap_alloc_mb"`
	HeapSysMB     uint64 `json:"heap_sys_mb"`
	StackInUseMB  uint64 `json:"stack_inuse_mb"`
	GCCount       uint32 `json:"gc_count"`
	GPUMemoryMB   uint64 `json:"gpu_memory_mb,omitempty"`
}

// NewMemoryTracker creates a new memory tracker
func NewMemoryTracker() *MemoryTracker {
	return &MemoryTracker{
		samples:  make([]MemorySample, 0, 1000), // Pre-allocate for 1000 samples
		stopChan: make(chan bool, 1),
	}
}

// StartMonitoring begins continuous memory monitoring
func (mt *MemoryTracker) StartMonitoring() {
	mt.mutex.Lock()
	if mt.isMonitoring {
		mt.mutex.Unlock()
		return
	}

	mt.isMonitoring = true
	mt.startTime = time.Now()
	mt.samples = mt.samples[:0] // Clear existing samples
	mt.peakMemory = 0
	mt.mutex.Unlock()

	// Start monitoring goroutine
	go mt.monitorLoop()
}

// StopMonitoring stops memory monitoring
func (mt *MemoryTracker) StopMonitoring() {
	mt.mutex.Lock()
	if !mt.isMonitoring {
		mt.mutex.Unlock()
		return
	}
	mt.isMonitoring = false
	mt.mutex.Unlock()

	// Signal stop
	select {
	case mt.stopChan <- true:
	default:
	}
}

// monitorLoop runs the continuous monitoring
func (mt *MemoryTracker) monitorLoop() {
	ticker := time.NewTicker(10 * time.Millisecond) // Sample every 10ms
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			mt.recordSample()
		case <-mt.stopChan:
			return
		}
	}
}

// recordSample takes a memory measurement
func (mt *MemoryTracker) recordSample() {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	sample := MemorySample{
		TimestampMS:  time.Since(mt.startTime).Milliseconds(),
		AllocMB:      bytesToMB(m.Alloc),
		SysMB:        bytesToMB(m.Sys),
		HeapAllocMB:  bytesToMB(m.HeapAlloc),
		HeapSysMB:    bytesToMB(m.HeapSys),
		StackInUseMB: bytesToMB(m.StackInuse),
		GCCount:      m.NumGC,
		GPUMemoryMB:  mt.getGPUMemoryUsage(),
	}

	mt.mutex.Lock()
	defer mt.mutex.Unlock()

	if mt.isMonitoring {
		mt.samples = append(mt.samples, sample)

		// Track peak memory
		totalMemory := sample.AllocMB + sample.GPUMemoryMB
		if totalMemory > mt.peakMemory {
			mt.peakMemory = totalMemory
		}
	}
}

// RecordSnapshot manually records a memory snapshot with context
func (mt *MemoryTracker) RecordSnapshot(contextID int, context interface{}) {
	sample := mt.getCurrentSample()
	sample.TimestampMS = time.Since(mt.startTime).Milliseconds()

	mt.mutex.Lock()
	defer mt.mutex.Unlock()

	if len(mt.samples) < cap(mt.samples) {
		mt.samples = append(mt.samples, sample)
	}
}

// getCurrentSample gets current memory stats
func (mt *MemoryTracker) getCurrentSample() MemorySample {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	return MemorySample{
		AllocMB:      bytesToMB(m.Alloc),
		SysMB:        bytesToMB(m.Sys),
		HeapAllocMB:  bytesToMB(m.HeapAlloc),
		HeapSysMB:    bytesToMB(m.HeapSys),
		StackInUseMB: bytesToMB(m.StackInuse),
		GCCount:      m.NumGC,
		GPUMemoryMB:  mt.getGPUMemoryUsage(),
	}
}

// getGPUMemoryUsage attempts to get GPU memory usage
func (mt *MemoryTracker) getGPUMemoryUsage() uint64 {
	// Try to get GPU memory from discover package
	gpus := discover.GetGPUInfo()
	if len(gpus) > 0 {
		totalUsed := uint64(0)
		for _, gpu := range gpus {
			used := gpu.TotalMemory - gpu.FreeMemory
			totalUsed += used
		}
		return bytesToMB(totalUsed)
	}
	return 0
}

// GetTrace returns the memory trace
func (mt *MemoryTracker) GetTrace() []MemorySample {
	mt.mutex.RLock()
	defer mt.mutex.RUnlock()

	// Return a copy to avoid race conditions
	trace := make([]MemorySample, len(mt.samples))
	copy(trace, mt.samples)
	return trace
}

// GetPeakMemory returns the peak memory usage
func (mt *MemoryTracker) GetPeakMemory() uint64 {
	mt.mutex.RLock()
	defer mt.mutex.RUnlock()
	return mt.peakMemory
}

// GetAverageMemory calculates average memory usage
func (mt *MemoryTracker) GetAverageMemory() uint64 {
	mt.mutex.RLock()
	defer mt.mutex.RUnlock()

	if len(mt.samples) == 0 {
		return 0
	}

	total := uint64(0)
	for _, sample := range mt.samples {
		total += sample.AllocMB + sample.GPUMemoryMB
	}

	return total / uint64(len(mt.samples))
}

// GetMemoryStats returns comprehensive memory statistics
func (mt *MemoryTracker) GetMemoryStats() MemoryStats {
	mt.mutex.RLock()
	defer mt.mutex.RUnlock()

	if len(mt.samples) == 0 {
		return MemoryStats{}
	}

	peak := uint64(0)
	total := uint64(0)
	minMem := uint64(^uint64(0)) // Max uint64
	maxMem := uint64(0)

	for _, sample := range mt.samples {
		mem := sample.AllocMB + sample.GPUMemoryMB
		total += mem

		if mem > peak {
			peak = mem
		}
		if mem > maxMem {
			maxMem = mem
		}
		if mem < minMem {
			minMem = mem
		}
	}

	avg := total / uint64(len(mt.samples))

	return MemoryStats{
		CurrentMemory:   mt.samples[len(mt.samples)-1].AllocMB + mt.samples[len(mt.samples)-1].GPUMemoryMB,
		MaxMemory:       maxMem,
		PeakMemory:      peak,
		AverageMemory:   avg,
		MinMemory:       minMem,
		SampleCount:     len(mt.samples),
	}
}

// ForceGC triggers garbage collection and waits for it to complete
func (mt *MemoryTracker) ForceGC() {
	runtime.GC()
	runtime.GC() // Call twice to ensure cleanup
	time.Sleep(10 * time.Millisecond) // Allow GC to complete
}

// Reset clears all tracking data
func (mt *MemoryTracker) Reset() {
	mt.mutex.Lock()
	defer mt.mutex.Unlock()

	mt.samples = mt.samples[:0]
	mt.peakMemory = 0
	mt.startTime = time.Now()
}

// bytesToMB converts bytes to megabytes
func bytesToMB(bytes uint64) uint64 {
	return bytes / (1024 * 1024)
}

// Enhanced MemoryStats with additional fields
type MemoryStats struct {
	CurrentMemory   uint64
	MaxMemory       uint64
	PeakMemory      uint64
	AverageMemory   uint64
	MinMemory       uint64
	SampleCount     int
	PebbledNodes    int // From existing struct
	CheckpointNodes int // From existing struct
	MemoryRatio     float64 // From existing struct
}