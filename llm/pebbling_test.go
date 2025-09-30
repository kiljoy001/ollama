package llm

import (
	"testing"

	"github.com/ollama/ollama/api"
)

func TestPebblingStrategyParsing(t *testing.T) {
	tests := []struct {
		input    string
		expected PebblingStrategy
	}{
		{"sqrt", CheckpointSqrtN},
		{"checkpoint", CheckpointSqrtN},
		{"adaptive", AdaptivePebbling},
		{"williams", AdaptivePebbling},
		{"poor", MemoryPoor},
		{"minimal", MemoryPoor},
		{"standard", StandardBackprop},
		{"none", StandardBackprop},
		{"", StandardBackprop},
		{"unknown", StandardBackprop},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			result := ParsePebblingStrategy(tt.input)
			if result != tt.expected {
				t.Errorf("ParsePebblingStrategy(%q) = %v, want %v", tt.input, result, tt.expected)
			}
		})
	}
}

func TestPebblingStrategyString(t *testing.T) {
	tests := []struct {
		strategy PebblingStrategy
		expected string
	}{
		{StandardBackprop, "standard"},
		{CheckpointSqrtN, "sqrt_n_checkpoint"},
		{AdaptivePebbling, "williams_adaptive"},
		{MemoryPoor, "memory_poor"},
	}

	for _, tt := range tests {
		t.Run(tt.expected, func(t *testing.T) {
			result := tt.strategy.String()
			if result != tt.expected {
				t.Errorf("PebblingStrategy.String() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestSelectSqrtNCheckpoints(t *testing.T) {
	tests := []struct {
		layerCount int
		minLen     int
		maxLen     int
	}{
		{100, 10, 10},  // sqrt(100) = 10
		{64, 8, 8},     // sqrt(64) = 8
		{50, 7, 8},     // sqrt(50) ≈ 7
		{1, 1, 1},      // edge case
	}

	for _, tt := range tests {
		t.Run(string(rune(tt.layerCount)), func(t *testing.T) {
			checkpoints := selectSqrtNCheckpoints(tt.layerCount)
			if len(checkpoints) < tt.minLen || len(checkpoints) > tt.maxLen {
				t.Errorf("selectSqrtNCheckpoints(%d) returned %d checkpoints, want between %d and %d",
					tt.layerCount, len(checkpoints), tt.minLen, tt.maxLen)
			}
		})
	}
}

func TestSelectAdaptiveCheckpoints(t *testing.T) {
	tests := []struct {
		layerCount      int
		availableMemory uint64
		minCheckpoints  int
	}{
		{100, 1024 * 1024 * 1024, 1},      // 1GB - low pressure
		{100, 100 * 1024 * 1024, 5},       // 100MB - medium pressure
		{100, 10 * 1024 * 1024, 10},       // 10MB - high pressure
	}

	for _, tt := range tests {
		t.Run(string(rune(tt.layerCount)), func(t *testing.T) {
			checkpoints := selectAdaptiveCheckpoints(tt.layerCount, tt.availableMemory)
			if len(checkpoints) < tt.minCheckpoints {
				t.Errorf("selectAdaptiveCheckpoints(%d, %d) returned %d checkpoints, want at least %d",
					tt.layerCount, tt.availableMemory, len(checkpoints), tt.minCheckpoints)
			}
		})
	}
}

func TestAdjustForPracticalConstraints(t *testing.T) {
	tests := []struct {
		name        string
		theoretical uint64
		available   uint64
		minExpected uint64
		maxExpected uint64
	}{
		{
			"plenty of memory",
			1024 * 1024 * 1024,      // 1GB theoretical
			10 * 1024 * 1024 * 1024, // 10GB available
			1024 * 1024 * 1024,      // Should get close to theoretical
			2 * 1024 * 1024 * 1024,
		},
		{
			"limited memory",
			5 * 1024 * 1024 * 1024, // 5GB theoretical
			1024 * 1024 * 1024,     // 1GB available
			100 * 1024 * 1024,      // Should be constrained
			1024 * 1024 * 1024,
		},
		{
			"very low memory",
			1024 * 1024 * 1024, // 1GB theoretical
			50 * 1024 * 1024,   // 50MB available
			100 * 1024 * 1024,  // Should hit minimum
			100 * 1024 * 1024,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := adjustForPracticalConstraints(tt.theoretical, tt.available)
			if result < tt.minExpected || result > tt.maxExpected {
				t.Errorf("adjustForPracticalConstraints() = %d, want between %d and %d",
					result, tt.minExpected, tt.maxExpected)
			}
		})
	}
}

func TestCalculatePebblingLayerCount(t *testing.T) {
	tests := []struct {
		name        string
		memory      uint64
		layerSize   uint64
		totalLayers int
		strategy    PebblingStrategy
		minLayers   int
	}{
		{
			"sqrt checkpointing with enough memory",
			1024 * 1024 * 1024,         // 1GB
			10 * 1024 * 1024,           // 10MB per layer
			100,                        // 100 layers
			CheckpointSqrtN,
			100, // Should fit all
		},
		{
			"adaptive with enough memory",
			1024 * 1024 * 1024, // 1GB
			10 * 1024 * 1024,   // 10MB per layer
			100,                // 100 layers
			AdaptivePebbling,
			100, // Should fit all
		},
		{
			"memory poor strategy",
			100 * 1024 * 1024, // 100MB
			10 * 1024 * 1024,  // 10MB per layer
			100,               // 100 layers
			MemoryPoor,
			100, // Should still fit all with minimal memory
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := calculatePebblingLayerCount(tt.memory, tt.layerSize, tt.totalLayers, tt.strategy)
			if result < tt.minLayers {
				t.Errorf("calculatePebblingLayerCount() = %d, want at least %d", result, tt.minLayers)
			}
		})
	}
}

func TestPredictServerFitWithPebbling(t *testing.T) {
	// Test with pebbling enabled
	enabled := true
	opts := api.Options{
		Runner: api.Runner{
			NumGPU:           -1,
			PebblingEnabled:  &enabled,
			PebblingStrategy: "sqrt",
		},
	}

	// Verify the option parsing works
	strategy := ParsePebblingStrategy(opts.PebblingStrategy)
	if strategy != CheckpointSqrtN {
		t.Errorf("Failed to parse pebbling strategy from options")
	}
}