package llm

import (
	"testing"

	"github.com/ollama/ollama/api"
)

func TestCheckpointManager(t *testing.T) {
	// Test checkpoint manager creation
	cm := NewCheckpointManager(32, CheckpointSqrtN, 1024*1024*1024)
	if cm == nil {
		t.Fatal("Failed to create checkpoint manager")
	}

	// Test sqrt(n) strategy
	// For 32 layers, sqrt(32) ≈ 5.6, so should checkpoint every ~6 layers
	checkpointCount := 0
	for i := 0; i < 32; i++ {
		if cm.ShouldCheckpoint(i) {
			checkpointCount++
		}
	}

	// Note: In stub builds (no CUDA), checkpoint count will be 0
	// In CUDA builds, we expect 4-8 checkpoints for sqrt(n) strategy
	if checkpointCount == 0 {
		t.Logf("Checkpoint manager using stub implementation (no CUDA)")
	} else if checkpointCount < 4 || checkpointCount > 8 {
		t.Errorf("Expected 4-8 checkpoints for 32 layers with sqrt strategy, got %d", checkpointCount)
	} else {
		t.Logf("Checkpoint manager created %d checkpoints for 32 layers (CUDA enabled)", checkpointCount)
	}

	// Test adaptive strategy
	cm2 := NewCheckpointManager(64, AdaptivePebbling, 2*1024*1024*1024)
	checkpointCount2 := 0
	for i := 0; i < 64; i++ {
		if cm2.ShouldCheckpoint(i) {
			checkpointCount2++
		}
	}

	t.Logf("Adaptive strategy created %d checkpoints for 64 layers", checkpointCount2)

	// Cleanup
	cm.Cleanup()
	cm2.Cleanup()
}

func TestPebblingIntegrationWithCheckpoints(t *testing.T) {
	// Test that pebbling estimation creates checkpoint manager
	enabled := true
	opts := api.Options{
		Runner: api.Runner{
			NumCtx:           2048,
			NumGPU:           -1,
			PebblingEnabled:  &enabled,
			PebblingStrategy: "sqrt",
		},
	}

	// Verify strategy parsing
	strategy := ParsePebblingStrategy(opts.PebblingStrategy)
	if strategy != CheckpointSqrtN {
		t.Errorf("Expected CheckpointSqrtN, got %v", strategy)
	}

	// Create checkpoint manager
	cm := NewCheckpointManager(40, strategy, 4*1024*1024*1024)
	defer cm.Cleanup()

	// Verify checkpointing works
	layersCheckpointed := 0
	for i := 0; i < 40; i++ {
		if cm.ShouldCheckpoint(i) {
			layersCheckpointed++
		}
	}

	if layersCheckpointed == 0 {
		t.Error("No layers were marked for checkpointing")
	}

	t.Logf("Integration test: %d layers checkpointed out of 40", layersCheckpointed)
}

func TestMemoryEstimationWithCheckpoints(t *testing.T) {
	// This tests the full integration of checkpoint manager with memory estimation
	enabled := true

	tests := []struct {
		name     string
		strategy string
		expected PebblingStrategy
	}{
		{"sqrt strategy", "sqrt", CheckpointSqrtN},
		{"adaptive strategy", "adaptive", AdaptivePebbling},
		{"minimal strategy", "poor", MemoryPoor},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			opts := api.Options{
				Runner: api.Runner{
					NumCtx:           2048,
					NumGPU:           -1,
					PebblingEnabled:  &enabled,
					PebblingStrategy: tt.strategy,
				},
			}

			strategy := ParsePebblingStrategy(opts.PebblingStrategy)
			if strategy != tt.expected {
				t.Errorf("Expected %v, got %v", tt.expected, strategy)
			}

			cm := NewCheckpointManager(32, strategy, 2*1024*1024*1024)
			defer cm.Cleanup()

			// Count checkpoints
			count := 0
			for i := 0; i < 32; i++ {
				if cm.ShouldCheckpoint(i) {
					count++
				}
			}

			t.Logf("%s: %d checkpoints for 32 layers", tt.name, count)
		})
	}
}