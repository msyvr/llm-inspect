## LLM inspect: interrogate open model layer activation patterns

### Overview

Extract neuron activations to identify which neurons encode some state.

### Quick Start

```bash
uv pip install torch transformers numpy matplotlib tqdm

uv run src/find_activations.py
```

### TL;DR

1. Load model
2. Register forward hook on a model layer
3. Extract activations for select prompts (eg: 8 uncertain + 8 certain)
4. Identify selective neurons using statistical analysis

### Output

The following plots are generated for visual inspection of the results:

1. State-specific activations across all neurons
2. Top selective neurons
3. Effect size distribution (Cohen's d)
4. Highest activation distribution
5. Activation pattern heatmap
6. Correlation map (top selective neurons)

### Activation Extraction

```python
class ActivationExtractor:
    def hook_fn(self, module, input, output):
        # Capture hidden states during forward pass
        hidden_states = output[0]  # (batch, seq, hidden_dim)
        self.activations = hidden_states.mean(dim=1)  # Average over sequence
```

This hooks into the transformer layer and collects the layer-specific activations during inference.

### Example

**Model**: GPT-2 Small

- 12 transformer layers
- 768 hidden dimensions (neurons per layer)
- 124M total parameters

**Activation Source**:

- Hook registered on `model.transformer.h[layer_idx]`
- Extracts hidden states: `(batch_size, sequence_length, 768)`
- Averages over sequence: `(batch_size, 768)`
