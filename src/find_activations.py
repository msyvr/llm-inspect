import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Configuration - LLM & target layer
MODEL_NAME = "gpt2"  # 124M parameters, runs fast on M2
TARGET_LAYER = 6     # Middle layer (GPT-2 has 12 layers, 0-11)
MAX_LENGTH = 50      # Max tokens for generation

# Configuration - target state
TARGET_STATE = "uncertainty"

# Configuration - plots
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14

# ============================================================================
# PROMPTS - certainty/uncertainty
# ============================================================================

HIGH_UNCERTAINTY_PROMPTS = [
    "Will NVIDIA's stock price go up or down tomorrow?",
    "Will it be sunny in Vancouver exactly 30 days from now?",
    "Who will win the next French Open?",
    "How many grains of sand are in the Sahara Desert?",
    "What's the exact population of Paris right now?",
    "Will AGI agents run the world by 2075?",
    "What will I want to do tomorrow?",
    "How many thoughts will I have today?",
]

LOW_UNCERTAINTY_PROMPTS = [
    "What is 2 + 2?",
    "What's the capital of France?",
    "How many sides does a triangle have?",
    "What color is the sky on a clear day?",
    "What's the boiling point of water at sea level?",
    "Who wrote Romeo and Juliet?",
    "What year did World War II end?",
    "What planet do we live on?",
]

# ============================================================================
# ACTIVATION EXTRACTION
# ============================================================================

class ActivationExtractor:
    """Extract activations from specific layer during forward pass."""
    
    def __init__(self, model, layer_idx: int):
        self.model = model
        self.layer_idx = layer_idx
        self.activations = None
        self.hook_handle = None
        
    def hook_fn(self, module, input, output):
        """Hook function to capture activations."""
        # output is a tuple, first element is the hidden states
        hidden_states = output[0]  # Shape: (batch, seq_len, hidden_dim)
        # Mean over sequence length gives per-sample activation
        self.activations = hidden_states.mean(dim=1).detach().cpu()  # (batch, hidden_dim)
        
    def register_hook(self):
        """Register the hook on target layer."""
        target_layer = self.model.transformer.h[self.layer_idx]
        self.hook_handle = target_layer.register_forward_hook(self.hook_fn)
        
    def remove_hook(self):
        """Remove the hook."""
        if self.hook_handle:
            self.hook_handle.remove()
            
    def get_activations(self, prompt: str, tokenizer) -> np.ndarray:
        """Get activations for a prompt."""
        inputs = tokenizer(prompt, return_tensors="pt", max_length=MAX_LENGTH, 
                          truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward pass (hook captures activations)
        with torch.no_grad():
            _ = self.model(**inputs)
            
        return self.activations.numpy()[0]

# ============================================================================
# DATA GENERATION/COLLECTION
# ============================================================================

def load_model_and_tokenizer(model_name: str = MODEL_NAME):
    """Load model and tokenizer."""
    if model_name.startswith("gpt"):
        print("\nLoading GPT-2 Small (124M parameters)...")
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
    else:
        print("Model name not recognized")
    model.to(device)
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded on {device}")
    print(f"  - Parameters: 124M")
    print(f"  - Layers: 12")
    print(f"  - Hidden dim: 768")
    print(f"  - Target layer: {TARGET_LAYER}")
    
    return model, tokenizer

def collect_experimental_data(model, tokenizer) -> Tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Collect REAL activation data for high and low uncertainty prompts.
    
    Returns:
        high_uncertainty_activations: (n_samples, n_neurons)
        low_uncertainty_activations: (n_samples, n_neurons)
        all_results: list of result dictionaries
    """
    extractor = ActivationExtractor(model, TARGET_LAYER)
    extractor.register_hook()
    
    print("\n[1/2] Collecting HIGH uncertainty activations...")
    high_results = []
    for prompt in tqdm(HIGH_UNCERTAINTY_PROMPTS):
        activations = extractor.get_activations(prompt, tokenizer)
        high_results.append({
            'prompt': prompt,
            'activations': activations
        })
    
    print("\n[2/2] Collecting LOW uncertainty activations...")
    low_results = []
    for prompt in tqdm(LOW_UNCERTAINTY_PROMPTS):
        activations = extractor.get_activations(prompt, tokenizer)
        low_results.append({
            'prompt': prompt,
            'activations': activations
        })
    
    extractor.remove_hook()
    
    high_activations = np.array([r['activations'] for r in high_results])
    low_activations = np.array([r['activations'] for r in low_results])
    
    print(f"\nCollected activations:")
    print(f"  - Shape: {high_activations.shape}")
    print(f"  - High uncertainty samples: {len(high_results)}")
    print(f"  - Low uncertainty samples: {len(low_results)}")
    
    return high_activations, low_activations, high_results + low_results

# ============================================================================
# ANALYSIS
# ============================================================================

def identify_uncertainty_neurons(high_act: np.ndarray, low_act: np.ndarray, 
                                top_k: int = 20) -> np.ndarray:
    """Identify neurons most strongly associated with uncertainty."""
    high_mean = np.mean(high_act, axis=0)
    low_mean = np.mean(low_act, axis=0)
    diff = high_mean - low_mean
    top_neurons = np.argsort(np.abs(diff))[-top_k:][::-1]
    return top_neurons

def compute_effect_sizes(high_act: np.ndarray, low_act: np.ndarray) -> np.ndarray:
    """Compute Cohen's d effect size for each neuron."""
    high_mean = np.mean(high_act, axis=0)
    low_mean = np.mean(low_act, axis=0)
    high_std = np.std(high_act, axis=0)
    low_std = np.std(low_act, axis=0)
    pooled_std = np.sqrt((high_std**2 + low_std**2) / 2)
    cohen_d = (high_mean - low_mean) / (pooled_std + 1e-8)
    return cohen_d

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(high_act: np.ndarray, low_act: np.ndarray, 
                top_neurons: np.ndarray, all_results: list[dict]):
    """Create clean diagnostic plots."""
    
    num_neurons = high_act.shape[1]
    color_high = '#2ecc71'
    color_low = '#3498db'
    color_accent = '#e74c3c'
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(f'LLM inspect for {TARGET_STATE}: activations from {MODEL_NAME}', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Plot 1: Neuron selectivity
    ax = axes[0, 0]
    diff = np.mean(high_act, axis=0) - np.mean(low_act, axis=0)
    ax.bar(range(len(diff)), diff, alpha=0.3, color='gray', width=1.0, linewidth=0)
    ax.bar(top_neurons[:20], diff[top_neurons[:20]], 
           color=color_accent, alpha=0.8, width=1.0, linewidth=0)
    ax.set_xlabel('Neuron index')
    ax.set_ylabel('Activation delta (high - low)')
    ax.set_title('Neuron selectivity', fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-', alpha=0.3)
    ax.set_xlim(-10, num_neurons + 10)
    
    # Plot 2: Top 10 neurons
    ax = axes[0, 1]
    top_10 = top_neurons[:10]
    x = np.arange(len(top_10))
    width = 0.35
    high_vals = np.mean(high_act[:, top_10], axis=0)
    low_vals = np.mean(low_act[:, top_10], axis=0)
    ax.bar(x - width/2, high_vals, width, label=f'High {TARGET_STATE}', 
           color=color_high, alpha=0.8, linewidth=0)
    ax.bar(x + width/2, low_vals, width, label=f'Low {TARGET_STATE}', 
           color=color_low, alpha=0.8, linewidth=0)
    ax.set_xlabel('Top neurons')
    ax.set_ylabel('Mean activation')
    ax.set_title('Top 10 neurons', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{n}' for n in top_10], fontsize=8)
    ax.legend(frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Plot 3: Effect sizes
    ax = axes[0, 2]
    effect_sizes = compute_effect_sizes(high_act, low_act)
    counts, bins, patches = ax.hist(effect_sizes, bins=40, alpha=0.7, 
                                    color='gray', edgecolor='none')
    for i, patch in enumerate(patches):
        if bins[i] > 0.8 or bins[i] < -0.8:
            patch.set_facecolor(color_accent)
            patch.set_alpha(0.8)
    top_effect = effect_sizes[top_neurons[0]]
    ax.axvline(top_effect, color=color_accent, linestyle='--', linewidth=2,
               label=f'Top neuron (d={top_effect:.2f})')
    ax.set_xlabel("Cohen's d")
    ax.set_ylabel('Count')
    ax.set_title('Effect Sizes', fontweight='bold')
    ax.legend(frameon=False, fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Plot 4: Best neuron distribution
    ax = axes[1, 0]
    best_neuron = top_neurons[0]
    parts = ax.violinplot([high_act[:, best_neuron], low_act[:, best_neuron]],
                          positions=[0, 1], widths=0.6, showmeans=True, 
                          showextrema=True)
    parts['bodies'][0].set_facecolor(color_high)
    parts['bodies'][0].set_alpha(0.7)
    parts['bodies'][1].set_facecolor(color_low)
    parts['bodies'][1].set_alpha(0.7)
    for pc in parts['bodies']:
        pc.set_edgecolor('black')
        pc.set_linewidth(1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f'High\n{TARGET_STATE}', f'Low\n{TARGET_STATE}'])
    ax.set_ylabel('Activation')
    ax.set_title(f'Best neuron (N{best_neuron})', fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Plot 5: Activation heatmap
    ax = axes[1, 1]
    # Find interesting window around top neurons
    top_neuron_idx = top_neurons[0]
    window_start = max(0, top_neuron_idx - 30)
    window_end = min(num_neurons, top_neuron_idx + 30)
    window = slice(window_start, window_end)
    
    all_act = np.vstack([high_act[:, window], low_act[:, window]])
    im = ax.imshow(all_act, aspect='auto', cmap='viridis', 
                   interpolation='nearest')
    ax.set_xlabel('Neuron index')
    ax.set_ylabel('Sample')
    ax.set_title(f'Activation pattern (N{window_start}-{window_end})', fontweight='bold')
    
    n_ticks = 5
    tick_positions = np.linspace(0, window_end - window_start - 1, n_ticks)
    tick_labels = np.linspace(window_start, window_end - 1, n_ticks).astype(int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.axhline(y=len(high_act) - 0.5, color='white', linewidth=2, linestyle='--')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Activation', rotation=270, labelpad=15)
    
    # Plot 6: Correlation
    ax = axes[1, 2]
    top_5 = top_neurons[:5]
    corr = np.corrcoef(high_act[:, top_5].T)
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    for i in range(len(top_5)):
        for j in range(len(top_5)):
            text = ax.text(j, i, f'{corr[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    ax.set_xticks(range(len(top_5)))
    ax.set_yticks(range(len(top_5)))
    ax.set_xticklabels([f'{n}' for n in top_5])
    ax.set_yticklabels([f'{n}' for n in top_5])
    ax.set_title('Top 5 Neuron correlation', fontweight='bold')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.savefig(f'results/analysis_{TARGET_STATE}_{MODEL_NAME}.png', 
                dpi=150, bbox_inches='tight', facecolor='white')
    print("\n✓ Saved: uncertainty_analysis_gpt2.png")
    
    return fig

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

def print_summary(high_act: np.ndarray, low_act: np.ndarray, 
                 top_neurons: np.ndarray, all_results: list[dict]):
    """Print summary statistics."""
    print("\n" + "="*70)
    print("EXPERIMENTAL SUMMARY")
    print("="*70)
    
    print(f"\nData collected:")
    print(f"  - High uncertainty samples: {len(high_act)}")
    print(f"  - Low uncertainty samples: {len(low_act)}")
    print(f"  - Total neurons analyzed: {high_act.shape[1]}")
    print(f"  - Layer: {TARGET_LAYER}")
    
    print(f"\nTop 5 uncertainty neurons (by activation difference):")
    for i, neuron_idx in enumerate(top_neurons[:5], 1):
        high_mean = np.mean(high_act[:, neuron_idx])
        low_mean = np.mean(low_act[:, neuron_idx])
        effect_size = compute_effect_sizes(high_act, low_act)[neuron_idx]
        print(f"  {i}. Neuron {neuron_idx}: "
              f"High={high_mean:.3f}, Low={low_mean:.3f}, "
              f"Δ={high_mean-low_mean:.3f}, d={effect_size:.2f}")
    
    print(f"\nExample prompts:")
    print(f"  HIGH: {all_results[0]['prompt']}")
    print(f"  LOW:  {all_results[-1]['prompt']}")
    print("\n" + "="*70)

# ============================================================================
# TECHNICAL ISSUES
# ============================================================================

def analyze_technical_issues(high_act: np.ndarray, low_act: np.ndarray, 
                            top_neurons: np.ndarray):
    """Identify potential technical issues."""
    print("\n" + "="*70)
    print("TECHNICAL ISSUES")
    print("="*70)
    
    issues = []
    
    # Sample size
    if len(high_act) < 30:
        issues.append(f"Small sample size: {len(high_act)} per condition (need 30+)")
    
    # Effect sizes
    effect_sizes = compute_effect_sizes(high_act, low_act)
    weak_effects = np.mean(np.abs(effect_sizes) < 0.2)
    if weak_effects > 0.9:
        issues.append(f"Weak effects: {weak_effects*100:.0f}% neurons show d < 0.2")
    
    # Correlation
    top_5 = top_neurons[:5]
    corr_matrix = np.corrcoef(high_act[:, top_5].T)
    high_corr = np.mean(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])) > 0.8
    if high_corr:
        issues.append("High correlation among top neurons (may be redundant)")
    
    issues.append("Need validation: ablation studies to test causality")
    issues.append("Single layer only: check other layers for distributed patterns")
    
    if issues:
        for issue in issues:
            print(f"\n{issue}")
    else:
        print("\nNo major issues detected")
    
    print("\n" + "="*70)

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run the complete experiment."""
    print("="*70)
    print("MECHANISTIC INTERPRETABILITY: REAL ACTIVATIONS")
    print("Model: GPT-2 Small (124M parameters)")
    print("Target: Uncertainty Detection")
    print("="*70)

    # Load model
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    
    # Collect data
    print("\n[1/5] Collecting activation data...")
    high_act, low_act, all_results = collect_experimental_data(model, tokenizer)
    
    # Identify neurons
    print("\n[2/5] Identifying uncertainty neurons...")
    top_neurons = identify_uncertainty_neurons(high_act, low_act, top_k=20)
    
    # Visualize
    print("\n[3/5] Generating visualizations...")
    plot_results(high_act, low_act, top_neurons, all_results)
    
    # Summary
    print("\n[4/5] Computing summary statistics...")
    print_summary(high_act, low_act, top_neurons, all_results)
    
    # Issues
    print("\n[5/5] Analyzing technical issues...")
    analyze_technical_issues(high_act, low_act, top_neurons)
    
    print("\n✓ Experiment complete!")
    print("\nNext steps:")
    print("  1. Try different layers (0-11)")
    print("  2. Increase sample size")
    print("  3. Implement ablation studies")
    print("  4. Test generalization with new prompts")

if __name__ == "__main__":
    main()
