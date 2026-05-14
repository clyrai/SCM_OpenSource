import matplotlib.pyplot as plt
import numpy as np

# Data
tests = [
    "Working Memory\nCapacity",
    "Memory Retention\n(5 turns)",
    "Memory Retention\n(10 turns)",
    "Sleep Consolidation\nBenefit",
    "Forgetting\nEffectiveness",
    "Graph\nTraversal",
    "Latency\nScaling",
    "Multi-Session\nPersistence"
]
scores = [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]
colors = ['#2ecc71'] * 8  # All green for perfect scores

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(tests, scores, color=colors, edgecolor='black', linewidth=0.5)

# Add score labels
for bar, score in zip(bars, scores):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            f'{score:.2f}', va='center', fontsize=10, fontweight='bold')

ax.set_xlim(0, 1.15)
ax.set_xlabel('Score', fontsize=12)
ax.set_title('SleepAI Benchmark Results', fontsize=14, fontweight='bold', pad=15)
ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('/Users/saish/Downloads/SleepAI/research/latex/figures/benchmark_results.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('/Users/saish/Downloads/SleepAI/research/latex/figures/benchmark_results.png', format='png', bbox_inches='tight', dpi=300)
print("Benchmark figure generated.")
