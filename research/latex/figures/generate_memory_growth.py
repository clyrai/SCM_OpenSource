import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(9, 4.5))

# Simulate 20 sleep cycles with 5 new concepts per cycle
np.random.seed(42)
cycles = np.arange(0, 21)

# Without forgetting: linear growth
without_forgetting = cycles * 5 + 10

# With forgetting: growth with pruning
with_forgetting = np.zeros_like(cycles, dtype=float)
with_forgetting[0] = 10
for i in range(1, len(cycles)):
    new_concepts = 5
    # Prune low-importance concepts based on current size
    prune_rate = min(0.4, 0.1 + 0.02 * i)
    pruned = int(with_forgetting[i-1] * prune_rate)
    with_forgetting[i] = with_forgetting[i-1] + new_concepts - pruned

ax.plot(cycles, without_forgetting, 'o-', color='#e74c3c', linewidth=2, markersize=6, label='Without ForgettingModule')
ax.plot(cycles, with_forgetting, 's-', color='#2ecc71', linewidth=2, markersize=6, label='With ForgettingModule')
ax.fill_between(cycles, with_forgetting, without_forgetting, alpha=0.15, color='#2ecc71')

ax.set_xlabel('Sleep Cycle', fontsize=12)
ax.set_ylabel('LTM Concept Count', fontsize=12)
ax.set_title('Memory Growth With and Without Intentional Forgetting', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='upper left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(0, 20)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/saish/Downloads/SleepAI/research/latex/figures/memory_growth.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('/Users/saish/Downloads/SleepAI/research/latex/figures/memory_growth.png', format='png', bbox_inches='tight', dpi=300)
print("Memory growth figure generated.")
