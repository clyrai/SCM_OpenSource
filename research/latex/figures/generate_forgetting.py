import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Left: buggy formula (beta2=0.4)
ax = axes[0]
categories = ['Important\nFacts', 'Noise\nConcepts']
before = [5, 50]
after_buggy = [5, 50]  # nothing forgotten
x = np.arange(len(categories))
width = 0.3
bars1 = ax.bar(x - width/2, before, width, label='Before Sleep', color='#3498db', edgecolor='black')
bars2 = ax.bar(x + width/2, after_buggy, width, label='After Sleep (Buggy)', color='#e74c3c', edgecolor='black')
ax.set_ylabel('Concept Count', fontsize=11)
ax.set_title(r'Buggy: $\beta_2 = 0.4$' + '\n0% noise reduction', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylim(0, 60)
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(int(bar.get_height())),
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# Right: corrected formula (beta2=0.2)
ax = axes[1]
after_fixed = [5, 5]  # 45 of 50 noise removed
bars3 = ax.bar(x - width/2, before, width, label='Before Sleep', color='#3498db', edgecolor='black')
bars4 = ax.bar(x + width/2, after_fixed, width, label='After Sleep (Fixed)', color='#2ecc71', edgecolor='black')
ax.set_ylabel('Concept Count', fontsize=11)
ax.set_title(r'Fixed: $\beta_2 = 0.2$' + '\n90.9% noise reduction', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylim(0, 60)
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for bar in bars4:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(int(bar.get_height())),
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.suptitle('Forgetting Formula Correction', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/Users/saish/Downloads/SleepAI/research/latex/figures/forgetting_curve.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('/Users/saish/Downloads/SleepAI/research/latex/figures/forgetting_curve.png', format='png', bbox_inches='tight', dpi=300)
print("Forgetting curve figure generated.")
