import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis('off')

# Color scheme
c_wake = '#3498db'
c_sleep = '#9b59b6'
c_memory = '#2ecc71'
c_flow = '#34495e'

def draw_box(ax, x, y, w, h, text, color, fontsize=9):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.15",
                          facecolor=color, edgecolor='black', linewidth=1.2, alpha=0.9)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color='white', wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, color=c_flow):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

# Title
ax.text(6, 7.6, 'SleepAI Architecture', ha='center', va='center', fontsize=16, fontweight='bold')

# Wake Phase label
ax.text(1.5, 7.1, 'WAKE PHASE', ha='center', va='center', fontsize=11, fontweight='bold', color=c_wake)

# Top row: processing pipeline
draw_box(ax, 0.5, 6.0, 2.0, 0.8, 'User Input', '#7f8c8d')
draw_box(ax, 3.0, 6.0, 2.2, 0.8, 'MeaningEncoder\n(LLM extraction)', c_wake)
draw_box(ax, 5.7, 6.0, 2.2, 0.8, 'ValueTagger\n(4D importance)', c_wake)
draw_box(ax, 8.4, 6.0, 2.2, 0.8, 'WorkingMemory\n(7 items, FIFO)', '#e74c3c')

draw_arrow(ax, 2.5, 6.4, 3.0, 6.4)
draw_arrow(ax, 5.2, 6.4, 5.7, 6.4)
draw_arrow(ax, 7.9, 6.4, 8.4, 6.4)

# Sleep Phase label
ax.text(1.5, 5.1, 'SLEEP PHASE', ha='center', va='center', fontsize=11, fontweight='bold', color=c_sleep)

# Sleep trigger
draw_box(ax, 0.5, 4.2, 2.5, 0.7, 'Sleep Trigger\n(entropy / conflict / time)', '#f39c12')

# NREM
draw_box(ax, 3.5, 4.2, 2.5, 0.7, 'NREM Consolidation\n(replay + Hebbian + downscale)', c_sleep)

# REM
draw_box(ax, 6.5, 4.2, 2.5, 0.7, 'REM Dreaming\n(novel combinations)', c_sleep)

# Forgetting
draw_box(ax, 9.5, 4.2, 2.0, 0.7, 'Forgetting\n(value-based prune)', '#e74c3c')

draw_arrow(ax, 3.0, 4.55, 3.5, 4.55)
draw_arrow(ax, 6.0, 4.55, 6.5, 4.55)
draw_arrow(ax, 9.0, 4.55, 9.5, 4.55)

# Long-Term Memory (big box at bottom)
ltm_box = FancyBboxPatch((1.5, 1.0), 9.0, 2.5, boxstyle="round,pad=0.03,rounding_size=0.2",
                          facecolor=c_memory, edgecolor='black', linewidth=2, alpha=0.3)
ax.add_patch(ltm_box)
ax.text(6, 3.1, 'Long-Term Memory (NetworkX Graph + SQLite)', ha='center', va='center',
        fontsize=12, fontweight='bold', color='#27ae60')

# Internal LTM components
draw_box(ax, 2.0, 1.5, 2.5, 1.0, 'Semantic Search\n(cosine similarity)', c_memory)
draw_box(ax, 4.8, 1.5, 2.5, 1.0, 'Graph Traversal\n(relation edges)', c_memory)
draw_box(ax, 7.6, 1.5, 2.5, 1.0, 'Importance Ranking\n(value score)', c_memory)

# Arrows from WM to LTM and Sleep
ax.annotate('', xy=(6, 3.5), xytext=(6, 4.2),
            arrowprops=dict(arrowstyle='->', color=c_flow, lw=2))
ax.annotate('', xy=(6, 3.5), xytext=(9.5, 4.2),
            arrowprops=dict(arrowstyle='->', color=c_flow, lw=1.5, connectionstyle='arc3,rad=-0.2'))

# Self-Model (side box)
draw_box(ax, 0.2, 1.5, 1.5, 1.8, 'Self-Model\n(identity +\nintrospection +\nsleep memories)', '#8e44ad')
ax.annotate('', xy=(1.5, 2.4), xytext=(2.0, 2.4),
            arrowprops=dict(arrowstyle='<->', color='#8e44ad', lw=1.5))

# Legend
legend_elements = [
    mpatches.Patch(facecolor=c_wake, label='Wake processing'),
    mpatches.Patch(facecolor=c_sleep, label='Sleep processing'),
    mpatches.Patch(facecolor=c_memory, label='Memory storage'),
    mpatches.Patch(facecolor='#e74c3c', label='Capacity limit / prune'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.9)

plt.tight_layout()
plt.savefig('/Users/saish/Downloads/SleepAI/research/latex/figures/architecture.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('/Users/saish/Downloads/SleepAI/research/latex/figures/architecture.png', format='png', bbox_inches='tight', dpi=300)
print("Architecture figure generated.")
