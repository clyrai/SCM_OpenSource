import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(figsize=(12, 5.5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 5.5)
ax.axis('off')

def draw_state(ax, x, y, w, h, text, color, fontsize=11):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.25",
                          facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.92)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color='white', wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, label='', fontsize=9, color='#2c3e50', curved=False, rad=0):
    if curved:
        style = f"arc3,rad={rad}"
        arrowprops = dict(arrowstyle='->', color=color, lw=1.8, connectionstyle=style)
    else:
        arrowprops = dict(arrowstyle='->', color=color, lw=1.8)
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=arrowprops)
    if label:
        if curved:
            mx, my = (x1+x2)/2, (y1+y2)/2
            if rad > 0:
                my += 0.35
            else:
                my -= 0.35
            ax.text(mx, my, label, ha='center', va='center', fontsize=fontsize, 
                    style='italic', color='#34495e', fontweight='medium',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85))
        else:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx, my + 0.18, label, ha='center', va='bottom', fontsize=fontsize,
                    style='italic', color='#34495e', fontweight='medium',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85))

# Main horizontal flow: WAKE -> NREM -> REM -> WAKE
# Position states in a clear row
draw_state(ax, 0.5, 2.8, 2.2, 1.0, 'WAKE\n(process input)', '#3498db')
draw_state(ax, 3.5, 2.8, 2.2, 1.0, 'NREM\n(consolidate)', '#9b59b6')
draw_state(ax, 6.5, 2.8, 2.2, 1.0, 'REM\n(dream)', '#e67e22')
draw_state(ax, 9.5, 2.8, 2.2, 1.0, 'WAKE\n(resume)', '#3498db')

# Top arrows (main flow)
draw_arrow(ax, 2.7, 3.3, 3.5, 3.3, 'entropy > 0.9\nconflict > 0.3\nt > 1 hr')
draw_arrow(ax, 5.7, 3.3, 6.5, 3.3, 'replay\ndownscale')
draw_arrow(ax, 8.7, 3.3, 9.5, 3.3, 'dream\nintegrate')

# Bottom row: Forgetting module
draw_state(ax, 4.5, 0.9, 3.0, 0.9, 'FORGET\n(value-based pruning)', '#e74c3c')

# Arrows from NREM and REM to FORGET
draw_arrow(ax, 4.6, 2.8, 5.2, 1.8, '', curved=True, rad=-0.25)
draw_arrow(ax, 7.4, 2.8, 6.8, 1.8, '', curved=True, rad=0.25)

# Label for forgetting triggers
ax.text(4.3, 2.15, 'value < threshold', ha='center', va='center', fontsize=9,
        style='italic', color='#e74c3c', fontweight='bold')
ax.text(7.7, 2.15, 'value < threshold', ha='center', va='center', fontsize=9,
        style='italic', color='#e74c3c', fontweight='bold')

# Return arrow from FORGET back to main flow (to WAKE resume)
draw_arrow(ax, 7.5, 1.35, 10.6, 2.8, 'prune &\nresume', curved=True, rad=0.3)

# Self-loop on WAKE (processing continues)
ax.annotate('', xy=(1.6, 2.8), xytext=(1.6, 3.8),
            arrowprops=dict(arrowstyle='->', color='#3498db', lw=1.5, connectionstyle='arc3,rad=0.4'))
ax.text(0.6, 3.8, 'process\ninput', ha='center', va='center', fontsize=9,
        style='italic', color='#3498db')

# Title
ax.text(6, 5.1, 'SCM Sleep Cycle State Machine', ha='center', va='center', fontsize=15, fontweight='bold')
ax.text(6, 4.7, 'WAKE  →  NREM  →  REM  →  WAKE  with forgetting at each transition',
        ha='center', va='center', fontsize=11, style='italic', color='#7f8c8d')

# Legend
legend_elements = [
    mpatches.Patch(facecolor='#3498db', label='Wake phase (input processing)'),
    mpatches.Patch(facecolor='#9b59b6', label='NREM (consolidation)'),
    mpatches.Patch(facecolor='#e67e22', label='REM (dreaming)'),
    mpatches.Patch(facecolor='#e74c3c', label='Forgetting (pruning)'),
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=9, framealpha=0.9, ncol=2)

plt.tight_layout()
plt.savefig('/Users/saish/Downloads/SleepAI/research/latex/figures/sleep_state_machine.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('/Users/saish/Downloads/SleepAI/research/latex/figures/sleep_state_machine.png', format='png', bbox_inches='tight', dpi=300)
print("State machine figure regenerated (cleaner version).")
