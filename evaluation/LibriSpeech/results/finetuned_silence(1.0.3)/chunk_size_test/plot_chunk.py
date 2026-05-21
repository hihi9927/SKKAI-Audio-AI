import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mp

data = {
    'chunk0.25':   {'wer': 0.14928, 'ftl': 3.714, 'runtime': 1.086, 'fsl': 1.613, 'seg_ratio': 0.871},
    'chunk0.25_2': {'wer': 0.10925, 'ftl': 3.401, 'runtime': 0.376, 'fsl': 0.866, 'seg_ratio': 0.991},
    'chunk0.5':    {'wer': 0.09824, 'ftl': 3.743, 'runtime': 0.849, 'fsl': 1.378, 'seg_ratio': 0.815},
    'chunk1.0':    {'wer': 0.08713, 'ftl': 4.164, 'runtime': 1.007, 'fsl': 1.009, 'seg_ratio': 0.660},
    'chunk1.5':    {'wer': 0.09613, 'ftl': 4.526, 'runtime': 1.207, 'fsl': 1.163, 'seg_ratio': 0.599},
    'chunk2.0':    {'wer': 0.10602, 'ftl': 4.728, 'runtime': 1.135, 'fsl': 1.010, 'seg_ratio': 0.496},
}

labels  = list(data.keys())
x_pos   = [0, 0.6, 1.4, 2.2, 3.0, 3.8]
x_lbls  = ['0.25\n(run1)', '0.25\n(run2)', '0.5', '1.0', '1.5', '2.0']
hatches = ['///', '', '', '', '', '']

wer   = [data[k]['wer']*100  for k in labels]
ftl   = [data[k]['ftl']      for k in labels]
rt    = [data[k]['runtime']  for k in labels]
fsl   = [data[k]['fsl']      for k in labels]
seg_r = [data[k]['seg_ratio']*100 for k in labels]

def bcolors(values, good='low'):
    best = min(values) if good == 'low' else max(values)
    return ['#2ecc71' if v == best else ('#f39c12' if '0.25' in labels[i] else '#95a5a6')
            for i, v in enumerate(values)]

def draw_bar(ax, values, title, ylabel, colors):
    bars = ax.bar(x_pos, values, color=colors, edgecolor='black', lw=0.8, width=0.5)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.012,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_lbls)
    ax.set_xlabel('Chunk Size (s)')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(values) * 1.2)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle('Chunk Size Sweep — finetuned_silence 1.0.3', fontsize=14, fontweight='bold')

draw_bar(axes[0][0], wer, 'WER (%)',                'WER (%)', bcolors(wer))
draw_bar(axes[0][1], ftl, 'First Token Latency (s)', 'sec',    bcolors(ftl))
draw_bar(axes[1][0], rt,  'Model Runtime (s)',       'sec',    bcolors(rt))

ax = axes[1][1]
bc = ['#f39c12' if '0.25' in labels[i] else '#3498db' for i in range(len(labels))]
bars = ax.bar(x_pos, seg_r, color=bc, alpha=0.75, edgecolor='black', lw=0.8, width=0.5)
for bar, h in zip(bars, hatches):
    bar.set_hatch(h)
for bar, val in zip(bars, seg_r):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax2 = ax.twinx()
ax2.plot(x_pos, fsl, 'ro-', lw=2, ms=7)
for xp, val in zip(x_pos, fsl):
    ax2.text(xp, val + 0.04, f'{val:.2f}', ha='center', va='bottom', fontsize=8, color='red')
ax.set_title('SEG Ratio (%) & avg FSL (s)', fontsize=11, fontweight='bold')
ax.set_ylabel('SEG ratio (%)')
ax2.set_ylabel('avg FSL (s)', color='red')
ax.set_xticks(x_pos)
ax.set_xticklabels(x_lbls)
ax.set_xlabel('Chunk Size (s)')
ax.set_ylim(0, 120)
ax2.set_ylim(0, 2.5)
ax.grid(axis='y', alpha=0.3)
h1 = mp.Patch(fc='#f39c12', ec='black', label='0.25 runs (///=run1)')
h2 = mp.Patch(fc='#3498db', ec='black', label='other sizes')
h3 = plt.Line2D([0], [0], color='red', marker='o', lw=2, label='avg FSL (s)')
ax.legend(handles=[h1, h2, h3], loc='upper right', fontsize=8)

plt.tight_layout()
import os
out = os.path.join(os.path.dirname(__file__), 'chunk_sweep_result.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print('saved:', out)
