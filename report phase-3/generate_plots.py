import re
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

os.makedirs('report phase-3/plots', exist_ok=True)

with open('language_H/configs/sweep_results/res.txt') as f:
    text = f.read()

# Pattern for runs
pattern = r'\[(\d+)/(\d+)\]\s+([^\n]+).*?step\s+200/1000\s+val_loss=([\d\.]+).*?step\s+400/1000\s+val_loss=([\d\.]+).*?step\s+600/1000\s+val_loss=([\d\.]+).*?step\s+800/1000\s+val_loss=([\d\.]+).*?step\s+1000/1000\s+val_loss=([\d\.]+).*?✓\s+val_loss=([\d\.]+)\s+PPL=([\d\.]+)\s+BPB=([\d\.]+)\s+params=([\d,]+)'
matches = re.findall(pattern, text, re.DOTALL)

runs = []
for m in matches:
    name = m[2].strip()
    parts = name.split('_')
    family = parts[1]
    act = 'SwiGLU' if 'swiglu' in parts else 'GELU'
    ctx = int(re.search(r'ctx(\d+)', name).group(1))
    bs = int(re.search(r'bs(\d+)', name).group(1))
    lr = float(re.search(r'lr([\de\-\.]+)', name).group(1))
    steps = [200, 400, 600, 800, 1000]
    history = [float(m[3]), float(m[4]), float(m[5]), float(m[6]), float(m[7])]
    val_loss = float(m[8])
    ppl = float(m[9])
    bpb = float(m[10])
    params = int(m[11].replace(',', ''))
    
    runs.append({
        'run': int(m[0]),
        'name': name,
        'family': family,
        'act': act,
        'ctx': ctx,
        'bs': bs,
        'lr': lr,
        'history': history,
        'val_loss': val_loss,
        'ppl': ppl,
        'bpb': bpb,
        'params': params
    })

print(f'Parsed {len(runs)} runs successfully.')

# Styling
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14
})

# -------------------------------------------------------------
# PLOT 1: Architecture Family Comparison (Deep vs Base vs Wide & GELU vs SwiGLU)
# -------------------------------------------------------------
families = ['deep', 'base', 'wide']
family_labels = ['Deep (12L / 384d)', 'Base (6L / 512d)', 'Wide (4L / 640d)']
acts = ['GELU', 'SwiGLU']

gelu_means = [np.mean([r['val_loss'] for r in runs if r['family'] == f and r['act'] == 'GELU']) for f in families]
swiglu_means = [np.mean([r['val_loss'] for r in runs if r['family'] == f and r['act'] == 'SwiGLU']) for f in families]

gelu_bests = [min([r['val_loss'] for r in runs if r['family'] == f and r['act'] == 'GELU']) for f in families]
swiglu_bests = [min([r['val_loss'] for r in runs if r['family'] == f and r['act'] == 'SwiGLU']) for f in families]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
x = np.arange(len(families))
width = 0.35

# Mean Val Loss
rects1 = ax1.bar(x - width/2, gelu_means, width, label='GELU', color='#4A90E2', alpha=0.9, edgecolor='black', linewidth=0.8)
rects2 = ax1.bar(x + width/2, swiglu_means, width, label='SwiGLU', color='#50E3C2', alpha=0.9, edgecolor='black', linewidth=0.8)
ax1.set_ylabel('Mean Validation Loss (nats)')
ax1.set_title('Average Validation Loss across Grid')
ax1.set_xticks(x)
ax1.set_xticklabels(family_labels)
ax1.legend()
ax1.grid(axis='y', linestyle='--', alpha=0.5)
ax1.set_ylim(6.0, 7.2)

for rect in rects1:
    h = rect.get_height()
    ax1.annotate(f'{h:.2f}', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
for rect in rects2:
    h = rect.get_height()
    ax1.annotate(f'{h:.2f}', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

# Best Val Loss
rects3 = ax2.bar(x - width/2, gelu_bests, width, label='GELU', color='#4A90E2', alpha=0.9, edgecolor='black', linewidth=0.8)
rects4 = ax2.bar(x + width/2, swiglu_bests, width, label='SwiGLU', color='#50E3C2', alpha=0.9, edgecolor='black', linewidth=0.8)
ax2.set_ylabel('Best Validation Loss (nats)')
ax2.set_title('Best Validation Loss (at ctx512, bs16, lr1e-3)')
ax2.set_xticks(x)
ax2.set_xticklabels(family_labels)
ax2.legend()
ax2.grid(axis='y', linestyle='--', alpha=0.5)
ax2.set_ylim(5.5, 6.8)

for rect in rects3:
    h = rect.get_height()
    ax2.annotate(f'{h:.2f}', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
for rect in rects4:
    h = rect.get_height()
    ax2.annotate(f'{h:.2f}', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

plt.suptitle('Architecture Family & Activation Function Comparison (~25M Parameter Budget)', y=1.02)
plt.tight_layout()
plt.savefig('report phase-3/plots/architecture_family_comparison.png', dpi=200, bbox_inches='tight')
plt.close()
print('Generated architecture_family_comparison.png')

# -------------------------------------------------------------
# PLOT 2: Hyperparameter Trends (LR, Batch Size, Context Length)
# -------------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))

# LR Impact
lr_3e4 = [r['val_loss'] for r in runs if r['lr'] == 3e-4]
lr_1e3 = [r['val_loss'] for r in runs if r['lr'] == 1e-3]
bplot1 = ax1.boxplot([lr_3e4, lr_1e3], tick_labels=['3e-4', '1e-3'], patch_artist=True, medianprops=dict(color='black', linewidth=1.5))
for patch, c in zip(bplot1['boxes'], ['#F5A623', '#7ED321']):
    patch.set_facecolor(c)
ax1.set_ylabel('Validation Loss (nats)')
ax1.set_xlabel('Learning Rate')
ax1.set_title('(a) Learning Rate Impact')
ax1.grid(axis='y', linestyle='--', alpha=0.5)

# Batch Size Impact
bs_8 = [r['val_loss'] for r in runs if r['bs'] == 8]
bs_16 = [r['val_loss'] for r in runs if r['bs'] == 16]
bplot2 = ax2.boxplot([bs_8, bs_16], tick_labels=['8', '16'], patch_artist=True, medianprops=dict(color='black', linewidth=1.5))
for patch, c in zip(bplot2['boxes'], ['#BD10E0', '#9013FE']):
    patch.set_facecolor(c)
ax2.set_ylabel('Validation Loss (nats)')
ax2.set_xlabel('Batch Size')
ax2.set_title('(b) Batch Size Impact')
ax2.grid(axis='y', linestyle='--', alpha=0.5)

# Context Length Impact
ctx_256 = [r['val_loss'] for r in runs if r['ctx'] == 256]
ctx_512 = [r['val_loss'] for r in runs if r['ctx'] == 512]
bplot3 = ax3.boxplot([ctx_256, ctx_512], tick_labels=['256', '512'], patch_artist=True, medianprops=dict(color='black', linewidth=1.5))
for patch, c in zip(bplot3['boxes'], ['#4A90E2', '#50E3C2']):
    patch.set_facecolor(c)
ax3.set_ylabel('Validation Loss (nats)')
ax3.set_xlabel('Context Length (tokens)')
ax3.set_title('(c) Context Length Impact')
ax3.grid(axis='y', linestyle='--', alpha=0.5)

plt.suptitle('Hyperparameter Sweep Marginal Distributions on Validation Loss', y=1.03)
plt.tight_layout()
plt.savefig('report phase-3/plots/hparam_marginal_distributions.png', dpi=200, bbox_inches='tight')
plt.close()
print('Generated hparam_marginal_distributions.png')

# -------------------------------------------------------------
# PLOT 3: Convergence Curves for Key Configurations
# -------------------------------------------------------------
plt.figure(figsize=(10, 5.5))
key_runs = [
    ('run48', 'Wide SwiGLU (4L / 640d) [ctx512, bs16, lr1e-3]', '#E74C3C', '-'),
    ('run32', 'Base SwiGLU (6L / 512d) [ctx512, bs16, lr1e-3]', '#2ECC71', '-'),
    ('run40', 'Deep SwiGLU (12L / 384d) [ctx512, bs16, lr1e-3]', '#3498DB', '-'),
    ('run24', 'Wide GELU (4L / 640d) [ctx512, bs16, lr1e-3]', '#E67E22', '--'),
    ('run08', 'Base GELU (6L / 512d) [ctx512, bs16, lr1e-3]', '#1ABC9C', '--'),
    ('run16', 'Deep GELU (12L / 384d) [ctx512, bs16, lr1e-3]', '#9B59B6', '--'),
]

steps = [200, 400, 600, 800, 1000]
for prefix, label, color, style in key_runs:
    matched = [x for x in runs if prefix in x['name']]
    if matched:
        r = matched[0]
        final_l = r['val_loss']
        plt.plot(steps, r['history'], marker='o', color=color, linestyle=style, linewidth=2.0, markersize=5, label=f"{label} -> {final_l:.2f}")

plt.xlabel('Probe Training Steps')
plt.ylabel('Validation Loss (nats)')
plt.title('Validation Loss Trajectory over 1,000 Probe Steps')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
plt.tight_layout()
plt.savefig('report phase-3/plots/sweep_convergence_curves.png', dpi=200, bbox_inches='tight')
plt.close()
print('Generated sweep_convergence_curves.png')

# -------------------------------------------------------------
# PLOT 4: Depth vs Width Tradeoff across Constant Parameter Budget
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 5))
depths = [4, 6, 12]
swiglu_best = [5.7857, 5.9057, 6.1337]
gelu_best = [5.9257, 6.1357, 6.4423]

ax.plot(depths, swiglu_best, marker='s', markersize=8, linewidth=2.5, color='#2ECC71', label='SwiGLU Activation (Best probe loss)')
ax.plot(depths, gelu_best, marker='^', markersize=8, linewidth=2.5, color='#3498DB', label='GELU Activation (Best probe loss)')

ax.annotate('Wide (4L, 640d)\nHigh capacity/step\nShallow composition', xy=(4, 5.7857), xytext=(4.15, 5.86),
            arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6), fontsize=9)

ax.annotate('Base (6L, 512d) [Selected Architecture]\nBalanced depth & width\nOptimal multi-hop reasoning\nStandard d_head=64 tensor layout',
            xy=(6, 5.9057), xytext=(6.2, 5.95),
            arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6), fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc="#FFF9C4", ec="#FBC02D", lw=1.5))

ax.annotate('Deep (12L, 384d)\nSlow probe convergence\nNarrow representation bottleneck', xy=(12, 6.1337), xytext=(9.2, 6.27),
            arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6), fontsize=9)

ax.set_xlabel('Network Depth (Number of Transformer Layers N)')
ax.set_ylabel('Validation Loss at 1k steps (nats)')
ax.set_title('Depth vs. Width Trade-off under ~25M Parameter Budget')
ax.set_xticks([4, 6, 8, 10, 12])
ax.set_ylim(5.6, 6.6)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig('report phase-3/plots/depth_vs_width_tradeoff.png', dpi=200, bbox_inches='tight')
plt.close()
print('Generated depth_vs_width_tradeoff.png')
