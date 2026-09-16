"""
Generate publication-quality vector figures for paper appendices.
1. Cross-Attention Entropy and Delimiter Sink Reduction across languages.
2. Performance Scaling dynamics across sentence length buckets.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Set matplotlib parameters for publication quality
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['mathtext.fontset'] = 'cm'

out_dir = os.path.join(os.path.dirname(__file__), '..', 'docs', 'paper', 'latex', 'figures')
os.makedirs(out_dir, exist_ok=True)

def plot_entropy_sink():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 3.8), dpi=300)
    
    # Subplot 1: Attention Entropy & Top-1 Concentration (Rhade)
    metrics = [r'Entropy $\mathcal{H}(\alpha)$ $\downarrow$', r'Top-1 Mass (%) $\uparrow$']
    vanilla_rhade = [0.4835, 41.28]
    tssa_rhade = [0.2383, 90.39]
    
    x = np.arange(len(metrics))
    width = 0.32
    
    rects1 = ax1.bar(x - width/2, vanilla_rhade, width, label='Vanilla BARTpho', color='#94A3B8', edgecolor='#475569', lw=1.2)
    rects2 = ax1.bar(x + width/2, tssa_rhade, width, label='UniTSSA (Ours)', color='#2563EB', edgecolor='#1E40AF', lw=1.2)
    
    ax1.set_title('(a) Attention Concentration (Rhade)', fontsize=11, fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=10)
    ax1.legend(frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1', fontsize=9)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.set_ylim(0, 115)
    
    # Annotations on bars
    ax1.text(x[0] - width/2, vanilla_rhade[0] + 3, '0.48', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#475569')
    ax1.text(x[0] + width/2, tssa_rhade[0] + 3, '0.24\n(-50.7%)', ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#1E40AF')
    ax1.text(x[1] - width/2, vanilla_rhade[1] + 3, '41.3%', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#475569')
    ax1.text(x[1] + width/2, tssa_rhade[1] + 3, '90.4%\n(+2.2x)', ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#1E40AF')

    # Subplot 2: Delimiter Attention Sink Mass onto <s>
    langs = ['Tay', 'Bahnaric']
    vanilla_sink = [93.11, 89.93]
    tssa_sink = [40.17, 48.79]
    
    x2 = np.arange(len(langs))
    rects3 = ax2.bar(x2 - width/2, vanilla_sink, width, label='Vanilla (Severe Sink)', color='#EF4444', edgecolor='#B91C1C', lw=1.2)
    rects4 = ax2.bar(x2 + width/2, tssa_sink, width, label='UniTSSA (Anchored)', color='#10B981', edgecolor='#047857', lw=1.2)
    
    ax2.set_title(r'(b) Delimiter Sink Mass ($<s>$) $\downarrow$', fontsize=11, fontweight='bold', pad=10)
    ax2.set_xticks(x2)
    ax2.set_xticklabels([f'{l} $\\rightarrow$ vi' for l in langs], fontsize=10)
    ax2.legend(frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1', fontsize=9, loc='upper center')
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Probability Mass on Delimiter (%)', fontsize=9.5)
    ax2.set_ylim(0, 125)
    
    ax2.text(x2[0] - width/2, vanilla_sink[0] + 3, '93.1%', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#991B1B')
    ax2.text(x2[0] + width/2, tssa_sink[0] + 3, '40.2%\n(-52.9%)', ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#065F46')
    ax2.text(x2[1] - width/2, vanilla_sink[1] + 3, '89.9%', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#991B1B')
    ax2.text(x2[1] + width/2, tssa_sink[1] + 3, '48.8%\n(-41.1%)', ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#065F46')

    plt.tight_layout()
    pdf_path = os.path.join(out_dir, 'appendix_entropy_sink.pdf')
    png_path = os.path.join(out_dir, 'appendix_entropy_sink.png')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {pdf_path}")

def plot_length_scaling():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 3.8), dpi=300)
    
    buckets = ['Short\n(≤ 12 words)', 'Medium\n(13–25 words)', 'Long\n(> 25 words)']
    
    vanilla_bleu = [54.81, 22.60, 15.14]
    tssa_bleu = [55.17, 23.09, 16.09]
    delta_bleu = [0.36, 0.49, 0.95]
    
    vanilla_comet = [-0.3539, -0.2578, -0.5332]
    tssa_comet = [-0.3299, -0.2084, -0.4515]
    delta_comet = [0.0240, 0.0494, 0.0817]
    
    x = np.arange(len(buckets))
    
    # Subplot 1: BLEU Absolute & Delta
    ax1.plot(x, vanilla_bleu, 'o--', color='#64748B', lw=1.8, ms=7, label='Vanilla BARTpho')
    ax1.plot(x, tssa_bleu, 's-', color='#2563EB', lw=2.2, ms=8, label='UniTSSA (Ours)')
    ax1.set_title('(a) BLEU Score Across Length Buckets', fontsize=11, fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(buckets, fontsize=9.5)
    ax1.set_ylabel('SacreBLEU Score', fontsize=9.5)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1', fontsize=9)
    ax1.set_ylim(10, 62)
    
    # Add gain callouts
    for i in range(len(x)):
        ax1.annotate(f'+{delta_bleu[i]:.2f}',
                     xy=(x[i], tssa_bleu[i]),
                     xytext=(0, 10), textcoords='offset points',
                     ha='center', fontsize=9, fontweight='bold',
                     color='#1E40AF',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='#EFF6FF', edgecolor='#2563EB', lw=0.8))
                     
    # Subplot 2: Neural Semantic Fidelity (COMET)
    ax2.plot(x, vanilla_comet, 'o--', color='#64748B', lw=1.8, ms=7, label='Vanilla BARTpho')
    ax2.plot(x, tssa_comet, '^-', color='#7C3AED', lw=2.2, ms=8, label='UniTSSA (Ours)')
    ax2.set_title('(b) COMET Semantic Fidelity Across Length', fontsize=11, fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(buckets, fontsize=9.5)
    ax2.set_ylabel('COMET Score', fontsize=9.5)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1', fontsize=9)
    ax2.set_ylim(-0.62, -0.15)
    
    for i in range(len(x)):
        ax2.annotate(f'+{delta_comet[i]:.4f}',
                     xy=(x[i], tssa_comet[i]),
                     xytext=(0, 10), textcoords='offset points',
                     ha='center', fontsize=8.5, fontweight='bold',
                     color='#6D28D9',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='#FAF5FF', edgecolor='#7C3AED', lw=0.8))

    plt.tight_layout()
    pdf_path = os.path.join(out_dir, 'appendix_length_scaling.pdf')
    png_path = os.path.join(out_dir, 'appendix_length_scaling.png')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {pdf_path}")

if __name__ == '__main__':
    plot_entropy_sink()
    plot_length_scaling()
