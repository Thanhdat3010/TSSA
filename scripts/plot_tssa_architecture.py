"""
Plot Target-Side Semantic Anchoring (UniTSSA) Architecture Diagram.
Publication-grade vector graphics generation for NAACL/ACL submission.
Version 3: Pristine orthogonal bus routing, flawless arrows, zero overlap.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, PathPatch
from matplotlib.path import Path
import os

# Matplotlib settings for publication quality
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['mathtext.fontset'] = 'cm'

def create_tssa_architecture_diagram():
    # Canvas dimension: 17.5 x 9.6 inches
    fig_w, fig_h = 17.5, 9.6
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 17.5)
    ax.set_ylim(0, 9.6)
    ax.axis('off')

    # Palette
    c_teacher_bg = '#FFFBEB'
    c_teacher_border = '#D97706'
    
    c_student_bg = '#EFF6FF'
    c_student_border = '#2563EB'
    
    c_align_bg = '#ECFDF5'
    c_align_border = '#059669'
    
    c_prime_bg = '#F0FDFA'
    c_prime_border = '#0D9488'
    
    c_router_bg = '#FAF5FF'
    c_router_border = '#7C3AED'
    
    c_loss_bg = '#FEF2F2'
    c_loss_border = '#DC2626'
    c_loss_text = '#991B1B'
    
    c_neutral_bg = '#F8FAFC'
    c_neutral_border = '#64748B'

    def draw_box(x, y, w, h, bg, border, lw=1.5, ls='-', corner=0.18):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.0,rounding_size={corner}",
            facecolor=bg, edgecolor=border,
            linewidth=lw, linestyle=ls, zorder=2
        )
        ax.add_patch(box)
        return box

    def draw_path(pts, color='#64748B', lw=1.5, ls='-', arrow=True):
        if arrow:
            # Draw line up to the second-to-last point to avoid double-stroking under arrow head
            if len(pts) > 2:
                path_data = [(Path.MOVETO, pts[0])]
                for p in pts[1:-1]:
                    path_data.append((Path.LINETO, p))
                codes, verts = zip(*path_data)
                path = Path(verts, codes)
                patch = PathPatch(path, facecolor='none', edgecolor=color, lw=lw, ls=ls, zorder=3)
                ax.add_patch(patch)
            # Arrow takes care of the final segment cleanly
            arr = FancyArrowPatch(
                pts[-2], pts[-1],
                arrowstyle='-|>,head_length=5.5,head_width=3.8',
                color=color, lw=lw, linestyle=ls, zorder=4
            )
            ax.add_patch(arr)
        else:
            path_data = [(Path.MOVETO, pts[0])]
            for p in pts[1:]:
                path_data.append((Path.LINETO, p))
            codes, verts = zip(*path_data)
            path = Path(verts, codes)
            patch = PathPatch(path, facecolor='none', edgecolor=color, lw=lw, ls=ls, zorder=3)
            ax.add_patch(patch)

    def draw_straight_arrow(p1, p2, color='#64748B', lw=1.5, ls='-'):
        arr = FancyArrowPatch(
            p1, p2,
            arrowstyle='-|>,head_length=5.5,head_width=3.8',
            color=color, lw=lw, linestyle=ls, zorder=4
        )
        ax.add_patch(arr)

    # -------------------------------------------------------------
    # 1. COLUMN 1: DUAL-STREAM INPUTS & ENCODERS (x in [0.5, 3.7])
    # -------------------------------------------------------------
    c1_x = 0.5
    c1_w = 3.2
    
    # Target Sentence Y
    y_tgt_in = 8.10
    h_tgt_in = 0.90
    draw_box(c1_x, y_tgt_in, c1_w, h_tgt_in, c_neutral_bg, c_neutral_border, lw=1.2)
    ax.text(c1_x + c1_w/2, y_tgt_in + 0.60, r"$\mathbf{Target\;Sentence}\;\mathcal{Y}$", 
            ha='center', va='center', fontsize=11, fontweight='bold', color='#1E293B')
    ax.text(c1_x + c1_w/2, y_tgt_in + 0.28, r"$(y_1, y_2, \dots, y_T)\;\;[\mathrm{\mathbf{Vietnamese}}]$", 
            ha='center', va='center', fontsize=9.5, color='#475569')

    # Online Frozen Teacher E_T
    y_teacher = 6.45
    h_teacher = 1.30
    draw_box(c1_x, y_teacher, c1_w, h_teacher, c_teacher_bg, c_teacher_border, lw=2.0, ls='--')
    ax.text(c1_x + c1_w/2, y_teacher + 0.95, r"$\mathbf{Online\;Frozen\;Teacher}\;\mathcal{E}_T$", 
            ha='center', va='center', fontsize=11, fontweight='bold', color='#92400E')
    ax.text(c1_x + c1_w/2, y_teacher + 0.62, r"$\mathcal{E}_T = \mathrm{stop\_gradient}(f_{\mathrm{enc}})$", 
            ha='center', va='center', fontsize=9.5, color='#B45309')
    # Frozen badge
    draw_box(c1_x + 0.35, y_teacher + 0.14, c1_w - 0.7, 0.32, '#FDE68A', '#D97706', lw=1.0)
    ax.text(c1_x + c1_w/2, y_teacher + 0.30, "[FROZEN / NO BACKPROP]", 
            ha='center', va='center', fontsize=8, fontweight='bold', color='#78350F')

    # Target States h^T
    y_tgt_h = 5.20
    h_tgt_h = 0.90
    draw_box(c1_x, y_tgt_h, c1_w, h_tgt_h, c_neutral_bg, c_neutral_border, lw=1.2)
    ax.text(c1_x + c1_w/2, y_tgt_h + 0.58, r"$\mathbf{Target\;States}\;h^T \in \mathbb{R}^{T \times d}$", 
            ha='center', va='center', fontsize=11, fontweight='bold', color='#1E293B')
    ax.text(c1_x + c1_w/2, y_tgt_h + 0.26, "Uncorrupted High-Resource Prior", 
            ha='center', va='center', fontsize=9, fontstyle='italic', color='#475569')

    # Connectors Col 1 (Top)
    draw_straight_arrow((c1_x + c1_w/2, y_tgt_in), (c1_x + c1_w/2, y_teacher + h_teacher), 
                        color=c_teacher_border, lw=1.8, ls='--')
    draw_straight_arrow((c1_x + c1_w/2, y_teacher), (c1_x + c1_w/2, y_tgt_h + h_tgt_h), 
                        color=c_teacher_border, lw=1.8, ls='--')

    # Source States h^S
    y_src_h = 3.90
    h_src_h = 0.90
    draw_box(c1_x, y_src_h, c1_w, h_src_h, c_neutral_bg, c_neutral_border, lw=1.2)
    ax.text(c1_x + c1_w/2, y_src_h + 0.58, r"$\mathbf{Source\;States}\;h^S \in \mathbb{R}^{S \times d}$", 
            ha='center', va='center', fontsize=11, fontweight='bold', color='#1E293B')
    ax.text(c1_x + c1_w/2, y_src_h + 0.26, "Latent Minority Representations", 
            ha='center', va='center', fontsize=9, fontstyle='italic', color='#475569')

    # Student Encoder f_enc
    y_student = 2.25
    h_student = 1.30
    draw_box(c1_x, y_student, c1_w, h_student, c_student_bg, c_student_border, lw=2.0)
    ax.text(c1_x + c1_w/2, y_student + 0.95, r"$\mathbf{Student\;Encoder}\;f_{\mathrm{enc}}$", 
            ha='center', va='center', fontsize=11, fontweight='bold', color='#1E40AF')
    ax.text(c1_x + c1_w/2, y_student + 0.62, "Low-Resource Sequence Encoder", 
            ha='center', va='center', fontsize=9.5, color='#2563EB')
    # Trainable badge
    draw_box(c1_x + 0.35, y_student + 0.14, c1_w - 0.7, 0.32, '#BFDBFE', '#2563EB', lw=1.0)
    ax.text(c1_x + c1_w/2, y_student + 0.30, "[TRAINABLE / BACKPROP]", 
            ha='center', va='center', fontsize=8, fontweight='bold', color='#1E3A8A')

    # Source Sentence X
    y_src_in = 1.00
    h_src_in = 0.90
    draw_box(c1_x, y_src_in, c1_w, h_src_in, c_neutral_bg, c_neutral_border, lw=1.2)
    ax.text(c1_x + c1_w/2, y_src_in + 0.60, r"$\mathbf{Source\;Sentence}\;\mathcal{X}$", 
            ha='center', va='center', fontsize=11, fontweight='bold', color='#1E293B')
    ax.text(c1_x + c1_w/2, y_src_in + 0.28, r"$(x_1, x_2, \dots, x_S)\;\;[\mathrm{\mathbf{Minority}}]$", 
            ha='center', va='center', fontsize=9.5, color='#475569')

    # Connectors Col 1 (Bottom)
    draw_straight_arrow((c1_x + c1_w/2, y_src_in + h_src_in), (c1_x + c1_w/2, y_student), 
                        color=c_student_border, lw=2.0)
    draw_straight_arrow((c1_x + c1_w/2, y_student + h_student), (c1_x + c1_w/2, y_src_h), 
                        color=c_student_border, lw=2.0)

    # -------------------------------------------------------------
    # 2. COLUMN 2: DYNAMIC POSTERIOR ALIGNMENT MATRIX (x in [4.4, 7.6])
    # -------------------------------------------------------------
    c2_x = 4.4
    c2_w = 3.2
    c2_y = 3.80
    c2_h = 2.50
    draw_box(c2_x, c2_y, c2_w, c2_h, c_align_bg, c_align_border, lw=2.0)
    
    ax.text(c2_x + c2_w/2, c2_y + 2.20, r"$\mathbf{Dynamic\;Posterior\;Alignment}$", 
            ha='center', va='center', fontsize=11, fontweight='bold', color='#065F46')
    ax.text(c2_x + c2_w/2, c2_y + 1.75, r"$M_{t,s} = \frac{\tilde{h}_t^T \cdot (\tilde{h}_s^S)^\top}{\tau_{\mathrm{align}}}$", 
            ha='center', va='center', fontsize=10.5, color='#047857')
    ax.text(c2_x + c2_w/2, c2_y + 1.25, r"$A_{t,s} = \frac{\exp(M_{t,s})}{\sum_{s'} \exp(M_{t,s'})} \in \mathbb{R}^{T \times S}$", 
            ha='center', va='center', fontsize=10.5, color='#047857')
    
    # Marginals sub-box
    draw_box(c2_x + 0.2, c2_y + 0.15, c2_w - 0.4, 0.75, '#D1FAE5', '#10B981', lw=1.0)
    ax.text(c2_x + c2_w/2, c2_y + 0.62, r"$\mathbf{Confidence\;Marginals:}$", 
            ha='center', va='center', fontsize=9.5, fontweight='bold', color='#065F46')
    ax.text(c2_x + c2_w/2, c2_y + 0.32, r"$c_s = \sum_{t=1}^T A_{t,s}$", 
            ha='center', va='center', fontsize=10.5, color='#065F46')

    # Direct Clean Inputs from Col 1 to Alignment Matrix
    draw_straight_arrow((c1_x + c1_w, y_tgt_h + h_tgt_h*0.5), (c2_x, y_tgt_h + h_tgt_h*0.5), 
                        color=c_teacher_border, lw=1.8, ls='--')
    draw_straight_arrow((c1_x + c1_w, y_src_h + h_src_h*0.5), (c2_x, y_src_h + h_src_h*0.5), 
                        color=c_student_border, lw=2.0)

    # -------------------------------------------------------------
    # 3. COLUMN 3: THREE ANCHORING MODULES & CROSS-ATTENTION (x in [8.4, 12.5])
    # -------------------------------------------------------------
    c3_x = 8.35
    c3_w = 4.15

    # Module 1: Target Barycenter Anchoring
    y_bary = 7.15
    h_bary = 1.85
    draw_box(c3_x, y_bary, c3_w, h_bary, c_align_bg, c_align_border, lw=2.0)
    ax.text(c3_x + c3_w/2, y_bary + 1.55, r"$\mathbf{Target\;Barycenter\;Anchoring}$", 
            ha='center', va='center', fontsize=11.5, fontweight='bold', color='#065F46')
    ax.text(c3_x + c3_w/2, y_bary + 1.18, r"$\bar{h}_s^T = \sum_{t=1}^T A_{t,s} h_t^T\;\;\;(\mathbf{Semantic\;Target})$", 
            ha='center', va='center', fontsize=10, color='#047857')
    
    # Projector box
    draw_box(c3_x + 0.2, y_bary + 0.15, c3_w - 0.4, 0.75, '#FFFFFF', '#059669', lw=1.0)
    ax.text(c3_x + c3_w/2, y_bary + 0.65, r"$\mathbf{Residual\;Semantic\;Projector}\;\Phi_\theta:$", 
            ha='center', va='center', fontsize=9.5, fontweight='bold', color='#065F46')
    ax.text(c3_x + c3_w/2, y_bary + 0.35, r"$\hat{h}_s^S = h_s^S + \mathrm{GELU}(\mathrm{LN}(h_s^S)W_1)W_2$", 
            ha='center', va='center', fontsize=10, color='#047857')

    # Module 2: Sentence Contrastive Priming
    y_prime = 5.25
    h_prime = 1.45
    draw_box(c3_x, y_prime, c3_w, h_prime, c_prime_bg, c_prime_border, lw=2.0)
    ax.text(c3_x + c3_w/2, y_prime + 1.15, r"$\mathbf{Sentence\;Contrastive\;Priming}$", 
            ha='center', va='center', fontsize=11.5, fontweight='bold', color='#115E59')
    ax.text(c3_x + c3_w/2, y_prime + 0.72, r"$\bar{z}_S = \mathrm{MeanPool}(h^S),\quad \bar{z}_T = \mathrm{MeanPool}(h^T)$", 
            ha='center', va='center', fontsize=10, color='#0F766E')
    ax.text(c3_x + c3_w/2, y_prime + 0.32, "Global Isotropic Cross-Lingual Semantic Space", 
            ha='center', va='center', fontsize=9, fontstyle='italic', color='#134E4A')

    # Module 3: Dynamic Head Routing Gate
    y_route = 3.00
    h_route = 1.80
    draw_box(c3_x, y_route, c3_w, h_route, c_router_bg, c_router_border, lw=2.0)
    ax.text(c3_x + c3_w/2, y_route + 1.50, r"$\mathbf{Dynamic\;Head\;Routing\;Gate}\;g_h$", 
            ha='center', va='center', fontsize=11.5, fontweight='bold', color='#5B21B6')
    ax.text(c3_x + c3_w/2, y_route + 1.15, r"$g_h = \sigma(w_r^\top \bar{h}^S + b_r) \in [0, 1]$", 
            ha='center', va='center', fontsize=10, color='#6D28D9')
    ax.text(c3_x + c3_w/2, y_route + 0.78, r"$\mathrm{Teacher\;Agreement:}\;\;\bar{\alpha}_h = \mathrm{MeanCos}(\mathrm{Attn}_h, A)$", 
            ha='center', va='center', fontsize=9.5, color='#4C1D95')
    
    # Budget sub-badge
    draw_box(c3_x + 0.25, y_route + 0.14, c3_w - 0.5, 0.45, '#EDE9FE', '#8B5CF6', lw=1.0)
    ax.text(c3_x + c3_w/2, y_route + 0.36, r"$\mathbf{Capacity\;Budget:}\;\;\frac{1}{H}\sum_{h=1}^H g_h \leq \rho^* = \min(0.333, \frac{4}{H})$", 
            ha='center', va='center', fontsize=8.8, fontweight='bold', color='#4C1D95')

    # Module 4: Decoder Cross-Attention
    y_attn = 1.00
    h_attn = 1.45
    draw_box(c3_x, y_attn, c3_w, h_attn, c_student_bg, c_student_border, lw=1.8)
    ax.text(c3_x + c3_w/2, y_attn + 1.15, r"$\mathbf{Decoder\;Cross\text{-}Attention}\;\;\mathrm{Attn}_h$", 
            ha='center', va='center', fontsize=11, fontweight='bold', color='#1E40AF')
    
    # Sub-head blocks
    draw_box(c3_x + 0.2, y_attn + 0.15, 1.8, 0.70, '#D1FAE5', '#059669', lw=1.0)
    ax.text(c3_x + 1.1, y_attn + 0.58, r"$\mathbf{Anchored}\;(g_h \approx 1)$", ha='center', va='center', fontsize=8.5, color='#065F46')
    ax.text(c3_x + 1.1, y_attn + 0.32, "Semantic Grounding\n(Zero Delimiter Sink)", ha='center', va='center', fontsize=7.2, color='#047857')

    draw_box(c3_x + 2.15, y_attn + 0.15, 1.8, 0.70, '#E2E8F0', '#64748B', lw=1.0)
    ax.text(c3_x + 3.05, y_attn + 0.58, r"$\mathbf{Free}\;(g_h \approx 0)$", ha='center', va='center', fontsize=8.5, color='#334155')
    ax.text(c3_x + 3.05, y_attn + 0.32, "Target Syntax\n& Generation Fluency", ha='center', va='center', fontsize=7.2, color='#475569')

    # Gate connection straight down to cross-attn
    draw_straight_arrow((c3_x + c3_w/2, y_route), (c3_x + c3_w/2, y_attn + h_attn), 
                        color=c_router_border, lw=2.0)

    # -------------------------------------------------------------
    # 4. COLUMN 4: LOSS OBJECTIVES & AUTOREGRESSIVE DECODER (x in [13.0, 17.0])
    # -------------------------------------------------------------
    c4_x = 13.0
    c4_w = 4.0

    # Loss 1: Structural Loss
    y_lstruct = 7.35
    h_lstruct = 1.45
    draw_box(c4_x, y_lstruct, c4_w, h_lstruct, c_loss_bg, c_loss_border, lw=2.0)
    ax.text(c4_x + c4_w/2, y_lstruct + 1.10, r"$\mathbf{Structural\;Loss}\;\;\mathcal{L}_{\mathrm{struct}}$", 
            ha='center', va='center', fontsize=11.5, fontweight='bold', color=c_loss_text)
    ax.text(c4_x + c4_w/2, y_lstruct + 0.65, r"$\frac{1}{\sum_{s} c_s} \sum_{s=1}^S c_s \cdot \mathrm{Smooth}_{L_1}(\hat{h}_s^S, \bar{h}_s^T)$", 
            ha='center', va='center', fontsize=10.5, color='#B91C1C')
    ax.text(c4_x + c4_w/2, y_lstruct + 0.25, "Confidence-Weighted Barycenter Matching", 
            ha='center', va='center', fontsize=8.5, fontstyle='italic', color='#7F1D1D')

    # Loss 2: Sentence Priming Loss
    y_lprime = 5.35
    h_lprime = 1.25
    draw_box(c4_x, y_lprime, c4_w, h_lprime, c_loss_bg, c_loss_border, lw=2.0)
    ax.text(c4_x + c4_w/2, y_lprime + 0.90, r"$\mathbf{Sentence\;Priming\;Loss}\;\;\mathcal{L}_{\mathrm{prime}}$", 
            ha='center', va='center', fontsize=11.5, fontweight='bold', color=c_loss_text)
    ax.text(c4_x + c4_w/2, y_lprime + 0.50, r"$\mathrm{In\text{-}Batch\;InfoNCE\;Objective}\;\;(\tau_{\mathrm{prime}}=0.07)$", 
            ha='center', va='center', fontsize=9.5, color='#B91C1C')
    ax.text(c4_x + c4_w/2, y_lprime + 0.20, "Anisotropic Delimiter Compression Defense", 
            ha='center', va='center', fontsize=8.5, fontstyle='italic', color='#7F1D1D')

    # Loss 3: Head Routing Loss
    y_lroute = 3.25
    h_lroute = 1.30
    draw_box(c4_x, y_lroute, c4_w, h_lroute, c_loss_bg, c_loss_border, lw=2.0)
    ax.text(c4_x + c4_w/2, y_lroute + 0.95, r"$\mathbf{Head\;Routing\;Loss}\;\;\mathcal{L}_{\mathrm{route}}$", 
            ha='center', va='center', fontsize=11.5, fontweight='bold', color=c_loss_text)
    ax.text(c4_x + c4_w/2, y_lroute + 0.55, r"$\mathrm{BCE}(g_h, \mathbf{1}(\bar{\alpha}_h \geq \tau_{\mathrm{head}}))$", 
            ha='center', va='center', fontsize=10.5, color='#B91C1C')
    ax.text(c4_x + c4_w/2, y_lroute + 0.22, "Decouples Semantic Anchoring from Fluency", 
            ha='center', va='center', fontsize=8.5, fontstyle='italic', color='#7F1D1D')

    # Autoregressive Decoder f_dec
    y_dec = 1.05
    h_dec = 1.35
    draw_box(c4_x, y_dec, c4_w, h_dec, c_student_bg, c_student_border, lw=2.0)
    ax.text(c4_x + c4_w/2, y_dec + 1.05, r"$\mathbf{Autoregressive\;Decoder}\;f_{\mathrm{dec}}$", 
            ha='center', va='center', fontsize=11.5, fontweight='bold', color='#1E40AF')
    ax.text(c4_x + c4_w/2, y_dec + 0.72, r"$\mathrm{Generates\;Target\;Tokens}\;\;\hat{\mathcal{Y}} = (\hat{y}_1, \dots, \hat{y}_T)$", 
            ha='center', va='center', fontsize=9.5, color='#2563EB')
    
    # Translation Loss MT badge
    draw_box(c4_x + 0.6, y_dec + 0.15, c4_w - 1.2, 0.40, c_loss_bg, c_loss_border, lw=1.2)
    ax.text(c4_x + c4_w/2, y_dec + 0.35, r"$\mathbf{Translation\;Loss}\;\;\mathcal{L}_{\mathrm{MT}}$", 
            ha='center', va='center', fontsize=9, fontweight='bold', color=c_loss_text)

    # -------------------------------------------------------------
    # 5. CONNECTORS: STRAIGHT HORIZONTAL BETWEEN COL 3 AND COL 4
    # -------------------------------------------------------------
    draw_straight_arrow((c3_x + c3_w, y_bary + h_bary*0.5), (c4_x, y_lstruct + h_lstruct*0.5), 
                        color=c_align_border, lw=2.0)
    draw_straight_arrow((c3_x + c3_w, y_prime + h_prime*0.5), (c4_x, y_lprime + h_lprime*0.5), 
                        color=c_prime_border, lw=2.0)
    draw_straight_arrow((c3_x + c3_w, y_route + h_route*0.5), (c4_x, y_lroute + h_lroute*0.5), 
                        color=c_router_border, lw=2.0)
    draw_straight_arrow((c3_x + c3_w, y_attn + h_attn*0.5), (c4_x, y_dec + h_dec*0.5), 
                        color=c_student_border, lw=2.0)

    # -------------------------------------------------------------
    # 6. DEDICATED ORTHOGONAL BUS ROUTING (ZERO CROSSINGS THROUGH BOXES!)
    # -------------------------------------------------------------
    
    # Track A: Target States h^T bypasses Col 2 ABOVE (y = 6.65)
    # Bus line coordinate x = 7.80, runway = 0.55 into Col 3
    bus_x_tgt = 7.78
    draw_path([
        (3.90, y_tgt_h + h_tgt_h*0.5),
        (3.90, 6.65),
        (bus_x_tgt, 6.65),
        (bus_x_tgt, y_bary + 1.20),
        (c3_x, y_bary + 1.20)
    ], color=c_teacher_border, lw=1.6, ls='--')

    # Also feeds Sentence Contrastive Priming from the bus
    draw_path([
        (bus_x_tgt, 6.65),
        (bus_x_tgt, y_prime + 1.05),
        (c3_x, y_prime + 1.05)
    ], color=c_teacher_border, lw=1.6, ls='--')

    # Track B: Source States h^S bypasses Col 2 BELOW (y = 2.65)
    bus_x_src = 7.96
    draw_path([
        (4.05, y_src_h + h_src_h*0.5),
        (4.05, 2.65),
        (bus_x_src, 2.65),
        (bus_x_src, y_bary + 0.35),
        (c3_x, y_bary + 0.35)
    ], color=c_student_border, lw=1.6)

    # Feeds Sentence Priming
    draw_path([
        (bus_x_src, 2.65),
        (bus_x_src, y_prime + 0.45),
        (c3_x, y_prime + 0.45)
    ], color=c_student_border, lw=1.6)

    # Feeds Head Router
    draw_path([
        (bus_x_src, 2.65),
        (bus_x_src, y_route + 0.60),
        (c3_x, y_route + 0.60)
    ], color=c_student_border, lw=1.6)

    # Feeds Decoder Cross-Attention
    draw_path([
        (bus_x_src, 2.65),
        (bus_x_src, y_attn + 0.60),
        (c3_x, y_attn + 0.60)
    ], color=c_student_border, lw=1.6)

    # Track C: Alignment Matrix A outputs to Col 3 (in corridor x in [7.6, 8.35])
    # A -> Barycenter
    draw_path([
        (c2_x + c2_w, c2_y + 1.80),
        (8.14, c2_y + 1.80),
        (8.14, y_bary + 0.75),
        (c3_x, y_bary + 0.75)
    ], color=c_align_border, lw=1.8)

    # A -> Router (Agreement)
    draw_path([
        (c2_x + c2_w, c2_y + 0.50),
        (8.14, c2_y + 0.50),
        (8.14, y_route + 1.15),
        (c3_x, y_route + 1.15)
    ], color=c_align_border, lw=1.8)

    # -------------------------------------------------------------
    # 7. BOTTOM BANNER: TOTAL MULTI-TASK LOSS OBJECTIVE
    # -------------------------------------------------------------
    banner_x = 0.5
    banner_w = 16.5
    banner_y = 0.12
    banner_h = 0.65
    
    draw_box(banner_x, banner_y, banner_w, banner_h, '#F8FAFC', '#2563EB', lw=1.8)
    banner_text = (
        r"$\mathbf{Total\;Multi\text{-}Task\;Training\;Objective:}\quad "
        r"\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{MT}} + "
        r"\lambda_{\mathrm{struct}}\mathcal{L}_{\mathrm{struct}} + "
        r"\lambda_{\mathrm{prime}}\mathcal{L}_{\mathrm{prime}} + "
        r"\lambda_{\mathrm{route}}\mathcal{L}_{\mathrm{route}}\quad "
        r"(\lambda_{\mathrm{struct}}=0.5,\;\lambda_{\mathrm{prime}}=0.1,\;\lambda_{\mathrm{route}}=0.2)$"
    )
    ax.text(banner_x + banner_w/2, banner_y + banner_h/2, banner_text, 
            ha='center', va='center', fontsize=11.5, fontweight='bold', color='#1E3A8A')

    # Output paths
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'docs', 'paper', 'latex', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    
    pdf_path = os.path.join(out_dir, 'tssa_architecture.pdf')
    png_path = os.path.join(out_dir, 'tssa_architecture.png')
    
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0.05)
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"Successfully generated:\n  {pdf_path}\n  {png_path}")

if __name__ == '__main__':
    create_tssa_architecture_diagram()
