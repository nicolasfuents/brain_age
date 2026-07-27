import os
import numpy as np

def generate_gaussian_label_numpy(age, num_classes=100, sigma=1.5):
    x = np.arange(num_classes)
    dist = np.exp(-0.5 * ((x - age) / sigma) ** 2)
    return dist / dist.sum()

def main():
    fig_dir = "/home/nfuentes/scratch/brain_age_project/openBHB_dataset/scripts/manuscript/paper/figures"
    os.makedirs(fig_dir, exist_ok=True)

    svg_path = os.path.join(fig_dir, "fig03_soft_labels_comparison.svg")
    png_path = os.path.join(fig_dir, "fig03_soft_labels_comparison.png")

    # Generate discrete data for 100 bins
    num_classes = 100
    target_age = 65

    # One-Hot vector
    one_hot = np.zeros(num_classes)
    one_hot[target_age] = 1.0

    # Soft Gaussian label vector (sigma = 1.8 for clear visual bars around age 65)
    soft_dist = generate_gaussian_label_numpy(target_age, num_classes=100, sigma=1.8)

    # Palette definition (Matching Q1 manuscript theme)
    c_dark      = "#0f172a"
    c_slate     = "#475569"
    c_red       = "#dc2626"
    c_red_light = "#fef2f2"
    c_red_bdr   = "#fca5a5"
    c_green     = "#059669"
    c_green_lt  = "#ecfdf5"
    c_green_bdr = "#6ee7b7"
    c_blue      = "#0284c7"
    c_blue_lt   = "#f0f9ff"
    c_blue_bdr  = "#7dd3fc"

    # SVG Canvas dimensions
    width = 1800
    height = 1060

    # Render bars for Panel A (One-Hot)
    def render_one_hot_bars():
        bars_svg = []
        x_min, x_max = 100, 820
        y_bottom, y_top = 410, 200
        w_bar = (x_max - x_min) / num_classes

        for i in range(num_classes):
            x_pos = x_min + i * w_bar
            val = one_hot[i]
            if val > 0:
                h_bar = (y_bottom - y_top) * val
                bars_svg.append(f'<rect x="{x_pos:.1f}" y="{y_bottom - h_bar:.1f}" width="{max(w_bar, 4.0):.1f}" height="{h_bar:.1f}" fill="{c_red}" rx="1"/>')
                bars_svg.append(f'<text x="{x_pos + w_bar/2:.1f}" y="{y_bottom - h_bar - 8:.1f}" font-size="14" font-weight="bold" fill="{c_red}" text-anchor="middle">1.0</text>')
            else:
                bars_svg.append(f'<rect x="{x_pos:.1f}" y="{y_bottom - 1:.1f}" width="{w_bar:.1f}" height="2" fill="#cbd5e1"/>')
        return "".join(bars_svg)

    # Render bars for Panel B (Soft Labels)
    def render_soft_bars():
        bars_svg = []
        x_min, x_max = 980, 1700
        y_bottom, y_top = 410, 200
        w_bar = (x_max - x_min) / num_classes
        max_val = np.max(soft_dist)

        for i in range(num_classes):
            x_pos = x_min + i * w_bar
            val = soft_dist[i]
            if val > 0.001:
                norm_h = val / max_val
                h_bar = (y_bottom - y_top) * norm_h
                color = c_green if abs(i - target_age) <= 1 else ("#10b981" if abs(i - target_age) <= 3 else "#34d399")
                bars_svg.append(f'<rect x="{x_pos:.1f}" y="{y_bottom - h_bar:.1f}" width="{max(w_bar, 3.5):.1f}" height="{h_bar:.1f}" fill="{color}" rx="1"/>')
            else:
                bars_svg.append(f'<rect x="{x_pos:.1f}" y="{y_bottom - 1:.1f}" width="{w_bar:.1f}" height="2" fill="#cbd5e1"/>')

        curve_pts = []
        for i in range(num_classes):
            x_pos = x_min + i * w_bar + w_bar/2
            val = soft_dist[i]
            norm_h = val / max_val
            y_pos = y_bottom - (y_bottom - y_top) * norm_h
            curve_pts.append(f"{x_pos:.1f},{y_pos:.1f}")

        curve_path = f'<path d="M {" L ".join(curve_pts)}" fill="none" stroke="#047857" stroke-width="2.5"/>'
        x_peak = x_min + target_age * w_bar + w_bar/2
        peak_text = f'<text x="{x_peak:.1f}" y="{y_top - 8:.1f}" font-size="14" font-weight="bold" fill="{c_green}" text-anchor="middle">Peak at y = 65 (y_target &#8776; 0.22)</text>'

        return "".join(bars_svg) + curve_path + peak_text

    # Loss curve comparison points for Panel C
    delta_x = np.linspace(-10, 10, 200)
    ce_loss = np.where(np.abs(delta_x) < 0.5, 0.0, 4.2)
    # Scaled kl_loss so max penalty at dx = ±10 is 4.2 (fits perfectly within axes)
    kl_loss = 0.042 * (delta_x ** 2)

    def render_loss_curves():
        x_min, x_max = 140, 800
        y_bottom, y_top = 920, 670
        x_center = (x_min + x_max) / 2

        def map_x(dx):
            return x_center + (dx / 10.0) * ((x_max - x_min) / 2)

        def map_y(loss_val):
            return y_bottom - (loss_val / 5.0) * (y_bottom - y_top)

        ce_pts = []
        for dx, l in zip(delta_x, ce_loss):
            ce_pts.append(f"{map_x(dx):.1f},{map_y(l):.1f}")
        ce_path = f'<path d="M {" L ".join(ce_pts)}" fill="none" stroke="{c_red}" stroke-width="3" stroke-dasharray="6,4"/>'

        kl_pts = []
        for dx, l in zip(delta_x, kl_loss):
            kl_pts.append(f"{map_x(dx):.1f},{map_y(l):.1f}")
        kl_path = f'<path d="M {" L ".join(kl_pts)}" fill="none" stroke="{c_green}" stroke-width="3.5"/>'

        return ce_path + kl_path

    svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%" style="background-color: #ffffff; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;">
  <defs>
    <!-- Drop Shadow Filter -->
    <filter id="shadow" x="-4%" y="-4%" width="108%" height="108%">
      <feDropShadow dx="2" dy="4" stdDeviation="5" flood-color="#000000" flood-opacity="0.06"/>
    </filter>

    <!-- Arrow Markers -->
    <marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#475569"/>
    </marker>
    <marker id="arrow-green" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="{c_green}"/>
    </marker>
    <marker id="arrow-blue" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="{c_blue}"/>
    </marker>
  </defs>

  <!-- Background -->
  <rect width="{width}" height="{height}" fill="#ffffff" />

  <!-- MAIN TITLE & SUBTITLE -->
  <text x="{width/2}" y="42" font-size="26" font-weight="bold" fill="{c_dark}" text-anchor="middle">Target Representation &amp; Loss Optimization: One-Hot vs. Soft Gaussian Labels</text>
  <text x="{width/2}" y="70" font-size="15" font-weight="500" fill="{c_slate}" text-anchor="middle">Comparison of Ordinal-Aware Gaussian Distribution Modeling vs. Hard Discrete Classification in Brain Age Prediction</text>

  <!-- ========================================================================= -->
  <!-- PANEL A: TRADITIONAL ONE-HOT ENCODING -->
  <!-- ========================================================================= -->
  <g transform="translate(0, 0)">
    <!-- Container Box -->
    <rect x="50" y="100" width="820" height="465" rx="12" ry="12" fill="{c_red_light}" stroke="{c_red_bdr}" stroke-width="1.8" filter="url(#shadow)"/>
    
    <!-- Panel Header -->
    <rect x="50" y="100" width="820" height="45" rx="12" ry="12" fill="#fee2e2"/>
    <rect x="50" y="133" width="820" height="12" fill="#fee2e2"/>
    <text x="70" y="129" font-size="17" font-weight="bold" fill="#991b1b">(A) Traditional Hard One-Hot Encoding (Ordinal-Unaware)</text>
    <text x="850" y="129" font-size="14" font-weight="bold" fill="#991b1b" text-anchor="end">Target Age: y = 65.0</text>

    <!-- Plot Axes -->
    <line x1="100" y1="410" x2="100" y2="190" stroke="{c_slate}" stroke-width="1.5"/>
    <text x="90" y="200" font-size="12" font-weight="bold" fill="{c_slate}" text-anchor="end">1.0</text>
    <text x="90" y="305" font-size="12" font-weight="bold" fill="{c_slate}" text-anchor="end">0.5</text>
    <text x="90" y="410" font-size="12" font-weight="bold" fill="{c_slate}" text-anchor="end">0.0</text>
    <text x="55" y="300" font-size="13" font-weight="bold" fill="{c_slate}" text-anchor="middle" transform="rotate(-90 55 300)">Target Probability y_k</text>

    <!-- X-axis -->
    <line x1="100" y1="410" x2="820" y2="410" stroke="{c_slate}" stroke-width="1.5"/>
    <text x="100" y="430" font-size="12" font-weight="bold" fill="{c_slate}" text-anchor="middle">0</text>
    <text x="244" y="430" font-size="12" font-weight="bold" fill="{c_slate}" text-anchor="middle">20</text>
    <text x="388" y="430" font-size="12" font-weight="bold" fill="{c_slate}" text-anchor="middle">40</text>
    <text x="532" y="430" font-size="12" font-weight="bold" fill="{c_slate}" text-anchor="middle">60</text>
    <text x="568" y="430" font-size="13" font-weight="bold" fill="{c_red}" text-anchor="middle">65</text>
    <text x="676" y="430" font-size="12" font-weight="bold" fill="{c_slate}" text-anchor="middle">80</text>
    <text x="820" y="430" font-size="12" font-weight="bold" fill="{c_slate}" text-anchor="middle">99</text>
    <text x="460" y="450" font-size="14" font-weight="bold" fill="{c_slate}" text-anchor="middle">Discrete Age Bins (k &#8712; [0, 99])</text>

    <!-- Bars -->
    {render_one_hot_bars()}

    <!-- Key Limitations Cards -->
    <g transform="translate(70, 472)">
      <!-- Card 1: Ordinal Blindness -->
      <rect x="0" y="0" width="245" height="75" rx="8" ry="8" fill="#ffffff" stroke="#fca5a5" stroke-width="1.2"/>
      <text x="12" y="22" font-size="13" font-weight="bold" fill="#991b1b">&#9888; Ordinal Blindness</text>
      <text x="12" y="42" font-size="11.5" fill="{c_slate}">Bin 64 (error = 1 yr) &amp; Bin 10</text>
      <text x="12" y="58" font-size="11.5" fill="{c_slate}">(error = 55 yrs) receive probability 0.</text>

      <!-- Card 2: Quantization Error -->
      <rect x="260" y="0" width="245" height="75" rx="8" ry="8" fill="#ffffff" stroke="#fca5a5" stroke-width="1.2"/>
      <text x="272" y="22" font-size="13" font-weight="bold" fill="#991b1b">&#9888; Quantization Error</text>
      <text x="272" y="42" font-size="11.5" fill="{c_slate}">Forces continuous age into integer</text>
      <text x="272" y="58" font-size="11.5" fill="{c_slate}">bins. Error bound: &#177;0.5 years.</text>

      <!-- Card 3: Hard Decision -->
      <rect x="520" y="0" width="260" height="75" rx="8" ry="8" fill="#ffffff" stroke="#fca5a5" stroke-width="1.2"/>
      <text x="532" y="22" font-size="13" font-weight="bold" fill="#991b1b">&#9888; Argmax Hard Decision</text>
      <text x="532" y="42" font-size="11.5" fill="{c_slate}">y_pred = argmax(p). Non-differentiable</text>
      <text x="532" y="58" font-size="11.5" fill="{c_slate}">step function, no decimal resolution.</text>
    </g>
  </g>

  <!-- ========================================================================= -->
  <!-- PANEL B: PROPOSED SOFT GAUSSIAN LABELS -->
  <!-- ========================================================================= -->
  <g transform="translate(0, 0)">
    <!-- Container Box -->
    <rect x="930" y="100" width="820" height="465" rx="12" ry="12" fill="{c_green_lt}" stroke="{c_green_bdr}" stroke-width="1.8" filter="url(#shadow)"/>
    
    <!-- Panel Header -->
    <rect x="930" y="100" width="820" height="45" rx="12" ry="12" fill="#d1fae5"/>
    <rect x="930" y="133" width="820" height="12" fill="#d1fae5"/>
    <text x="950" y="129" font-size="17" font-weight="bold" fill="#065f46">(B) Proposed Soft Gaussian Label Distribution (Ours)</text>
    <text x="1730" y="129" font-size="14" font-weight="bold" fill="#065f46" text-anchor="end">Target: y = 65.0 | &#963; = 1.8</text>

    <!-- Plot Axes -->
    <line x1="980" y1="410" x2="980" y2="190" stroke="{c_slate}" stroke-width="1.5"/>
    <text x="970" y="200" font-size="12" font-weight="bold" fill="{c_slate}" text-anchor="end">0.25</text>
    <text x="970" y="305" font-size="12" font-weight="bold" fill="{c_slate}" text-anchor="end">0.12</text>
    <text x="970" y="410" font-size="12" font-weight="bold" fill="{c_slate}" text-anchor="end">0.00</text>
    <text x="935" y="300" font-size="13" font-weight="bold" fill="{c_slate}" text-anchor="middle" transform="rotate(-90 935 300)">Target Density y_k</text>

    <!-- X-axis -->
    <line x1="980" y1="410" x2="1700" y2="410" stroke="{c_slate}" stroke-width="1.5"/>
    <text x="980" y="430" font-size="12" font-weight="bold" fill="{c_slate}" text-anchor="middle">0</text>
    <text x="1124" y="430" font-size="12" font-weight="bold" fill="{c_slate}" text-anchor="middle">20</text>
    <text x="1268" y="430" font-size="12" font-weight="bold" fill="{c_slate}" text-anchor="middle">40</text>
    <text x="1412" y="430" font-size="12" font-weight="bold" fill="{c_slate}" text-anchor="middle">60</text>
    <text x="1448" y="430" font-size="13" font-weight="bold" fill="{c_green}" text-anchor="middle">65</text>
    <text x="1556" y="430" font-size="12" font-weight="bold" fill="{c_slate}" text-anchor="middle">80</text>
    <text x="1700" y="430" font-size="12" font-weight="bold" fill="{c_slate}" text-anchor="middle">99</text>
    <text x="1340" y="450" font-size="14" font-weight="bold" fill="{c_slate}" text-anchor="middle">Discrete Age Bins (k &#8712; [0, 99])</text>

    <!-- Bars & Curve -->
    {render_soft_bars()}

    <!-- Key Advantages Cards -->
    <g transform="translate(950, 472)">
      <!-- Card 1: Gaussian Label Formula -->
      <rect x="0" y="0" width="245" height="75" rx="8" ry="8" fill="#ffffff" stroke="#6ee7b7" stroke-width="1.2"/>
      <text x="12" y="22" font-size="13" font-weight="bold" fill="#065f46">&#10004; Ordinal Soft Labeling</text>
      <text x="12" y="42" font-size="11.5" fill="{c_slate}">y_k &#8733; exp( -(k - y)&#178; / 2&#963;&#178; )</text>
      <text x="12" y="58" font-size="11.5" fill="{c_slate}">Penalizes errors proportional to distance.</text>

      <!-- Card 2: Soft-argmax Expectation -->
      <rect x="260" y="0" width="245" height="75" rx="8" ry="8" fill="#ffffff" stroke="#6ee7b7" stroke-width="1.2"/>
      <text x="272" y="22" font-size="13" font-weight="bold" fill="#065f46">&#10004; Continuous Expectation</text>
      <text x="272" y="42" font-size="11.5" fill="{c_slate}">y_pred = E[age] = &#8721; k &#183; p(k)</text>
      <text x="272" y="58" font-size="11.5" fill="{c_slate}">Unlocks sub-bin decimal age resolution.</text>

      <!-- Card 3: KL Divergence Loss -->
      <rect x="520" y="0" width="260" height="75" rx="8" ry="8" fill="#ffffff" stroke="#6ee7b7" stroke-width="1.2"/>
      <text x="532" y="22" font-size="13" font-weight="bold" fill="#065f46">&#10004; KL Divergence Loss</text>
      <text x="532" y="42" font-size="11.5" fill="{c_slate}">L_KL = &#8721; y_k log( y_k / p_k )</text>
      <text x="532" y="58" font-size="11.5" fill="{c_slate}">Fully end-to-end gradient optimization.</text>
    </g>
  </g>

  <!-- ========================================================================= -->
  <!-- PANEL C: LOSS BEHAVIOR & PIPELINE WORKFLOW -->
  <!-- ========================================================================= -->
  <g transform="translate(0, 0)">
    <!-- Left Box: Loss & Gradient Curve Comparison -->
    <rect x="50" y="595" width="820" height="425" rx="12" ry="12" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.8" filter="url(#shadow)"/>
    
    <rect x="50" y="595" width="820" height="42" rx="12" ry="12" fill="#e2e8f0"/>
    <rect x="50" y="625" width="820" height="12" fill="#e2e8f0"/>
    <text x="70" y="622" font-size="16" font-weight="bold" fill="{c_dark}">(C) Loss Surface &amp; Distance Penalty Comparison</text>

    <!-- Axes for Loss Plot -->
    <line x1="140" y1="920" x2="140" y2="670" stroke="{c_slate}" stroke-width="1.5"/>
    <line x1="140" y1="920" x2="800" y2="920" stroke="{c_slate}" stroke-width="1.5"/>
    <line x1="470" y1="920" x2="470" y2="670" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="3,3"/>

    <text x="130" y="680" font-size="12" font-weight="bold" fill="{c_slate}" text-anchor="end">High</text>
    <text x="130" y="920" font-size="12" font-weight="bold" fill="{c_slate}" text-anchor="end">0.0</text>
    <text x="95" y="795" font-size="13" font-weight="bold" fill="{c_slate}" text-anchor="middle" transform="rotate(-90 95 795)">Loss Penalty L</text>

    <text x="140" y="942" font-size="12" font-weight="bold" fill="{c_slate}" text-anchor="middle">-10</text>
    <text x="305" y="942" font-size="12" font-weight="bold" fill="{c_slate}" text-anchor="middle">-5</text>
    <text x="470" y="942" font-size="13" font-weight="bold" fill="{c_dark}" text-anchor="middle">0 (Target y)</text>
    <text x="635" y="942" font-size="12" font-weight="bold" fill="{c_slate}" text-anchor="middle">+5</text>
    <text x="800" y="942" font-size="12" font-weight="bold" fill="{c_slate}" text-anchor="middle">+10</text>
    <text x="470" y="968" font-size="14" font-weight="bold" fill="{c_slate}" text-anchor="middle">Prediction Error &#916;y = y_pred - y (Years)</text>

    <!-- Render Curves -->
    {render_loss_curves()}

    <!-- Legend Box -->
    <rect x="520" y="650" width="260" height="70" rx="6" ry="6" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>
    <line x1="535" y1="670" x2="570" y2="670" stroke="{c_red}" stroke-width="3" stroke-dasharray="6,4"/>
    <text x="580" y="674" font-size="12" font-weight="bold" fill="#991b1b">One-Hot + Cross Entropy</text>
    <text x="580" y="687" font-size="10.5" fill="{c_slate}">(Step penalty, no distance sensitivity)</text>

    <line x1="535" y1="705" x2="570" y2="705" stroke="{c_green}" stroke-width="3.5"/>
    <text x="580" y="709" font-size="12" font-weight="bold" fill="#065f46">Soft Label + KL Divergence</text>
    <text x="580" y="722" font-size="10.5" fill="{c_slate}">(Smooth, distance-proportional penalty)</text>

    <!-- Right Box: Code Workflow Execution (Matching train_improved_ipw.py) -->
    <rect x="930" y="595" width="820" height="425" rx="12" ry="12" fill="{c_blue_lt}" stroke="{c_blue_bdr}" stroke-width="1.8" filter="url(#shadow)"/>
    
    <rect x="930" y="595" width="820" height="42" rx="12" ry="12" fill="#e0f2fe"/>
    <rect x="930" y="625" width="820" height="12" fill="#e0f2fe"/>
    <text x="950" y="622" font-size="16" font-weight="bold" fill="#0369a1">(D) End-to-End Implementation Flow in `train_improved_ipw.py`</text>

    <!-- Workflow Steps Boxes -->
    <!-- Step 1 -->
    <rect x="960" y="650" width="760" height="70" rx="8" ry="8" fill="#ffffff" stroke="{c_blue_bdr}" stroke-width="1.2"/>
    <rect x="970" y="660" width="30" height="30" rx="15" ry="15" fill="{c_blue}"/>
    <text x="985" y="680" font-size="14" font-weight="bold" fill="#ffffff" text-anchor="middle">1</text>
    <text x="1015" y="672" font-size="14" font-weight="bold" fill="#0369a1">Target Generation: `generate_gaussian_label(age, sigma=1.0)`</text>
    <text x="1015" y="692" font-size="12" fill="{c_slate}">Computes dist = exp(-0.5 * ((x - age) / sigma)&#178;) and normalizes over 100 age classes.</text>

    <path d="M 1340 720 L 1340 735" fill="none" stroke="{c_blue}" stroke-width="2" marker-end="url(#arrow-blue)"/>

    <!-- Step 2 -->
    <rect x="960" y="735" width="760" height="70" rx="8" ry="8" fill="#ffffff" stroke="{c_blue_bdr}" stroke-width="1.2"/>
    <rect x="970" y="745" width="30" height="30" rx="15" ry="15" fill="{c_blue}"/>
    <text x="985" y="765" font-size="14" font-weight="bold" fill="#ffffff" text-anchor="middle">2</text>
    <text x="1015" y="757" font-size="14" font-weight="bold" fill="#0369a1">Model Output: `logits = model(xinput)` &#8594; [Batch, 100]</text>
    <text x="1015" y="777" font-size="12" fill="{c_slate}">Global-Local Transformer outputs 100 raw logits for Global and Local patch heads.</text>

    <path d="M 1340 805 L 1340 820" fill="none" stroke="{c_blue}" stroke-width="2" marker-end="url(#arrow-blue)"/>

    <!-- Step 3 -->
    <rect x="960" y="820" width="760" height="70" rx="8" ry="8" fill="#ffffff" stroke="{c_blue_bdr}" stroke-width="1.2"/>
    <rect x="970" y="830" width="30" height="30" rx="15" ry="15" fill="{c_blue}"/>
    <text x="985" y="850" font-size="14" font-weight="bold" fill="#ffffff" text-anchor="middle">3</text>
    <text x="1015" y="842" font-size="14" font-weight="bold" fill="#0369a1">Loss Optimization: `nn.KLDivLoss()(F.log_softmax(logits), soft_labels)`</text>
    <text x="1015" y="862" font-size="12" fill="{c_slate}">Optimizes log-probabilities against Gaussian soft targets using IPW reweighting.</text>

    <path d="M 1340 890 L 1340 905" fill="none" stroke="{c_blue}" stroke-width="2" marker-end="url(#arrow-blue)"/>

    <!-- Step 4 -->
    <rect x="960" y="905" width="760" height="70" rx="8" ry="8" fill="#ffffff" stroke="{c_blue_bdr}" stroke-width="1.2"/>
    <rect x="970" y="915" width="30" height="30" rx="15" ry="15" fill="{c_blue}"/>
    <text x="985" y="935" font-size="14" font-weight="bold" fill="#ffffff" text-anchor="middle">4</text>
    <text x="1015" y="927" font-size="14" font-weight="bold" fill="#0369a1">Continuous Inference Decoding: `decode_age(logits) = (probs * bins).sum()`</text>
    <text x="1015" y="947" font-size="12" fill="{c_slate}">Calculates expectation E[age] over 100 bins for high-precision continuous age output.</text>
  </g>
</svg>"""

    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg_content)

    os.system(f"rsvg-convert -b '#ffffff' -f png -o {png_path} {svg_path}")
    print("Done. Generated clean fig03_soft_labels_comparison.png and .svg")

if __name__ == "__main__":
    main()
