import os
import base64

def get_base64_image(path):
    with open(path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:image/png;base64,{encoded}"

def main():
    fig_dir = "/home/nfuentes/scratch/brain_age_project/openBHB_dataset/scripts/manuscript/paper/figures"
    extracted_dir = os.path.join(fig_dir, "extracted")

    axial_b64    = get_base64_image(os.path.join(extracted_dir, "axial.png"))
    coronal_b64  = get_base64_image(os.path.join(extracted_dir, "coronal.png"))
    sagittal_b64 = get_base64_image(os.path.join(extracted_dir, "sagittal.png"))

    # =========================================================================
    # PARAMETROS DE PERSONALIZACION DE PREDICCION FINAL
    # =========================================================================
    # Podés ajustar 'FINAL_PRED_BG_OPACITY' entre "0.0" (totalmente transparente) y "1.0" (opaco total)
    FINAL_PRED_BG_COLOR   = "#cbd5e1"  # Color gris de fondo para el círculo de predicción final
    FINAL_PRED_BG_OPACITY = "0.15"     # <<< PARAMETRO DE OPACIDAD (Ajustable por el usuario)

    # =========================================================================
    # COORDINATE SYSTEM & GRID LAYOUT (Canvas width: 2050, height: 1080)
    #
    # Column Center X Coordinates (Headers perfectly centered above blocks):
    #   1. INPUT STACK:              X = 135   (Box x: 60..210)
    #   2. DUAL-PATHWAY EXTRACTION:  X = 380   (Box x: 270..490)  [Gap: 60px]
    #   3. CROSS-ATTENTION XFMR:     X = 675   (Box x: 560..790)  [Gap: 70px]
    #   4. MULTI-HEAD PREDICTION:    X = 950   (Box x: 860..1040) [Gap: 70px]
    #   5. ACTIVATION PROCESSING:    X = 1185  (Box x: 1100..1270)[Gap: 60px]
    #   6. PARTIAL PREDICTION:       X = 1380  (Box x: 1330..1430)[Gap: 60px]
    #   7. STACKER & FINAL:          X = 1630 / 1900
    # =========================================================================

    def render_pathway(plane_name, dims, backbone, nblocks, grid, loss_type, act_type, ysub, b64_img):
        c_slate   = "#475569"
        c_blue    = "#1e3a5f"
        c_sky_hdr = "#0369a1"
        c_purple  = "#4c1d5c"
        c_green   = "#1e5f1e"
        c_gold    = "#7f6000"
        c_brown   = "#5f3a1e"

        # Determine activation block SVG & symmetric connectors to mu (at y=210)
        if act_type == "soft_argmax":
            act_block = f"""
    <!-- Activation Processing (Softmax + Soft-argmax) -->
    <g filter="url(#shadow)">
      <rect x="1100" y="125" width="170" height="80" rx="8" ry="8" fill="#f8fafc" stroke="#64748b" stroke-width="1.5"/>
      <text x="1185" y="156" font-size="14" font-weight="bold" fill="#0f172a" text-anchor="middle">Softmax &amp;</text>
      <text x="1185" y="176" font-size="14" font-weight="bold" fill="#0f172a" text-anchor="middle">Soft-argmax</text>
      <text x="1185" y="194" font-size="11" font-weight="bold" fill="#475569" text-anchor="middle">Discrete &#8594; Continuous</text>

      <rect x="1100" y="215" width="170" height="80" rx="8" ry="8" fill="#f8fafc" stroke="#64748b" stroke-width="1.5"/>
      <text x="1185" y="246" font-size="14" font-weight="bold" fill="#0f172a" text-anchor="middle">Softmax &amp;</text>
      <text x="1185" y="266" font-size="14" font-weight="bold" fill="#0f172a" text-anchor="middle">Soft-argmax</text>
      <text x="1185" y="284" font-size="11" font-weight="bold" fill="#475569" text-anchor="middle">Discrete &#8594; Continuous</text>
    </g>

    <!-- Connectors: Heads -> Activation -->
    <path d="M 1040 165 L 1100 165" fill="none" stroke="{c_slate}" stroke-width="1.5" marker-end="url(#arrow)"/>
    <path d="M 1040 255 L 1100 255" fill="none" stroke="{c_slate}" stroke-width="1.5" marker-end="url(#arrow)"/>

    <!-- Connectors: Activation -> Mean node (Symmetric entry at y=210: dy=-45 and dy=+45) -->
    <path d="M 1270 165 C 1315 165, 1315 210, 1330 210" fill="none" stroke="{c_slate}" stroke-width="1.5" marker-end="url(#arrow)"/>
    <path d="M 1270 255 C 1315 255, 1315 210, 1330 210" fill="none" stroke="{c_slate}" stroke-width="1.5" marker-end="url(#arrow)"/>"""
        else:
            # Direct regression (Smooth L1 or MSE) - symmetric flow straight to mean node at y=210
            act_block = f"""
    <!-- Direct Regression Flow (Symmetric entry at y=210: dy=-45 and dy=+45) -->
    <path d="M 1040 165 C 1220 165, 1290 210, 1330 210" fill="none" stroke="{c_slate}" stroke-width="1.5" marker-end="url(#arrow)"/>
    <path d="M 1040 255 C 1220 255, 1290 210, 1330 210" fill="none" stroke="{c_slate}" stroke-width="1.5" marker-end="url(#arrow)"/>"""

        # Stacked image drawing matching exact original script dimensions/aspect ratios
        if plane_name == "Axial":
            img_rects = "".join([f'''
      <rect x="{60+i*4}" y="{150+i*4}" width="130" height="156" rx="8" ry="8" fill="#ffffff" stroke="#475569" stroke-width="{"2" if i==4 else "1"}"/>
      <image href="{b64_img}" x="{62+i*4}" y="{152+i*4}" width="126" height="152" clip-path="url(#clip-axial-{i})"/>''' for i in range(5)])
            start_x = 212
        elif plane_name == "Coronal":
            img_rects = "".join([f'''
      <rect x="{60+i*4}" y="{163+i*4}" width="130" height="130" rx="8" ry="8" fill="#ffffff" stroke="#475569" stroke-width="{"2" if i==4 else "1"}"/>
      <image href="{b64_img}" x="{62+i*4}" y="{165+i*4}" width="126" height="126" clip-path="url(#clip-coronal-{i})"/>''' for i in range(5)])
            start_x = 212
        else: # Sagittal: stack rightmost edge is at x=216, so start arrow at x=224 to avoid overlapping frame!
            img_rects = "".join([f'''
      <rect x="{44+i*4}" y="{147+i*4}" width="156" height="142" rx="8" ry="8" fill="#ffffff" stroke="#475569" stroke-width="{"2" if i==4 else "1"}"/>
      <image href="{b64_img}" x="{46+i*4}" y="{150+i*4}" width="152" height="126" clip-path="url(#clip-sagittal-{i})"/>''' for i in range(5)])
            start_x = 224

        return f"""
  <!-- ==================== PATHWAY: {plane_name.upper()} ==================== -->
  <g transform="translate(0, {20 if plane_name=='Axial' else (350 if plane_name=='Coronal' else 680)})">
    <!-- 1. Input Stack -->
    <g filter="url(#shadow)">
      {img_rects}
    </g>
    <text x="135" y="345" font-size="14" font-weight="bold" fill="{c_slate}" text-anchor="middle">{plane_name} [{dims}]</text>

    <!-- Smooth Split Arrows from Input Stack to Dual Pathways -->
    <path d="M {start_x} 255 C {start_x+25} 255, {start_x+35} 165, 270 165" fill="none" stroke="{c_slate}" stroke-width="1.5" marker-end="url(#arrow)"/>
    <path d="M {start_x} 255 C {start_x+25} 255, {start_x+35} 345, 270 345" fill="none" stroke="{c_slate}" stroke-width="1.5" marker-end="url(#arrow)"/>

    <!-- 2. Dual-Pathway Extraction -->
    <!-- Global Path Container -->
    <g filter="url(#shadow)">
      <rect x="270" y="125" width="220" height="80" rx="10" ry="10" fill="url(#grad-dual-path)" stroke="#0284c7" stroke-width="1.8"/>
      <text x="380" y="155" font-size="18" font-weight="bold" fill="{c_sky_hdr}" text-anchor="middle">Global Path</text>
      <text x="380" y="183" font-size="15" font-weight="bold" fill="{c_blue}" text-anchor="middle">{backbone}</text>
    </g>

    <!-- Local Path Container -->
    <g filter="url(#shadow)">
      <rect x="270" y="305" width="220" height="80" rx="10" ry="10" fill="url(#grad-dual-path)" stroke="#0284c7" stroke-width="1.8"/>
      <text x="380" y="333" font-size="18" font-weight="bold" fill="{c_sky_hdr}" text-anchor="middle">Local Path</text>
      <text x="380" y="357" font-size="14" font-weight="bold" fill="{c_blue}" text-anchor="middle">{backbone}</text>
      <text x="380" y="375" font-size="12" font-weight="bold" fill="{c_sky_hdr}" text-anchor="middle">Patch: 64 &#215; 64</text>
    </g>

    <!-- K, V connection to Transformer -->
    <path d="M 490 165 L 525 165 L 525 235 L 560 235" fill="none" stroke="{c_slate}" stroke-width="1.5" marker-end="url(#arrow)"/>
    <text x="530" y="198" font-size="14" font-weight="bold" fill="{c_sky_hdr}" text-anchor="start">K,V</text>

    <!-- Dashed Global Feature bypass line to Global Head -->
    <path d="M 490 165 L 860 165" fill="none" stroke="{c_slate}" stroke-width="1.2" stroke-dasharray="4,4"/>

    <!-- Q connection to Transformer -->
    <path d="M 490 345 L 525 345 L 525 275 L 560 275" fill="none" stroke="{c_slate}" stroke-width="1.5" marker-end="url(#arrow)"/>
    <text x="530" y="322" font-size="14" font-weight="bold" fill="{c_sky_hdr}" text-anchor="start">Q</text>

    <!-- 3. Cross-Attention GL Transformer (Green palette) -->
    <g filter="url(#shadow)">
      <rect x="560" y="195" width="230" height="120" rx="10" ry="10" fill="url(#grad-transformer)" stroke="#15803d" stroke-width="2"/>
      <text x="675" y="238" font-size="21" font-weight="bold" fill="{c_green}" text-anchor="middle">GL Transformer</text>
      <text x="675" y="263" font-size="16" font-weight="bold" fill="{c_green}" text-anchor="middle">{nblocks} &#215; Blocks</text>
      <text x="675" y="288" font-size="14" font-style="italic" fill="{c_green}" text-anchor="middle">Grid: {grid}</text>
    </g>

    <!-- Transformer output arrow -->
    <path d="M 790 255 L 860 255" fill="none" stroke="{c_slate}" stroke-width="1.5" marker-end="url(#arrow)"/>

    <!-- 4. Multi-Head Prediction (Gold/Yellow palette) -->
    <g filter="url(#shadow)">
      <!-- Global Head -->
      <rect x="860" y="125" width="180" height="80" rx="8" ry="8" fill="url(#grad-head)" stroke="#d97706" stroke-width="1.5"/>
      <text x="950" y="157" font-size="17" font-weight="bold" fill="{c_gold}" text-anchor="middle">Global Head</text>
      <text x="950" y="182" font-size="13" font-weight="bold" fill="#b45309" text-anchor="middle">{loss_type}</text>

      <!-- Patch Heads -->
      <rect x="860" y="215" width="180" height="80" rx="8" ry="8" fill="url(#grad-head)" stroke="#d97706" stroke-width="1.5"/>
      <text x="950" y="247" font-size="17" font-weight="bold" fill="{c_gold}" text-anchor="middle">Patch Heads</text>
      <text x="950" y="272" font-size="13" font-weight="bold" fill="#b45309" text-anchor="middle">{loss_type}</text>
    </g>

    {act_block}

    <!-- 6. Partial Prediction Node (mu & yhat) -->
    <circle cx="1350" cy="210" r="20" fill="url(#grad-partial)" stroke="#b45309" stroke-width="1.8" filter="url(#shadow)"/>
    <text x="1350" y="217" font-size="24" font-weight="bold" fill="{c_brown}" text-anchor="middle">&#956;</text>

    <path d="M 1370 210 L 1390 210" fill="none" stroke="{c_slate}" stroke-width="1.5" marker-end="url(#arrow)"/>

    <g filter="url(#shadow)">
      <rect x="1390" y="185" width="80" height="50" rx="6" ry="6" fill="url(#grad-partial)" stroke="#b45309" stroke-width="1.8"/>
      <text x="1430" y="217" font-size="28" font-weight="bold" fill="{c_brown}" text-anchor="middle">&#375;<tspan font-size="18" dy="4">{ysub}</tspan></text>
    </g>
  </g>
"""

    svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2050 1080" width="100%" height="100%" style="background-color: #ffffff; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;">
  <defs>
    <!-- Drop Shadow Filter -->
    <filter id="shadow" x="-5%" y="-5%" width="110%" height="110%">
      <feDropShadow dx="2" dy="3" stdDeviation="4" flood-color="#000000" flood-opacity="0.08"/>
    </filter>

    <!-- Unified Dual Pathway Gradient -->
    <linearGradient id="grad-dual-path" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#f0f9ff"/>
      <stop offset="100%" stop-color="#e0f2fe"/>
    </linearGradient>

    <!-- Green Gradient for GL Transformer -->
    <linearGradient id="grad-transformer" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#f0fdf4"/>
      <stop offset="100%" stop-color="#dcfce7"/>
    </linearGradient>

    <!-- Gold/Yellow Gradient for Multi-Head Prediction -->
    <linearGradient id="grad-head" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#fffbeb"/>
      <stop offset="100%" stop-color="#fef3c7"/>
    </linearGradient>

    <linearGradient id="grad-partial" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#fff7ed"/>
      <stop offset="100%" stop-color="#ffedd5"/>
    </linearGradient>

    <linearGradient id="grad-stacker" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#fef2f2"/>
      <stop offset="100%" stop-color="#fee2e2"/>
    </linearGradient>

    <!-- Marker for Arrows -->
    <marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#475569"/>
    </marker>

    <!-- Clip Paths for Stacked Images -->
    {"".join([f'<clipPath id="clip-axial-{i}"><rect x="{62+i*4}" y="{152+i*4}" width="126" height="152" rx="6" ry="6"/></clipPath>' for i in range(5)])}
    {"".join([f'<clipPath id="clip-coronal-{i}"><rect x="{62+i*4}" y="{165+i*4}" width="126" height="126" rx="6" ry="6"/></clipPath>' for i in range(5)])}
    {"".join([f'<clipPath id="clip-sagittal-{i}"><rect x="{44+i*4}" y="{147+i*4}" width="152" height="132" rx="6" ry="6"/></clipPath>' for i in range(5)])}
  </defs>

  <!-- Background -->
  <rect width="2050" height="1080" fill="#ffffff" />

  <!-- Main Title -->
  <text x="1025" y="45" font-size="30" font-weight="bold" fill="#0f172a" text-anchor="middle">Triplanar Global-Local Transformer Prediction Workflow</text>

  <!-- HIGH-VISIBILITY COLUMN HEADERS -->
  <g font-size="14" font-weight="bold" text-anchor="middle" letter-spacing="0.5">
    <text x="135" y="105" fill="#475569">INPUT STACK</text>

    <text x="380" y="96" fill="#0369a1">DUAL-PATHWAY</text>
    <text x="380" y="114" fill="#0369a1">EXTRACTION</text>

    <text x="675" y="96" fill="#1e5f1e">CROSS-ATTENTION</text>
    <text x="675" y="114" fill="#1e5f1e">TRANSFORMER</text>

    <text x="950" y="96" fill="#7f6000">MULTI-HEAD</text>
    <text x="950" y="114" fill="#7f6000">PREDICTION</text>

    <text x="1185" y="96" fill="#475569">ACTIVATION</text>
    <text x="1185" y="114" fill="#475569">PROCESSING</text>

    <text x="1430" y="96" fill="#5f3a1e">PARTIAL</text>
    <text x="1430" y="114" fill="#5f3a1e">PREDICTION</text>
  </g>

  <!-- PATHWAYS -->
  {render_pathway("Axial", "5, 182, 218", "ResNet-18", "6", "23 &#215; 28", "KL Divergence", "soft_argmax", "ax", axial_b64)}
  {render_pathway("Coronal", "5, 182, 182", "ResNet-34", "8", "23 &#215; 23", "Smooth L1", "direct", "cor", coronal_b64)}
  {render_pathway("Sagittal", "5, 218, 182", "ResNet-18", "6", "28 &#215; 23", "MSE", "direct", "sag", sagittal_b64)}

  <!-- ABSOLUTE CONNECTORS -->
  <g>
    <path d="M 1470 230 C 1535 230, 1535 560, 1560 560" fill="none" stroke="#475569" stroke-width="1.8" marker-end="url(#arrow)"/>
    <path d="M 1470 560 L 1560 560" fill="none" stroke="#475569" stroke-width="1.8" marker-end="url(#arrow)"/>
    <path d="M 1470 890 C 1535 890, 1535 560, 1560 560" fill="none" stroke="#475569" stroke-width="1.8" marker-end="url(#arrow)"/>
  </g>

  <!-- LATE FUSION RIDGE STACKER & FINAL PREDICTION -->
  <g transform="translate(1560, 522.5)">
    <!-- Stacker Title -->
    <text x="110" y="-18" font-size="15" font-weight="bold" fill="#991b1b" text-anchor="middle">LATE FUSION STACKER</text>
    
    <!-- Stacker Box -->
    <g filter="url(#shadow)">
      <rect x="0" y="0" width="220" height="75" rx="12" ry="12" fill="url(#grad-stacker)" stroke="#991b1b" stroke-width="2.2"/>
      <text x="110" y="31" font-size="19" font-weight="bold" fill="#0f172a" text-anchor="middle">&#375;<tspan font-size="13" dy="5">ens</tspan><tspan font-size="19" dy="-5"> = &#946;&#8218; + &#946;&#8321;&#375;</tspan><tspan font-size="13" dy="5">ax</tspan></text>
      <text x="110" y="56" font-size="19" font-weight="bold" fill="#0f172a" text-anchor="middle">+ &#946;&#8322;&#375;<tspan font-size="13" dy="5">cor</tspan><tspan font-size="19" dy="-5"> + &#946;&#8323;&#375;</tspan><tspan font-size="13" dy="5">sag</tspan></text>
    </g>

    <!-- Arrow from Stacker to Final Prediction Circle -->
    <path d="M 220 37.5 L 290 37.5" fill="none" stroke="#475569" stroke-width="1.8" marker-end="url(#arrow)"/>

    <!-- Final Prediction Title (matching INPUT STACK color #475569) -->
    <text x="335" y="-18" font-size="15" font-weight="bold" fill="#475569" text-anchor="middle">FINAL PREDICTION</text>

    <!-- Final Prediction Circle (Grey background with adjustable opacity parameter) -->
    <g filter="url(#shadow)">
      <circle cx="335" cy="37.5" r="42" fill="{FINAL_PRED_BG_COLOR}" fill-opacity="{FINAL_PRED_BG_OPACITY}" stroke="#475569" stroke-width="2.5"/>
      <text x="335" y="46" font-size="27" font-weight="bold" fill="#475569" text-anchor="middle">&#375;<tspan font-size="19" dy="6">ens</tspan></text>
      <text x="335" y="100" font-size="18" font-weight="bold" fill="#475569" text-anchor="middle">Brain Age</text>
    </g>
  </g>
</svg>"""

    svg_path = os.path.join(fig_dir, "fig01_architecture_svg_v2.svg")
    png_path = os.path.join(fig_dir, "fig01_architecture_svg_v2.png")

    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg_content)

    os.system(f"rsvg-convert -b '#ffffff' -f png -o {png_path} {svg_path}")
    print("Done.")

if __name__ == "__main__":
    main()
