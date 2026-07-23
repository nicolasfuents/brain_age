import os
import base64

def get_base64_image(path):
    with open(path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:image/png;base64,{encoded}"

def make_basic_block(bx, cx, name, conv_text_1="Conv2D (3×3)", has_1x1=False):
    svg_elements = [
        f'<!-- {name} -->',
        # Dashed container box (starting at y=160)
        f'<rect x="{bx}" y="160" width="110" height="230" rx="8" ry="8" fill="#ffffff" stroke="#475569" stroke-dasharray="3,3" stroke-width="1.2"/>',
        # Block title placed OUTSIDE (above) the dashed container box
        f'<text x="{cx}" y="150" font-size="11" font-weight="bold" fill="#475569" text-anchor="middle">{name}</text>',
        
        # Segmented flow lines (shifted down by 20px)
        f'<line x1="{cx}" y1="160" x2="{cx}" y2="182" stroke="#475569" stroke-width="1.2" />',
        f'<line x1="{cx}" y1="202" x2="{cx}" y2="210" stroke="#475569" stroke-width="1.2" />',
        f'<line x1="{cx}" y1="230" x2="{cx}" y2="238" stroke="#475569" stroke-width="1.2" />',
        f'<line x1="{cx}" y1="258" x2="{cx}" y2="266" stroke="#475569" stroke-width="1.2" />',
        f'<line x1="{cx}" y1="286" x2="{cx}" y2="294" stroke="#475569" stroke-width="1.2" />',
        f'<line x1="{cx}" y1="314" x2="{cx}" y2="320" stroke="#475569" stroke-width="1.2" />',
        f'<line x1="{cx}" y1="340" x2="{cx}" y2="348" stroke="#475569" stroke-width="1.2" />',
        f'<line x1="{cx}" y1="378" x2="{cx}" y2="390" stroke="#475569" stroke-width="1.2" />',
        
        # Box 1: Conv2D 1
        f'<rect x="{bx+10}" y="182" width="90" height="20" rx="4" ry="4" fill="#e0f2fe" stroke="#0284c7" stroke-width="1"/>',
        f'<text x="{cx}" y="195" font-size="9" font-weight="bold" fill="#0369a1" text-anchor="middle">{conv_text_1}</text>',
        
        # Box 2: BatchNorm2D 1
        f'<rect x="{bx+10}" y="210" width="90" height="20" rx="4" ry="4" fill="#fef3c7" stroke="#d97706" stroke-width="1"/>',
        f'<text x="{cx}" y="223" font-size="9" font-weight="bold" fill="#b45309" text-anchor="middle">BatchNorm2D</text>',
        
        # Box 3: ReLU 1
        f'<rect x="{bx+10}" y="238" width="90" height="20" rx="4" ry="4" fill="#dcfce7" stroke="#16a34a" stroke-width="1"/>',
        f'<text x="{cx}" y="251" font-size="9" font-weight="bold" fill="#15803d" text-anchor="middle">ReLU</text>',
        
        # Box 4: Conv2D 2
        f'<rect x="{bx+10}" y="266" width="90" height="20" rx="4" ry="4" fill="#e0f2fe" stroke="#0284c7" stroke-width="1"/>',
        f'<text x="{cx}" y="279" font-size="9" font-weight="bold" fill="#0369a1" text-anchor="middle">Conv2D (3×3)</text>',
        
        # Box 5: BatchNorm2D 2
        f'<rect x="{bx+10}" y="294" width="90" height="20" rx="4" ry="4" fill="#fef3c7" stroke="#d97706" stroke-width="1"/>',
        f'<text x="{cx}" y="307" font-size="9" font-weight="bold" fill="#b45309" text-anchor="middle">BatchNorm2D</text>',
        
        # Circle + (cy=330)
        f'<circle cx="{cx}" cy="330" r="10" fill="#ffffff" stroke="#475569" stroke-width="1.5"/>',
        f'<text x="{cx}" y="334" font-size="12" font-weight="bold" fill="#475569" text-anchor="middle">+</text>',
        
        # Box 6: ReLU 2
        f'<rect x="{bx+10}" y="348" width="90" height="30" rx="4" ry="4" fill="#dcfce7" stroke="#16a34a" stroke-width="1"/>',
        f'<text x="{cx}" y="367" font-size="10" font-weight="bold" fill="#15803d" text-anchor="middle">ReLU</text>',
        
        # Arrowheads in gaps (flow between layers)
        f'<polygon points="{cx-3},204 {cx+3},204 {cx},209" fill="#475569" />',
        f'<polygon points="{cx-3},232 {cx+3},232 {cx},237" fill="#475569" />',
        f'<polygon points="{cx-3},260 {cx+3},260 {cx},265" fill="#475569" />',
        f'<polygon points="{cx-3},288 {cx+3},288 {cx},293" fill="#475569" />',
        f'<polygon points="{cx-3},315 {cx+3},315 {cx},319" fill="#475569" />',
        f'<polygon points="{cx-3},343 {cx+3},343 {cx},347" fill="#475569" />',
        f'<polygon points="{cx-3},382 {cx+3},382 {cx},387" fill="#475569" />',
        
        # Shortcut path (Solid, outer-positioned, shifted down by 20px)
        f'<path d="M {cx} 172 L {bx+120} 172 L {bx+120} 330 L {cx+11} 330" fill="none" stroke="#475569" stroke-width="1.2" marker-end="url(#arrow)"/>'
    ]
    
    # 1x1 Conv box (drawn on top of the shortcut line, shifted down by 20px)
    if has_1x1:
        svg_elements.append(f'<rect x="{bx+95}" y="239" width="50" height="18" rx="3" ry="3" fill="#ffffff" stroke="#475569" stroke-width="1.2"/>')
        svg_elements.append(f'<text x="{bx+120}" y="252" font-size="9" font-weight="bold" fill="#475569" text-anchor="middle">1×1 Conv</text>')
        
    return "\n    ".join(svg_elements)

def make_stem_block():
    svg_elements = [
        '<!-- Stem Block -->',
        '<rect x="215" y="163" width="160" height="170" rx="10" ry="10" fill="#ffffff" stroke="#475569" stroke-width="1.5"/>',
        '<text x="295" y="190" font-size="14" font-weight="bold" fill="#475569" text-anchor="middle">STEM BLOCK</text>',
        
        # Segmented flow lines
        '<line x1="295" y1="201" x2="295" y2="209" stroke="#475569" stroke-width="1.2" />',
        '<line x1="295" y1="235" x2="295" y2="251" stroke="#475569" stroke-width="1.2" />',
        '<line x1="295" y1="277" x2="295" y2="293" stroke="#475569" stroke-width="1.2" />',
        '<line x1="295" y1="319" x2="295" y2="333" stroke="#475569" stroke-width="1.2" />',
        
        # Layer boxes (height=26)
        '<rect x="230" y="209" width="130" height="26" rx="6" ry="6" fill="#e0f2fe" stroke="#0284c7" stroke-width="1"/>',
        '<text x="295" y="225" font-size="13" font-weight="bold" fill="#0369a1" text-anchor="middle">Conv2D (3×3)</text>',
        
        '<rect x="230" y="251" width="130" height="26" rx="6" ry="6" fill="#fef3c7" stroke="#d97706" stroke-width="1"/>',
        '<text x="295" y="267" font-size="13" font-weight="bold" fill="#b45309" text-anchor="middle">BatchNorm2D</text>',
        
        '<rect x="230" y="293" width="130" height="26" rx="6" ry="6" fill="#dcfce7" stroke="#16a34a" stroke-width="1"/>',
        '<text x="295" y="309" font-size="13" font-weight="bold" fill="#15803d" text-anchor="middle">ReLU</text>',
        
        # Gaps arrowheads
        '<polygon points="292,242 298,242 295,246" fill="#475569" />',
        '<polygon points="292,284 298,284 295,288" fill="#475569" />',
        '<polygon points="292,326 298,326 295,330" fill="#475569" />'
    ]
    return "\n    ".join(svg_elements)

def main():
    fig_dir = "/home/nfuentes/scratch/brain_age_project/openBHB_dataset/scripts/manuscript/paper/figures"
    extracted_dir = os.path.join(fig_dir, "extracted")
    
    axial_b64 = get_base64_image(os.path.join(extracted_dir, "axial.png"))
    coronal_b64 = get_base64_image(os.path.join(extracted_dir, "coronal.png"))
    
    outer_stroke = '#475569'
    outer_text = 'fill="#475569"'
    
    # ==============================================================================
    # DIAGRAM 1: ResNet-18 (Axial)
    # ==============================================================================
    # Note: Stage containers have been made taller (height=316 instead of 296) to accommodate external BasicBlock titles.
    # The X-coordinates of the second BasicBlock in each stage are easily configured in the variables below.
    r18_bb2_x1 = 580
    r18_bb2_x2 = 925
    r18_bb2_x3 = 1270
    r18_bb2_x4 = 1615

    svg_r18 = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2000 520" width="100%" height="100%" style="background-color: #ffffff; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;">
  <defs>
    <filter id="shadow" x="-5%" y="-5%" width="110%" height="110%">
      <feDropShadow dx="3" dy="3" stdDeviation="5" flood-color="#000000" flood-opacity="0.1"/>
    </filter>
    
    <marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#475569"/>
    </marker>
    
    <clipPath id="clip-axial"><rect x="52" y="162" width="126" height="152" rx="6" ry="6"/></clipPath>
  </defs>

  <rect width="2000" height="520" fill="#ffffff" />
  <text x="1000" y="40" font-size="28" font-weight="bold" fill="#111827" text-anchor="middle">ResNet-18 Feature Extractor Backbone</text>

  <!-- 1. Input Image Stack -->
  <g filter="url(#shadow)">
    <rect x="50" y="160" width="130" height="156" rx="8" ry="8" fill="#ffffff" stroke="{outer_stroke}" stroke-width="1.5"/>
    <image href="{axial_b64}" x="52" y="162" width="126" height="152" clip-path="url(#clip-axial)"/>
  </g>
  <text x="115" y="340" font-size="16" font-weight="bold" fill="#1f2937" text-anchor="middle">Input Stack</text>
  <text x="115" y="360" font-size="13" font-weight="bold" fill="#1f2937" text-anchor="middle">5 × 182 × 218</text>

  <!-- Arrow to Stem -->
  <path d="M 180 238 L 215 238" fill="none" stroke="{outer_stroke}" stroke-width="1.5" marker-end="url(#arrow)"/>

  <!-- 2. Stem Block -->
  <g filter="url(#shadow)">
    {make_stem_block()}
  </g>
  <text x="295" y="357" font-size="13" fill="#4b5563" text-anchor="middle">64 Channels</text>
  <text x="295" y="377" font-size="13" font-weight="bold" fill="#1f2937" text-anchor="middle">64 × 182 × 218</text>

  <!-- Arrow to Stage 1 -->
  <path d="M 375 238 L 410 238" fill="none" stroke="{outer_stroke}" stroke-width="1.5" marker-end="url(#arrow)"/>

  <!-- 3. Stage 1 (64 Channels) -->
  <g filter="url(#shadow)">
    <rect x="410" y="90" width="310" height="316" rx="12" ry="12" fill="#ffffff" stroke="{outer_stroke}" stroke-width="1.5"/>
    <text x="565" y="113" font-size="15" font-weight="bold" {outer_text} text-anchor="middle">STAGE 1</text>
    <text x="565" y="130" font-size="12" font-weight="bold" {outer_text} text-anchor="middle">2× BasicBlocks</text>
    
    {make_basic_block(430, 485, "BasicBlock 1")}

    {make_basic_block(r18_bb2_x1, r18_bb2_x1 + 55, "BasicBlock 2")}
  </g>
  <text x="565" y="430" font-size="13" fill="#4b5563" text-anchor="middle">64 Channels</text>
  <text x="565" y="450" font-size="13" font-weight="bold" fill="#1f2937" text-anchor="middle">64 × 182 × 218</text>

  <!-- Arrow to Stage 2 -->
  <path d="M 720 238 L 755 238" fill="none" stroke="{outer_stroke}" stroke-width="1.5" marker-end="url(#arrow)"/>

  <!-- 4. Stage 2 (128 Channels) -->
  <g filter="url(#shadow)">
    <rect x="755" y="90" width="310" height="316" rx="12" ry="12" fill="#ffffff" stroke="{outer_stroke}" stroke-width="1.5"/>
    <text x="910" y="113" font-size="15" font-weight="bold" {outer_text} text-anchor="middle">STAGE 2</text>
    <text x="910" y="130" font-size="12" font-weight="bold" {outer_text} text-anchor="middle">2× BasicBlocks</text>
    
    {make_basic_block(775, 830, "BasicBlock 1", "Conv2D (3×3, s=2)", has_1x1=True)}

    {make_basic_block(r18_bb2_x2, r18_bb2_x2 + 55, "BasicBlock 2")}
  </g>
  <text x="910" y="430" font-size="13" fill="#4b5563" text-anchor="middle">128 Channels</text>
  <text x="910" y="450" font-size="13" font-weight="bold" fill="#1f2937" text-anchor="middle">128 × 91 × 109</text>

  <!-- Arrow to Stage 3 -->
  <path d="M 1065 238 L 1100 238" fill="none" stroke="{outer_stroke}" stroke-width="1.5" marker-end="url(#arrow)"/>

  <!-- 5. Stage 3 (256 Channels) -->
  <g filter="url(#shadow)">
    <rect x="1100" y="90" width="310" height="316" rx="12" ry="12" fill="#ffffff" stroke="{outer_stroke}" stroke-width="1.5"/>
    <text x="1255" y="113" font-size="15" font-weight="bold" {outer_text} text-anchor="middle">STAGE 3</text>
    <text x="1255" y="130" font-size="12" font-weight="bold" {outer_text} text-anchor="middle">2× BasicBlocks</text>
    
    {make_basic_block(1120, 1175, "BasicBlock 1", "Conv2D (3×3, s=2)", has_1x1=True)}

    {make_basic_block(r18_bb2_x3, r18_bb2_x3 + 55, "BasicBlock 2")}
  </g>
  <text x="1255" y="430" font-size="13" fill="#4b5563" text-anchor="middle">256 Channels</text>
  <text x="1255" y="450" font-size="13" font-weight="bold" fill="#1f2937" text-anchor="middle">256 × 46 × 55</text>

  <!-- Arrow to Stage 4 -->
  <path d="M 1410 238 L 1445 238" fill="none" stroke="{outer_stroke}" stroke-width="1.5" marker-end="url(#arrow)"/>

  <!-- 6. Stage 4 (512 Channels) -->
  <g filter="url(#shadow)">
    <rect x="1445" y="90" width="310" height="316" rx="12" ry="12" fill="#ffffff" stroke="{outer_stroke}" stroke-width="1.5"/>
    <text x="1600" y="113" font-size="15" font-weight="bold" {outer_text} text-anchor="middle">STAGE 4</text>
    <text x="1600" y="130" font-size="12" font-weight="bold" {outer_text} text-anchor="middle">2× BasicBlocks</text>
    
    {make_basic_block(1465, 1520, "BasicBlock 1", "Conv2D (3×3, s=2)", has_1x1=True)}

    {make_basic_block(r18_bb2_x4, r18_bb2_x4 + 55, "BasicBlock 2")}
  </g>
  <text x="1600" y="430" font-size="13" fill="#4b5563" text-anchor="middle">512 Channels</text>
  <text x="1600" y="450" font-size="13" font-weight="bold" fill="#1f2937" text-anchor="middle">512 × 23 × 28</text>
  <!-- Arrow to Feature Map -->
  <path d="M 1755 238 L 1790 238" fill="none" stroke="{outer_stroke}" stroke-width="1.5" marker-end="url(#arrow)"/>
  
  <!-- 7. Deep Feature Map -->
  <g filter="url(#shadow)">
    <rect x="1790" y="160" width="180" height="130" rx="8" ry="8" fill="#ffffff" stroke="{outer_stroke}" stroke-width="1.5"/>
    <rect x="1802" y="172" width="156" height="106" rx="6" ry="6" fill="#eff6ff" stroke="#2b5c8f" stroke-width="1"/>
    <text x="1880" y="208" font-size="16" font-weight="bold" fill="#1e3a5f" text-anchor="middle">Deep Features</text>
    <text x="1880" y="235" font-size="14" font-weight="bold" fill="#475569" text-anchor="middle">512 × 23 × 28</text>
    <text x="1880" y="260" font-size="11" fill="#4b5563" text-anchor="middle">(for Axial plane)</text>
  </g>
  <text x="1880" y="315" font-size="16" font-weight="bold" fill="#1e3a5f" text-anchor="middle">Feature Tensor</text>
</svg>"""

    # ==============================================================================
    # DIAGRAM 2: ResNet-34 (Coronal)
    # ==============================================================================
    # The X-coordinates of the second BasicBlock in each stage are easily configured in the variables below.
    r34_bb2_x1 = 585
    r34_bb2_x2 = 930
    r34_bb2_x3 = 1275
    r34_bb2_x4 = 1620

    svg_r34 = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2000 520" width="100%" height="100%" style="background-color: #ffffff; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;">
  <defs>
    <filter id="shadow" x="-5%" y="-5%" width="110%" height="110%">
      <feDropShadow dx="3" dy="3" stdDeviation="5" flood-color="#000000" flood-opacity="0.1"/>
    </filter>
    
    <marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#475569"/>
    </marker>
    
    <clipPath id="clip-coronal"><rect x="52" y="162" width="126" height="126" rx="6" ry="6"/></clipPath>
  </defs>

  <rect width="2000" height="520" fill="#ffffff" />
  <text x="1000" y="40" font-size="28" font-weight="bold" fill="#111827" text-anchor="middle">ResNet-34 Feature Extractor Backbone</text>

  <!-- 1. Input Image Stack -->
  <g filter="url(#shadow)">
    <rect x="50" y="160" width="130" height="130" rx="8" ry="8" fill="#ffffff" stroke="{outer_stroke}" stroke-width="1.5"/>
    <image href="{coronal_b64}" x="52" y="162" width="126" height="126" clip-path="clip-coronal)"/>
  </g>
  <text x="115" y="315" font-size="16" font-weight="bold" fill="#1f2937" text-anchor="middle">Input Stack</text>
  <text x="115" y="335" font-size="13" font-weight="bold" fill="#1f2937" text-anchor="middle">5 × 182 × 182</text>

  <!-- Arrow to Stem -->
  <path d="M 180 225 L 215 225" fill="none" stroke="{outer_stroke}" stroke-width="1.5" marker-end="url(#arrow)"/>

  <!-- 2. Stem Block -->
  <g filter="url(#shadow)">
    {make_stem_block()}
  </g>
  <text x="295" y="357" font-size="13" fill="#4b5563" text-anchor="middle">64 Channels</text>
  <text x="295" y="377" font-size="13" font-weight="bold" fill="#1f2937" text-anchor="middle">64 × 182 × 182</text>

  <!-- Arrow to Stage 1 -->
  <path d="M 375 228 L 410 228" fill="none" stroke="{outer_stroke}" stroke-width="1.5" marker-end="url(#arrow)"/>

  <!-- 3. Stage 1 (64 Channels) -->
  <g filter="url(#shadow)">
    <rect x="410" y="90" width="310" height="316" rx="12" ry="12" fill="#ffffff" stroke="{outer_stroke}" stroke-width="1.5"/>
    <text x="565" y="113" font-size="15" font-weight="bold" {outer_text} text-anchor="middle">STAGE 1</text>
    <text x="565" y="130" font-size="12" font-weight="bold" {outer_text} text-anchor="middle">3× BasicBlocks</text>
    
    {make_basic_block(425, 480, "BasicBlock 1")}
    
    # Ellipsis BasicBlock 2
    <text x="{(480 + r34_bb2_x1 + 55) // 2 + 5}" y="270" font-size="24" font-weight="bold" {outer_text} text-anchor="middle">...</text>
    
    {make_basic_block(r34_bb2_x1, r34_bb2_x1 + 55, "BasicBlock 3")}
  </g>
  <text x="565" y="430" font-size="13" fill="#4b5563" text-anchor="middle">64 Channels</text>
  <text x="565" y="450" font-size="13" font-weight="bold" fill="#1f2937" text-anchor="middle">64 × 182 × 182</text>

  <!-- Arrow to Stage 2 -->
  <path d="M 720 238 L 755 238" fill="none" stroke="{outer_stroke}" stroke-width="1.5" marker-end="url(#arrow)"/>

  <!-- 4. Stage 2 (128 Channels) -->
  <g filter="url(#shadow)">
    <rect x="755" y="90" width="310" height="316" rx="12" ry="12" fill="#ffffff" stroke="{outer_stroke}" stroke-width="1.5"/>
    <text x="910" y="113" font-size="15" font-weight="bold" {outer_text} text-anchor="middle">STAGE 2</text>
    <text x="910" y="130" font-size="12" font-weight="bold" {outer_text} text-anchor="middle">4× BasicBlocks</text>
    
    {make_basic_block(770, 825, "BasicBlock 1", "Conv2D (3×3, s=2)", has_1x1=True)}

    # Ellipsis BasicBlock 2/3
    <text x="{(825 + r34_bb2_x2 + 55) // 2 + 5}" y="270" font-size="24" font-weight="bold" {outer_text} text-anchor="middle">...</text>
    
    {make_basic_block(r34_bb2_x2, r34_bb2_x2 + 55, "BasicBlock 4")}
  </g>
  <text x="910" y="430" font-size="13" fill="#4b5563" text-anchor="middle">128 Channels</text>
  <text x="910" y="450" font-size="13" font-weight="bold" fill="#1f2937" text-anchor="middle">128 × 91 × 91</text>

  <!-- Arrow to Stage 3 -->
  <path d="M 1065 238 L 1100 238" fill="none" stroke="{outer_stroke}" stroke-width="1.5" marker-end="url(#arrow)"/>

  <!-- 5. Stage 3 (256 Channels) -->
  <g filter="url(#shadow)">
    <rect x="1100" y="90" width="310" height="316" rx="12" ry="12" fill="#ffffff" stroke="{outer_stroke}" stroke-width="1.5"/>
    <text x="1255" y="113" font-size="15" font-weight="bold" {outer_text} text-anchor="middle">STAGE 3</text>
    <text x="1255" y="130" font-size="12" font-weight="bold" {outer_text} text-anchor="middle">6× BasicBlocks</text>
    
    {make_basic_block(1115, 1170, "BasicBlock 1", "Conv2D (3×3, s=2)", has_1x1=True)}

    # Ellipsis BasicBlock 2/3/4/5
    <text x="{(1170 + r34_bb2_x3 + 55) // 2 + 5}" y="270" font-size="24" font-weight="bold" {outer_text} text-anchor="middle">...</text>
    
    {make_basic_block(r34_bb2_x3, r34_bb2_x3 + 55, "BasicBlock 6")}
  </g>
  <text x="1255" y="430" font-size="13" fill="#4b5563" text-anchor="middle">256 Channels</text>
  <text x="1255" y="450" font-size="13" font-weight="bold" fill="#1f2937" text-anchor="middle">256 × 46 × 46</text>

  <!-- Arrow to Stage 4 -->
  <path d="M 1410 238 L 1445 238" fill="none" stroke="{outer_stroke}" stroke-width="1.5" marker-end="url(#arrow)"/>

  <!-- 6. Stage 4 (512 Channels) -->
  <g filter="url(#shadow)">
    <rect x="1445" y="90" width="310" height="316" rx="12" ry="12" fill="#ffffff" stroke="{outer_stroke}" stroke-width="1.5"/>
    <text x="1600" y="113" font-size="15" font-weight="bold" {outer_text} text-anchor="middle">STAGE 4</text>
    <text x="1600" y="130" font-size="12" font-weight="bold" {outer_text} text-anchor="middle">3× BasicBlocks</text>
    
    {make_basic_block(1460, 1515, "BasicBlock 1", "Conv2D (3×3, s=2)", has_1x1=True)}

    # Ellipsis BasicBlock 2
    <text x="{(1515 + r34_bb2_x4 + 55) // 2 + 5}" y="270" font-size="24" font-weight="bold" {outer_text} text-anchor="middle">...</text>

    {make_basic_block(r34_bb2_x4, r34_bb2_x4 + 55, "BasicBlock 3")}
  </g>
  <text x="1600" y="430" font-size="13" fill="#4b5563" text-anchor="middle">512 Channels</text>
  <text x="1600" y="450" font-size="13" font-weight="bold" fill="#1f2937" text-anchor="middle">512 × 23 × 23</text>

  <!-- Arrow to Feature Map -->
  <path d="M 1755 238 L 1790 238" fill="none" stroke="{outer_stroke}" stroke-width="1.5" marker-end="url(#arrow)"/>

  <!-- 7. Deep Feature Map -->
  <g filter="url(#shadow)">
    <rect x="1790" y="160" width="180" height="130" rx="8" ry="8" fill="#ffffff" stroke="{outer_stroke}" stroke-width="1.5"/>
    <rect x="1802" y="172" width="156" height="106" rx="6" ry="6" fill="#eff6ff" stroke="#2b5c8f" stroke-width="1"/>
    <text x="1880" y="208" font-size="16" font-weight="bold" fill="#1e3a5f" text-anchor="middle">Deep Features</text>
    <text x="1880" y="235" font-size="14" font-weight="bold" fill="#475569" text-anchor="middle">512 × 23 × 23</text>
    <text x="1880" y="260" font-size="11" fill="#4b5563" text-anchor="middle">(for Coronal plane)</text>
  </g>
  <text x="1880" y="315" font-size="16" font-weight="bold" fill="#1e3a5f" text-anchor="middle">Feature Tensor</text>
</svg>"""

    # --- SAVE AND COMPILE R18 ---
    svg_path_r18 = "/home/nfuentes/scratch/brain_age_project/openBHB_dataset/scripts/manuscript/paper/figures/fig02_backbone_r18.svg"
    png_path_r18 = "/home/nfuentes/scratch/brain_age_project/openBHB_dataset/scripts/manuscript/paper/figures/fig02_backbone_r18.png"
    
    with open(svg_path_r18, "w", encoding="utf-8") as f:
        f.write(svg_r18)
    os.system(f"rsvg-convert -b '#ffffff' -f png -o {png_path_r18} {svg_path_r18}")
    
    # --- SAVE AND COMPILE R34 ---
    svg_path_r34 = "/home/nfuentes/scratch/brain_age_project/openBHB_dataset/scripts/manuscript/paper/figures/fig02_backbone_r34.svg"
    png_path_r34 = "/home/nfuentes/scratch/brain_age_project/openBHB_dataset/scripts/manuscript/paper/figures/fig02_backbone_r34.png"
    
    with open(svg_path_r34, "w", encoding="utf-8") as f:
        f.write(svg_r34)
    os.system(f"rsvg-convert -b '#ffffff' -f png -o {png_path_r34} {svg_path_r34}")
    
    print("Done.")

if __name__ == "__main__":
    main()