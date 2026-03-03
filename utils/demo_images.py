import base64

# Synthetic chest X-ray SVGs per pathology type
# In production: use actual NIH ChestX-ray14 or Kaggle pneumonia dataset images

def xray_base(findings=""):
    """Base chest X-ray SVG template."""
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="320" height="320" viewBox="0 0 320 320">
  <defs>
    <radialGradient id="bg" cx="50%" cy="45%" r="60%">
      <stop offset="0%" stop-color="#3a3a4a"/>
      <stop offset="100%" stop-color="#0d0d14"/>
    </radialGradient>
    <filter id="blur2"><feGaussianBlur stdDeviation="2"/></filter>
    <filter id="blur4"><feGaussianBlur stdDeviation="4"/></filter>
    <filter id="noise">
      <feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="4" result="noise"/>
      <feColorMatrix type="saturate" values="0" in="noise" result="gray"/>
      <feBlend in="SourceGraphic" in2="gray" mode="overlay" result="blend"/>
    </filter>
  </defs>

  <!-- Background -->
  <rect width="320" height="320" fill="url(#bg)"/>

  <!-- Lung fields - left -->
  <ellipse cx="100" cy="155" rx="62" ry="90" fill="#2a2a3a" opacity="0.9"/>
  <!-- Lung fields - right -->
  <ellipse cx="218" cy="155" rx="62" ry="90" fill="#2a2a3a" opacity="0.9"/>

  <!-- Bones / skeleton -->
  <g stroke="#7a8898" fill="none" opacity="0.55">
    <!-- Spine -->
    <line x1="160" y1="25" x2="160" y2="295" stroke-width="6" stroke="#8a9aaa"/>
    <!-- Clavicles -->
    <path d="M 160 50 Q 108 38 78 54" stroke-width="3"/>
    <path d="M 160 50 Q 212 38 242 54" stroke-width="3"/>
    <!-- Ribs L -->
    <path d="M 155 68 Q 78 75 55 105" stroke-width="2"/>
    <path d="M 155 90 Q 74 97 52 130" stroke-width="1.8"/>
    <path d="M 155 112 Q 70 120 50 156" stroke-width="1.8"/>
    <path d="M 155 135 Q 68 143 50 182" stroke-width="1.6"/>
    <path d="M 155 158 Q 66 167 50 208" stroke-width="1.6"/>
    <path d="M 155 180 Q 65 192 52 230" stroke-width="1.5"/>
    <!-- Ribs R -->
    <path d="M 165 68 Q 242 75 265 105" stroke-width="2"/>
    <path d="M 165 90 Q 246 97 268 130" stroke-width="1.8"/>
    <path d="M 165 112 Q 250 120 270 156" stroke-width="1.8"/>
    <path d="M 165 135 Q 252 143 270 182" stroke-width="1.6"/>
    <path d="M 165 158 Q 254 167 270 208" stroke-width="1.6"/>
    <path d="M 165 180 Q 255 192 268 230" stroke-width="1.5"/>
    <!-- Diaphragm -->
    <path d="M 48 238 Q 160 215 272 238" stroke-width="2.5" stroke="#9aabb8"/>
  </g>

  <!-- Heart -->
  <ellipse cx="152" cy="178" rx="42" ry="48" fill="#1e2030" stroke="#5a6a7a" stroke-width="1" opacity="0.85"/>

  <!-- Trachea -->
  <rect x="154" y="25" width="12" height="50" rx="6" fill="#1a2030" stroke="#6a7a88" stroke-width="1"/>

  <!-- Pathology findings -->
  {findings}

  <!-- Scan metadata -->
  <rect x="0" y="295" width="320" height="25" fill="rgba(0,0,0,0.6)"/>
  <text x="8" y="311" font-family="monospace" font-size="9" fill="#6a8090">PA CHEST · 2024</text>
  <text x="200" y="311" font-family="monospace" font-size="9" fill="#4a6070">kVp:120 mAs:4</text>
</svg>'''

DEMO_IMAGES = {
    "normal": xray_base(""),
    "pneumonia": xray_base('''
        <!-- Right lower lobe consolidation -->
        <ellipse cx="220" cy="205" rx="45" ry="38" fill="rgba(220,200,160,0.35)" filter="url(#blur4)"/>
        <ellipse cx="210" cy="215" rx="32" ry="28" fill="rgba(200,180,130,0.40)" filter="url(#blur2)"/>
        <!-- Air bronchograms -->
        <line x1="200" y1="188" x2="215" y2="230" stroke="rgba(160,140,100,0.5)" stroke-width="1.5"/>
        <line x1="215" y1="192" x2="225" y2="228" stroke="rgba(160,140,100,0.4)" stroke-width="1"/>
        <line x1="228" y1="195" x2="232" y2="225" stroke="rgba(160,140,100,0.35)" stroke-width="1"/>
    '''),
    "covid": xray_base('''
        <!-- Bilateral ground-glass opacities - peripheral -->
        <ellipse cx="78" cy="172" rx="38" ry="55" fill="rgba(180,200,190,0.22)" filter="url(#blur4)"/>
        <ellipse cx="240" cy="168" rx="38" ry="52" fill="rgba(180,200,190,0.22)" filter="url(#blur4)"/>
        <!-- Subpleural consolidation -->
        <ellipse cx="68" cy="210" rx="22" ry="18" fill="rgba(200,210,190,0.30)" filter="url(#blur2)"/>
        <ellipse cx="252" cy="208" rx="22" ry="18" fill="rgba(200,210,190,0.30)" filter="url(#blur2)"/>
        <!-- Lower zone haziness -->
        <ellipse cx="100" cy="230" rx="48" ry="25" fill="rgba(170,185,175,0.18)" filter="url(#blur4)"/>
        <ellipse cx="218" cy="228" rx="48" ry="25" fill="rgba(170,185,175,0.18)" filter="url(#blur4)"/>
    '''),
    "effusion": xray_base('''
        <!-- Right pleural effusion - blunted costophrenic angle -->
        <ellipse cx="235" cy="248" rx="55" ry="30" fill="rgba(160,175,195,0.55)" filter="url(#blur2)"/>
        <ellipse cx="248" cy="235" rx="35" ry="42" fill="rgba(150,168,190,0.45)" filter="url(#blur4)"/>
        <!-- Meniscus sign -->
        <path d="M 180 230 Q 240 210 272 240" stroke="rgba(180,195,210,0.6)" stroke-width="2" fill="none"/>
        <!-- Homogeneous opacity -->
        <rect x="185" y="225" width="85" height="60" fill="rgba(145,162,185,0.35)" rx="5" filter="url(#blur4)"/>
    '''),
    "cardiomegaly": xray_base('''
        <!-- Enlarged heart shadow CTR > 0.5 -->
        <ellipse cx="152" cy="178" rx="72" ry="78" fill="rgba(90,80,100,0.65)" filter="url(#blur2)"/>
        <!-- Pulmonary vascular congestion -->
        <line x1="160" y1="100" x2="105" y2="155" stroke="rgba(160,155,175,0.45)" stroke-width="2.5"/>
        <line x1="160" y1="100" x2="215" y2="152" stroke="rgba(160,155,175,0.45)" stroke-width="2.5"/>
        <line x1="160" y1="110" x2="92" y2="165" stroke="rgba(150,145,165,0.35)" stroke-width="1.8"/>
        <line x1="160" y1="110" x2="228" y2="162" stroke="rgba(150,145,165,0.35)" stroke-width="1.8"/>
    '''),
}

def get_demo_image(demo_type):
    svg = DEMO_IMAGES.get(demo_type, DEMO_IMAGES["normal"])
    b64 = base64.b64encode(svg.encode()).decode()
    return f"data:image/svg+xml;base64,{b64}"
