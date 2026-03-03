import base64
import math
import random
import io

# We generate a synthetic Grad-CAM heatmap as a base64 PNG using pure Python
# In production: use torch + torchvision hooks on actual model

# Activation hotspot definitions per class (normalized 0-1 coordinates)
CLASS_HOTSPOTS = {
    "Normal": [],
    "Pneumonia": [
        {"x": 0.35, "y": 0.65, "r": 0.18, "intensity": 0.9},
        {"x": 0.62, "y": 0.60, "r": 0.14, "intensity": 0.7},
        {"x": 0.40, "y": 0.55, "r": 0.10, "intensity": 0.5},
    ],
    "COVID-19": [
        {"x": 0.25, "y": 0.55, "r": 0.22, "intensity": 0.85},
        {"x": 0.72, "y": 0.52, "r": 0.20, "intensity": 0.80},
        {"x": 0.50, "y": 0.70, "r": 0.12, "intensity": 0.60},
    ],
    "Pleural Effusion": [
        {"x": 0.30, "y": 0.82, "r": 0.20, "intensity": 0.92},
        {"x": 0.68, "y": 0.79, "r": 0.16, "intensity": 0.75},
    ],
    "Cardiomegaly": [
        {"x": 0.50, "y": 0.55, "r": 0.28, "intensity": 0.95},
        {"x": 0.45, "y": 0.45, "r": 0.15, "intensity": 0.60},
    ],
    "Atelectasis": [
        {"x": 0.28, "y": 0.45, "r": 0.15, "intensity": 0.88},
        {"x": 0.38, "y": 0.58, "r": 0.10, "intensity": 0.65},
    ],
    "Consolidation": [
        {"x": 0.65, "y": 0.55, "r": 0.20, "intensity": 0.90},
        {"x": 0.58, "y": 0.45, "r": 0.12, "intensity": 0.65},
    ],
    "Edema": [
        {"x": 0.50, "y": 0.50, "r": 0.30, "intensity": 0.80},
        {"x": 0.35, "y": 0.62, "r": 0.18, "intensity": 0.70},
        {"x": 0.65, "y": 0.60, "r": 0.16, "intensity": 0.65},
    ],
}

def heat_color(val):
    """Map 0-1 activation value to RGBA heat color (blue→green→yellow→red)."""
    val = max(0.0, min(1.0, val))
    if val < 0.25:
        t = val / 0.25
        r, g, b = 0, int(t * 128), 255
    elif val < 0.5:
        t = (val - 0.25) / 0.25
        r, g, b = 0, 128 + int(t * 127), int(255 * (1 - t))
    elif val < 0.75:
        t = (val - 0.5) / 0.25
        r, g, b = int(t * 255), 255, 0
    else:
        t = (val - 0.75) / 0.25
        r, g, b = 255, int(255 * (1 - t)), 0
    alpha = int(val * 200)
    return (r, g, b, alpha)

def generate_gradcam_overlay(demo_type, predicted_class, width=320, height=320):
    """
    Generate a synthetic Grad-CAM heatmap as base64 PNG.
    Returns a data URI string.
    """
    hotspots = CLASS_HOTSPOTS.get(predicted_class, [])

    # Build activation map
    act = [[0.0] * width for _ in range(height)]

    for hs in hotspots:
        cx = int(hs["x"] * width)
        cy = int(hs["y"] * height)
        r  = hs["r"] * min(width, height)
        intensity = hs["intensity"]
        # Gaussian falloff
        for dy in range(-int(r * 1.5), int(r * 1.5) + 1):
            for dx in range(-int(r * 1.5), int(r * 1.5) + 1):
                px, py = cx + dx, cy + dy
                if 0 <= px < width and 0 <= py < height:
                    dist = math.sqrt(dx*dx + dy*dy)
                    val = intensity * math.exp(-(dist**2) / (2 * (r * 0.5)**2))
                    act[py][px] = max(act[py][px], val)

    # Add slight noise
    random.seed(42)
    for y in range(height):
        for x in range(width):
            act[y][x] = min(1.0, act[y][x] + random.uniform(0, 0.04))

    # Render as PNG using raw bytes (no PIL dependency)
    # We'll encode as a simple PPM then convert to base64 for inline use
    # Actually output as a CSS-renderable SVG heatmap for zero dependencies

    # Build SVG with radial gradients as heatmap
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    ]

    # Background: dark X-ray style
    svg_parts.append(f'<rect width="{width}" height="{height}" fill="#1a1a2e" rx="4"/>')

    # Draw stylized rib cage / chest outline
    svg_parts.append('''
        <g opacity="0.3" stroke="#4a6080" fill="none" stroke-width="1">
          <!-- Spine -->
          <line x1="160" y1="30" x2="160" y2="290" stroke="#6a8090" stroke-width="2"/>
          <!-- Ribs left -->
          <path d="M 160 70 Q 80 80 60 110" stroke-width="1.5"/>
          <path d="M 160 95 Q 75 105 55 138"/>
          <path d="M 160 120 Q 72 130 52 165"/>
          <path d="M 160 145 Q 70 155 50 192"/>
          <path d="M 160 170 Q 68 180 50 218"/>
          <!-- Ribs right -->
          <path d="M 160 70 Q 240 80 260 110" stroke-width="1.5"/>
          <path d="M 160 95 Q 245 105 265 138"/>
          <path d="M 160 120 Q 248 130 268 165"/>
          <path d="M 160 145 Q 250 155 270 192"/>
          <path d="M 160 170 Q 252 180 270 218"/>
          <!-- Clavicles -->
          <path d="M 160 55 Q 105 45 75 60" stroke-width="2"/>
          <path d="M 160 55 Q 215 45 245 60" stroke-width="2"/>
          <!-- Diaphragm -->
          <path d="M 55 230 Q 160 210 265 230" stroke-width="1.5"/>
          <!-- Heart outline -->
          <ellipse cx="155" cy="175" rx="45" ry="50" opacity="0.4"/>
          <!-- Lung borders -->
          <path d="M 75 65 Q 55 120 60 220 Q 80 240 130 235 Q 145 200 140 100 Q 130 65 75 65" opacity="0.25"/>
          <path d="M 245 65 Q 265 120 260 220 Q 240 240 190 235 Q 175 200 180 100 Q 190 65 245 65" opacity="0.25"/>
        </g>
    ''')

    # Heatmap overlay using radial gradients per hotspot
    defs = ['<defs>']
    overlays = []
    for i, hs in enumerate(hotspots):
        cx_pct = hs["x"] * 100
        cy_pct = hs["y"] * 100
        r_pct  = hs["r"] * 100
        intensity = hs["intensity"]
        gid = f"hm{i}"

        # Color at peak
        r, g, b, _ = heat_color(intensity)
        rm, gm, bm, _ = heat_color(intensity * 0.6)
        rl, gl, bl, _ = heat_color(intensity * 0.2)

        defs.append(f'''
            <radialGradient id="{gid}" cx="{cx_pct}%" cy="{cy_pct}%" r="{r_pct * 1.5}%" gradientUnits="userSpaceOnUse">
              <stop offset="0%"   stop-color="rgb({r},{g},{b})"   stop-opacity="{intensity * 0.75:.2f}"/>
              <stop offset="40%"  stop-color="rgb({rm},{gm},{bm})" stop-opacity="{intensity * 0.45:.2f}"/>
              <stop offset="80%"  stop-color="rgb({rl},{gl},{bl})" stop-opacity="{intensity * 0.15:.2f}"/>
              <stop offset="100%" stop-color="rgb(0,0,0)"          stop-opacity="0"/>
            </radialGradient>
        ''')
        overlays.append(f'<rect width="{width}" height="{height}" fill="url(#{gid})"/>')

    defs.append('</defs>')
    svg_parts.extend(defs)
    svg_parts.extend(overlays)

    # "Normal" case: light uniform green tint
    if not hotspots:
        svg_parts.append(f'<rect width="{width}" height="{height}" fill="rgba(0,180,80,0.08)"/>')

    # Border + label
    pred_short = predicted_class[:12]
    svg_parts.append(f'<rect width="{width}" height="{height}" fill="none" stroke="rgba(255,255,255,0.15)" stroke-width="1" rx="4"/>')
    svg_parts.append(f'<rect x="0" y="{height-28}" width="{width}" height="28" fill="rgba(0,0,0,0.55)" rx="0"/>')
    svg_parts.append(f'<text x="10" y="{height-10}" font-family="monospace" font-size="11" fill="#aab0c0">Grad-CAM · {pred_short}</text>')

    svg_parts.append('</svg>')
    svg_str = ''.join(svg_parts)
    b64 = base64.b64encode(svg_str.encode()).decode()
    return f'data:image/svg+xml;base64,{b64}'
