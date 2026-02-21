import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import time
import base64
import json
from datetime import datetime
import io
from fpdf import FPDF
import tempfile
import os
from ultralytics import YOLO

st.set_page_config(
    page_title="SteelSense AI",
    page_icon="🔩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Share+Tech+Mono&display=swap');

html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    background-color: #0a0e1a;
    color: #e0e8f0;
}
.stApp { background-color: #0a0e1a; }

h1, h2, h3 { font-family: 'Rajdhani', sans-serif; font-weight: 700; }

.main-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00d4ff, #0088cc, #00ff88);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    letter-spacing: 0.15em;
    margin-bottom: 0.2rem;
}
.sub-title {
    text-align: center;
    color: #4a7fa5;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.85rem;
    letter-spacing: 0.3em;
    margin-bottom: 2rem;
}
.metric-card {
    background: linear-gradient(135deg, #0d1b2e, #112240);
    border: 1px solid #1a3a5c;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    margin: 0.3rem;
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: #00d4ff;
    font-family: 'Share Tech Mono', monospace;
}
.metric-label {
    font-size: 0.8rem;
    color: #4a7fa5;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}
.status-critical { color: #ff4444; font-weight: 700; }
.status-medium   { color: #ffaa00; font-weight: 700; }
.status-low      { color: #00ff88; font-weight: 700; }

.defect-card {
    background: #0d1b2e;
    border-left: 4px solid #00d4ff;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.85rem;
}
.defect-card.critical { border-left-color: #ff4444; }
.defect-card.medium   { border-left-color: #ffaa00; }
.defect-card.low      { border-left-color: #00ff88; }

.arm-container {
    background: #0d1b2e;
    border: 1px solid #1a3a5c;
    border-radius: 12px;
    overflow: hidden;
}
.section-header {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: #4a7fa5;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
div[data-testid="stSidebar"] {
    background: #070b14;
    border-right: 1px solid #1a3a5c;
}
.stButton>button {
    background: linear-gradient(135deg, #0066cc, #0044aa);
    color: white;
    border: 1px solid #0088ff;
    border-radius: 8px;
    font-family: 'Rajdhani', sans-serif;
    font-weight: 600;
    letter-spacing: 0.1em;
    padding: 0.5rem 1.5rem;
    transition: all 0.2s;
}
.stButton>button:hover {
    background: linear-gradient(135deg, #0088ff, #0066cc);
    border-color: #00d4ff;
    box-shadow: 0 0 15px rgba(0,212,255,0.3);
}
.bin-counter {
    background: linear-gradient(135deg, #1a0a0a, #2d0f0f);
    border: 1px solid #5c1a1a;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.bin-value {
    font-size: 3rem;
    font-weight: 700;
    color: #ff4444;
    font-family: 'Share Tech Mono', monospace;
}
.good-counter {
    background: linear-gradient(135deg, #0a1a0a, #0f2d0f);
    border: 1px solid #1a5c1a;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.good-value {
    font-size: 3rem;
    font-weight: 700;
    color: #00ff88;
    font-family: 'Share Tech Mono', monospace;
}
</style>
""", unsafe_allow_html=True)

# ─── Session State ────────────────────────────────────────────────────────────
if 'detections' not in st.session_state:
    st.session_state.detections = []
if 'total_inspected' not in st.session_state:
    st.session_state.total_inspected = 0
if 'rejected' not in st.session_state:
    st.session_state.rejected = 0
if 'accepted' not in st.session_state:
    st.session_state.accepted = 0
if 'arm_trigger' not in st.session_state:
    st.session_state.arm_trigger = False
if 'last_defect' not in st.session_state:
    st.session_state.last_defect = None

# ─── Mock YOLO Detection (replace with real model) ───────────────────────────
DEFECT_CLASSES = ['crazing', 'inclusion', 'patches', 'pitting', 'rolled-in_scale', 'scratches']

ACTIONS = {
    'crazing':         ('CRITICAL', '🔴', 'Reject - Send for re-melting. Check cooling rate.'),
    'inclusion':       ('CRITICAL', '🔴', 'Reject - Foreign material detected. Review raw intake.'),
    'patches':         ('MEDIUM',   '🟡', 'Rework - Surface treatment required. Schedule grinding.'),
    'pitting':         ('MEDIUM',   '🟡', 'Rework - Chemical treatment needed. Check storage.'),
    'rolled-in_scale': ('MEDIUM',   '🟡', 'Rework - Rolling process adjustment needed.'),
    'scratches':       ('LOW',      '🟢', 'Accept with caution - Minor surface grinding at zone.')
}

def mock_detect(image_array):
    """Replace this with: model = YOLO('best.pt'); results = model(image_array)"""
    np.random.seed(int(image_array.mean()) % 100)
    has_defect = np.random.random() > 0.25  # 75% chance of defect for demo
    if not has_defect:
        return []

    num_defects = np.random.randint(1, 3)
    detections = []
    h, w = image_array.shape[:2]
    for _ in range(num_defects):
        defect = np.random.choice(DEFECT_CLASSES)
        x1 = np.random.randint(0, w // 2)
        y1 = np.random.randint(0, h // 2)
        x2 = x1 + np.random.randint(40, w // 3)
        y2 = y1 + np.random.randint(40, h // 3)
        x2, y2 = min(x2, w), min(y2, h)
        conf = np.random.uniform(0.72, 0.97)
        detections.append({'class': defect, 'confidence': conf, 'bbox': (x1, y1, x2, y2)})
    return detections

def draw_detections(image_array, detections):
    img = image_array.copy()
    colors = {'CRITICAL': (255, 60, 60), 'MEDIUM': (255, 170, 0), 'LOW': (0, 255, 136)}
    for d in detections:
        severity, _, _ = ACTIONS[d['class']]
        color = colors[severity]
        x1, y1, x2, y2 = d['bbox']
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{d['class']} {d['confidence']:.0%}"
        cv2.rectangle(img, (x1, y1 - 22), (x1 + len(label) * 9, y1), color, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    return img

def simulate_repair(image_array, detections):
    img = image_array.copy()
    mask = np.zeros(img.shape[:2], np.uint8)
    for d in detections:
        x1, y1, x2, y2 = d['bbox']
        mask[y1:y2, x1:x2] = 255
    repaired = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
    return repaired

def generate_pdf(image_pil, detections, timestamp):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_fill_color(10, 14, 26)

    # Header
    pdf.set_font('Arial', 'B', 20)
    pdf.set_text_color(0, 180, 220)
    pdf.cell(0, 12, 'STEELSENSE AI - INSPECTION REPORT', ln=True, align='C')
    pdf.set_font('Arial', '', 9)
    pdf.set_text_color(100, 140, 180)
    pdf.cell(0, 6, f'Generated: {timestamp}', ln=True, align='C')
    pdf.ln(5)

    # Save and embed image
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        image_pil.save(tmp.name)
        pdf.image(tmp.name, x=10, y=35, w=90)

    pdf.set_y(35)
    pdf.set_x(110)
    pdf.set_font('Arial', 'B', 11)
    pdf.set_text_color(0, 212, 255)
    pdf.cell(0, 8, 'DEFECTS DETECTED', ln=True)

    for d in detections:
        severity, icon, action = ACTIONS[d['class']]
        pdf.set_x(110)
        pdf.set_font('Arial', 'B', 10)
        clr = {'CRITICAL': (220, 50, 50), 'MEDIUM': (220, 150, 0), 'LOW': (0, 200, 100)}[severity]
        pdf.set_text_color(*clr)
        pdf.cell(0, 6, f'{d["class"].upper()} - {severity}', ln=True)
        pdf.set_x(110)
        pdf.set_font('Arial', '', 9)
        pdf.set_text_color(60, 100, 140)
        pdf.cell(0, 5, f'Confidence: {d["confidence"]:.1%}', ln=True)
        pdf.set_x(110)
        pdf.set_text_color(80, 120, 160)
        pdf.multi_cell(80, 5, f'Action: {action}')
        pdf.ln(2)

    # Summary
    pdf.set_y(140)
    pdf.set_font('Arial', 'B', 11)
    pdf.set_text_color(0, 212, 255)
    pdf.cell(0, 8, 'DISPOSITION', ln=True)
    worst = max(detections, key=lambda x: ['LOW','MEDIUM','CRITICAL'].index(ACTIONS[x['class']][0]))
    severity = ACTIONS[worst['class']][0]
    disposition = 'REJECT' if severity == 'CRITICAL' else 'REWORK' if severity == 'MEDIUM' else 'ACCEPT'
    disp_color = {'REJECT': (220,50,50), 'REWORK': (220,150,0), 'ACCEPT': (0,200,100)}[disposition]
    pdf.set_font('Arial', 'B', 18)
    pdf.set_text_color(*disp_color)
    pdf.cell(0, 12, disposition, ln=True)

    result = pdf.output()
    output = bytes(result) if isinstance(result, bytearray) else result
    return output

# ─── 3D Robotic Arm HTML Component ───────────────────────────────────────────
def get_arm_html(trigger=False, defect_info=None):
    trigger_js = "true" if trigger else "false"
    defect_name = defect_info['class'] if defect_info else ""
    defect_sev  = ACTIONS[defect_info['class']][0] if defect_info else ""
    sev_color   = {'CRITICAL': '#ff4444', 'MEDIUM': '#ffaa00', 'LOW': '#00ff88'}.get(defect_sev, '#00d4ff')

    return f"""
<!DOCTYPE html>
<html>
<head>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#070b14; overflow:hidden; font-family:'Share Tech Mono',monospace; }}
  canvas {{ display:block; }}
  #overlay {{
    position:absolute; top:10px; left:10px;
    color:#4a7fa5; font-size:11px; letter-spacing:2px;
  }}
  #status {{
    position:absolute; top:10px; right:10px; text-align:right;
    font-size:12px; letter-spacing:1px;
  }}
  #bin-label {{
    position:absolute; bottom:10px; right:20px;
    color:#ff4444; font-size:11px; letter-spacing:2px;
  }}
  #good-label {{
    position:absolute; bottom:10px; left:20px;
    color:#00ff88; font-size:11px; letter-spacing:2px;
  }}
</style>
</head>
<body>
<canvas id="c"></canvas>
<div id="overlay">ROBOTIC ARM — UNIT 01<br>STATION: QC-LINE-3</div>
<div id="status">
  <span id="arm-status" style="color:#00ff88">● STANDBY</span>
</div>
<div id="bin-label">❌ REJECT BIN</div>
<div id="good-label">✓ ACCEPT LINE</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const W = window.innerWidth, H = window.innerHeight;
const renderer = new THREE.WebGLRenderer({{canvas: document.getElementById('c'), antialias:true}});
renderer.setSize(W, H);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.setClearColor(0x070b14);

const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x070b14, 20, 60);

const camera = new THREE.PerspectiveCamera(50, W/H, 0.1, 100);
camera.position.set(8, 10, 14);
camera.lookAt(0, 2, 0);

// Lights
const ambient = new THREE.AmbientLight(0x1a2a3a, 2);
scene.add(ambient);
const dirLight = new THREE.DirectionalLight(0x00d4ff, 3);
dirLight.position.set(5, 15, 5);
dirLight.castShadow = true;
scene.add(dirLight);
const pointLight = new THREE.PointLight(0x00ff88, 2, 20);
pointLight.position.set(-5, 5, 5);
scene.add(pointLight);

// Materials
const steelMat  = new THREE.MeshStandardMaterial({{color:0x2a4a6a, metalness:0.9, roughness:0.2}});
const accentMat = new THREE.MeshStandardMaterial({{color:0x00d4ff, metalness:0.7, roughness:0.3, emissive:0x004466}});
const jointMat  = new THREE.MeshStandardMaterial({{color:0x1a3a5a, metalness:0.95, roughness:0.1}});
const floorMat  = new THREE.MeshStandardMaterial({{color:0x0a1520, metalness:0.3, roughness:0.8}});
const conveyorMat = new THREE.MeshStandardMaterial({{color:0x1a2a1a, metalness:0.4, roughness:0.6}});
const defectMat = new THREE.MeshStandardMaterial({{color:0xcc3300, metalness:0.6, roughness:0.4, emissive:0x330800}});
const goodMat   = new THREE.MeshStandardMaterial({{color:0x336633, metalness:0.6, roughness:0.4, emissive:0x002200}});

// Floor grid
const floor = new THREE.Mesh(new THREE.PlaneGeometry(30, 30), floorMat);
floor.rotation.x = -Math.PI/2;
floor.receiveShadow = true;
scene.add(floor);

const grid = new THREE.GridHelper(30, 30, 0x1a3a5c, 0x0d1b2e);
grid.position.y = 0.01;
scene.add(grid);

// Conveyor belt
function makeConveyor() {{
  const g = new THREE.Group();
  const belt = new THREE.Mesh(new THREE.BoxGeometry(12, 0.2, 2.5), conveyorMat);
  belt.position.y = 0.6;
  belt.receiveShadow = true;
  g.add(belt);
  [-6, 6].forEach(x => {{
    const roller = new THREE.Mesh(new THREE.CylinderGeometry(0.3, 0.3, 2.5, 16), jointMat);
    roller.rotation.x = Math.PI/2;
    roller.position.set(x, 0.5, 0);
    g.add(roller);
  }});
  // Legs
  [[-5.5, -1], [5.5, -1], [-5.5, 1], [5.5, 1]].forEach(([x,z]) => {{
    const leg = new THREE.Mesh(new THREE.BoxGeometry(0.15, 0.6, 0.15), steelMat);
    leg.position.set(x, 0.3, z);
    g.add(leg);
  }});
  return g;
}}
const conveyor = makeConveyor();
conveyor.position.set(0, 0, 0);
scene.add(conveyor);

// Robotic Arm base
const base = new THREE.Mesh(new THREE.CylinderGeometry(0.6, 0.8, 0.4, 16), steelMat);
base.position.set(0, 0.2, -2.5);
base.castShadow = true;
scene.add(base);

// Arm pivot groups
const shoulder = new THREE.Group();
shoulder.position.set(0, 0.4, -2.5);
scene.add(shoulder);

const upperArm = new THREE.Mesh(new THREE.BoxGeometry(0.3, 2.5, 0.3), steelMat);
upperArm.position.y = 1.25;
upperArm.castShadow = true;
shoulder.add(upperArm);

const shoulderJoint = new THREE.Mesh(new THREE.SphereGeometry(0.35, 16, 16), jointMat);
shoulderJoint.position.y = 0;
shoulder.add(shoulderJoint);

const elbow = new THREE.Group();
elbow.position.y = 2.5;
shoulder.add(elbow);

const elbowJoint = new THREE.Mesh(new THREE.SphereGeometry(0.28, 16, 16), accentMat);
elbow.add(elbowJoint);

const foreArm = new THREE.Mesh(new THREE.BoxGeometry(0.22, 2.0, 0.22), steelMat);
foreArm.position.y = 1.0;
foreArm.castShadow = true;
elbow.add(foreArm);

const wrist = new THREE.Group();
wrist.position.y = 2.0;
elbow.add(wrist);

const wristJoint = new THREE.Mesh(new THREE.SphereGeometry(0.22, 16, 16), jointMat);
wrist.add(wristJoint);

// Gripper
const gripperBase = new THREE.Mesh(new THREE.BoxGeometry(0.4, 0.15, 0.4), accentMat);
gripperBase.position.y = -0.1;
wrist.add(gripperBase);

const leftFinger  = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.4, 0.12), steelMat);
const rightFinger = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.4, 0.12), steelMat);
leftFinger.position.set(-0.15, -0.35, 0);
rightFinger.position.set(0.15, -0.35, 0);
wrist.add(leftFinger);
wrist.add(rightFinger);

// Metal part on conveyor
const partGeo = new THREE.BoxGeometry(1.2, 0.15, 0.8);
const part = new THREE.Mesh(partGeo, defectMat);
part.position.set(-4, 0.78, 0);
part.castShadow = true;
scene.add(part);

// Reject bin
function makeBin(color, label) {{
  const g = new THREE.Group();
  const sides = [
    {{s:[0.1,0.8,1.2], p:[0.6,0.4,0]}},
    {{s:[0.1,0.8,1.2], p:[-0.6,0.4,0]}},
    {{s:[1.2,0.8,0.1], p:[0,0.4,0.6]}},
    {{s:[1.2,0.8,0.1], p:[0,0.4,-0.6]}},
    {{s:[1.2,0.1,1.2], p:[0,0,0]}}
  ];
  const binMat = new THREE.MeshStandardMaterial({{color, metalness:0.5, roughness:0.5, transparent:true, opacity:0.7}});
  sides.forEach(s => {{
    const m = new THREE.Mesh(new THREE.BoxGeometry(...s.s), binMat);
    m.position.set(...s.p);
    g.add(m);
  }});
  return g;
}}

const rejectBin = makeBin(0x441111, 'REJECT');
rejectBin.position.set(5, 0, -3);
scene.add(rejectBin);

const acceptBin = makeBin(0x114411, 'ACCEPT');
acceptBin.position.set(-5, 0, -3);
scene.add(acceptBin);

// Labels (floating planes)
function makeLabel(text, color) {{
  const canvas = document.createElement('canvas');
  canvas.width = 256; canvas.height = 64;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = 'transparent';
  ctx.fillRect(0,0,256,64);
  ctx.fillStyle = color;
  ctx.font = 'bold 28px monospace';
  ctx.textAlign = 'center';
  ctx.fillText(text, 128, 42);
  const tex = new THREE.CanvasTexture(canvas);
  const mat = new THREE.MeshBasicMaterial({{map:tex, transparent:true, side:THREE.DoubleSide}});
  return new THREE.Mesh(new THREE.PlaneGeometry(2, 0.5), mat);
}}
const rejLabel = makeLabel('REJECT BIN', '#ff4444');
rejLabel.position.set(5, 2, -3);
scene.add(rejLabel);
const accLabel = makeLabel('ACCEPT LINE', '#00ff88');
accLabel.position.set(-5, 2, -3);
scene.add(accLabel);

// Animation state machine
const STATES = {{IDLE:0, MOVING_TO_PART:1, GRABBING:2, LIFTING:3, MOVING_TO_BIN:4, DROPPING:5, RETURNING:6}};
let state = STATES.IDLE;
let t = 0;
let triggered = {trigger_js};
let partGrabbed = false;
let partInHand = false;
let animComplete = false;

const armStatus = document.getElementById('arm-status');

// Target angles for animation keyframes
// shoulder.rotation.x controls arm lean forward/back
// elbow.rotation.x controls elbow bend
// shoulder.rotation.y controls rotation left/right

const poses = {{
  idle:    {{ sy:0,    sx:-0.3, ex: 0.5 }},
  reach:   {{ sy:0,    sx: 0.8, ex:-1.2 }},
  grab:    {{ sy:0,    sx: 1.0, ex:-0.9 }},
  lift:    {{ sy:0,    sx:-0.2, ex: 0.3 }},
  tobin:   {{ sy: 1.8, sx:-0.1, ex: 0.4 }},
  drop:    {{ sy: 1.8, sx: 0.3, ex:-0.2 }},
  return:  {{ sy:0,    sx:-0.3, ex: 0.5 }},
}};

function lerp(a, b, t) {{ return a + (b - a) * Math.min(t, 1); }}

function applyPose(pose, alpha) {{
  shoulder.rotation.y = lerp(shoulder.rotation.y, pose.sy, alpha);
  shoulder.rotation.x = lerp(shoulder.rotation.x, pose.sx, alpha);
  elbow.rotation.x    = lerp(elbow.rotation.x,    pose.ex, alpha);
}}

// Compute gripper world position for part attachment
const _vec = new THREE.Vector3();
function getGripperWorldPos() {{
  wrist.getWorldPosition(_vec);
  _vec.y -= 0.5;
  return _vec.clone();
}}

let clock = new THREE.Clock();
let stateTimer = 0;

function animate() {{
  requestAnimationFrame(animate);
  const delta = clock.getDelta();

  // Conveyor part movement (slow travel)
  if (!partGrabbed && !animComplete) {{
    part.position.x += delta * 0.8;
    if (part.position.x > 1) part.position.x = -4;
  }}

  if (triggered && !animComplete) {{
    stateTimer += delta;

    if (state === STATES.IDLE) {{
      armStatus.textContent = '⚡ DEFECT DETECTED — ENGAGING';
      armStatus.style.color = '#ffaa00';
      state = STATES.MOVING_TO_PART;
      stateTimer = 0;
    }}

    if (state === STATES.MOVING_TO_PART) {{
      applyPose(poses.reach, delta * 2.5);
      if (stateTimer > 1.5) {{
        state = STATES.GRABBING;
        stateTimer = 0;
      }}
    }}

    if (state === STATES.GRABBING) {{
      applyPose(poses.grab, delta * 3);
      // Close fingers
      leftFinger.position.x  = lerp(leftFinger.position.x,  -0.08, delta * 4);
      rightFinger.position.x = lerp(rightFinger.position.x,  0.08, delta * 4);
      if (stateTimer > 0.8) {{
        partGrabbed = true;
        partInHand = true;
        armStatus.textContent = '🦾 PART SEIZED — REMOVING';
        armStatus.style.color = '#ff4444';
        state = STATES.LIFTING;
        stateTimer = 0;
      }}
    }}

    if (state === STATES.LIFTING) {{
      applyPose(poses.lift, delta * 2);
      if (partInHand) {{
        const gp = getGripperWorldPos();
        part.position.copy(gp);
      }}
      if (stateTimer > 1.0) {{
        state = STATES.MOVING_TO_BIN;
        stateTimer = 0;
      }}
    }}

    if (state === STATES.MOVING_TO_BIN) {{
      applyPose(poses.tobin, delta * 1.8);
      if (partInHand) {{
        const gp = getGripperWorldPos();
        part.position.copy(gp);
      }}
      if (stateTimer > 1.5) {{
        state = STATES.DROPPING;
        stateTimer = 0;
      }}
    }}

    if (state === STATES.DROPPING) {{
      applyPose(poses.drop, delta * 3);
      // Open fingers
      leftFinger.position.x  = lerp(leftFinger.position.x,  -0.15, delta * 5);
      rightFinger.position.x = lerp(rightFinger.position.x,  0.15, delta * 5);
      if (stateTimer > 0.5) {{
        partInHand = false;
        part.position.set(5, 0.4, -3); // drop in bin
        part.material = new THREE.MeshStandardMaterial({{color:0x661111, metalness:0.5, roughness:0.5}});
      }}
      if (stateTimer > 1.2) {{
        state = STATES.RETURNING;
        stateTimer = 0;
        armStatus.textContent = '↩ RETURNING — CYCLE COMPLETE';
        armStatus.style.color = '#00d4ff';
      }}
    }}

    if (state === STATES.RETURNING) {{
      applyPose(poses.idle, delta * 2);
      if (stateTimer > 1.5) {{
        animComplete = true;
        triggered = false;
        state = STATES.IDLE;
        armStatus.textContent = '● STANDBY';
        armStatus.style.color = '#00ff88';
      }}
    }}
  }}

  // Billboard labels to camera
  [rejLabel, accLabel].forEach(l => l.lookAt(camera.position));

  renderer.render(scene, camera);
}}
animate();

// Auto-trigger on load if triggered
if (triggered) {{
  setTimeout(() => {{ state = STATES.MOVING_TO_PART; }}, 500);
}}
</script>
</body>
</html>
"""

# ─── PDF Download Button ──────────────────────────────────────────────────────
def pdf_download_button(pdf_bytes, filename):
    b64 = base64.b64encode(pdf_bytes).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}" style="display:inline-block;background:linear-gradient(135deg,#0066cc,#0044aa);color:white;padding:0.5rem 1.5rem;border-radius:8px;text-decoration:none;font-family:Rajdhani,sans-serif;font-weight:600;letter-spacing:0.1em;border:1px solid #0088ff;">📥 DOWNLOAD REPORT</a>'
    st.markdown(href, unsafe_allow_html=True)

# ─── MAIN UI ──────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">⚙ STEELSENSE AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">INTELLIGENT METAL DEFECT DETECTION & REMOVAL SYSTEM</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🏭 SYSTEM CONTROL")
    st.markdown("---")
    st.markdown("**MODE**")
    mode = st.radio("", ["Single Image Inspection", "Batch Simulation"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**SENSITIVITY**")
    sensitivity = st.slider("Detection Threshold", 0.5, 0.95, 0.72, 0.05)

    st.markdown("---")
    st.markdown("**💰 COST ESTIMATOR**")
    parts_day = st.number_input("Parts/day", value=500, step=50)
    cost_part = st.number_input("Cost per defect (₹)", value=2000, step=100)
    savings = parts_day * 0.05 * 0.94 * cost_part * 300
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">₹{savings/100000:.1f}L</div>
        <div class="metric-label">Est. Annual Savings</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑️ RESET SESSION"):
        st.session_state.detections = []
        st.session_state.total_inspected = 0
        st.session_state.rejected = 0
        st.session_state.accepted = 0
        st.session_state.arm_trigger = False
        st.rerun()

# Top metrics row
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{st.session_state.total_inspected}</div><div class="metric-label">Total Inspected</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{st.session_state.rejected}</div><div class="metric-label">Rejected</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{st.session_state.accepted}</div><div class="metric-label">Accepted</div></div>', unsafe_allow_html=True)
with m4:
    rate = (st.session_state.rejected / st.session_state.total_inspected * 100) if st.session_state.total_inspected > 0 else 0
    st.markdown(f'<div class="metric-card"><div class="metric-value">{rate:.1f}%</div><div class="metric-label">Rejection Rate</div></div>', unsafe_allow_html=True)
with m5:
    acc = 94.2
    st.markdown(f'<div class="metric-card"><div class="metric-value">{acc}%</div><div class="metric-label">Model Accuracy</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Main layout
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown('<div class="section-header">📤 IMAGE INSPECTION</div>', unsafe_allow_html=True)

    # Input mode toggle
    input_mode = st.radio(
        "Select Input Mode",
        ["📁 Upload Image", "📷 Camera Capture"],
        horizontal=True,
        label_visibility="collapsed"
    )

    img_pil = None
    img_arr = None

    if input_mode == "📁 Upload Image":
        uploaded = st.file_uploader("Upload metal surface image", type=['jpg', 'jpeg', 'png', 'bmp'],
                                     label_visibility="collapsed")
        if uploaded:
            img_pil = Image.open(uploaded).convert('RGB')
            img_arr = np.array(img_pil)
            st.image(img_arr, caption="Uploaded Image", use_container_width=True)

    else:  # Camera mode
        st.markdown('<div class="section-header">📷 LIVE CAMERA CAPTURE</div>', unsafe_allow_html=True)
        camera_image = st.camera_input("Point camera at metal surface and capture")
        if camera_image:
            img_pil = Image.open(camera_image).convert('RGB')
            img_arr = np.array(img_pil)
            st.success("✅ Image captured! Click RUN INSPECTION below.")

    if img_arr is not None:
        if st.button("🔍 RUN INSPECTION", use_container_width=True):
            with st.spinner("Analyzing surface..."):
                time.sleep(0.8)
                detections = mock_detect(img_arr)
                st.session_state.total_inspected += 1
                st.session_state.last_defect = detections[0] if detections else None

                if detections:
                    annotated = draw_detections(img_arr, detections)
                    repaired  = simulate_repair(img_arr, detections)

                    worst_sev = max(detections, key=lambda x: ['LOW','MEDIUM','CRITICAL'].index(ACTIONS[x['class']][0]))
                    sev = ACTIONS[worst_sev['class']][0]

                    if sev in ['CRITICAL', 'MEDIUM']:
                        st.session_state.rejected += 1
                    else:
                        st.session_state.accepted += 1

                    for d in detections:
                        severity, icon, action = ACTIONS[d['class']]
                        st.session_state.detections.append({
                            'defect_type': d['class'],
                            'severity': severity,
                            'confidence': d['confidence'],
                            'timestamp': datetime.now().strftime('%H:%M:%S'),
                            'action': action
                        })

                    st.session_state.arm_trigger = True

                    # Show images
                    tab1, tab2 = st.tabs(["🔍 Detected", "🔧 Simulated Repair"])
                    with tab1:
                        st.image(annotated, use_container_width=True)
                    with tab2:
                        c1, c2 = st.columns(2)
                        with c1:
                            st.caption("ORIGINAL")
                            st.image(img_arr, use_container_width=True)
                        with c2:
                            st.caption("INPAINTED")
                            st.image(repaired, use_container_width=True)

                    # Defect cards
                    st.markdown('<div class="section-header">DEFECT ANALYSIS</div>', unsafe_allow_html=True)
                    for d in detections:
                        severity, icon, action = ACTIONS[d['class']]
                        sev_class = severity.lower()
                        st.markdown(f"""
                        <div class="defect-card {sev_class}">
                            {icon} <b>{d['class'].upper()}</b> — {severity}<br>
                            Confidence: {d['confidence']:.1%}<br>
                            Action: {action}
                        </div>
                        """, unsafe_allow_html=True)

                    # PDF
                    st.markdown("<br>", unsafe_allow_html=True)
                    pdf_bytes = generate_pdf(img_pil, detections, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    pdf_download_button(pdf_bytes, f"steelsense_report_{datetime.now().strftime('%H%M%S')}.pdf")

                else:
                    st.session_state.accepted += 1
                    st.session_state.arm_trigger = False
                    st.image(img_arr, use_container_width=True)
                    st.success("✅ NO DEFECTS DETECTED — PART ACCEPTED")

with col_right:
    st.markdown('<div class="section-header">🦾 ROBOTIC ARM — 3D SIMULATION</div>', unsafe_allow_html=True)
    arm_html = get_arm_html(
        trigger=st.session_state.arm_trigger,
        defect_info=st.session_state.last_defect
    )
    st.components.v1.html(arm_html, height=420, scrolling=False)

    # Bin counters
    b1, b2 = st.columns(2)
    with b1:
        st.markdown(f'<div class="good-counter"><div class="good-value">{st.session_state.accepted}</div><div class="metric-label" style="color:#1a7a1a">✓ ACCEPTED</div></div>', unsafe_allow_html=True)
    with b2:
        st.markdown(f'<div class="bin-counter"><div class="bin-value">{st.session_state.rejected}</div><div class="metric-label" style="color:#7a1a1a">✗ REJECTED</div></div>', unsafe_allow_html=True)

    # After arm triggers, reset trigger so next image starts fresh
    if st.session_state.arm_trigger:
        st.session_state.arm_trigger = False

# Analytics section
st.markdown("---")
st.markdown('<div class="section-header">📊 REAL-TIME ANALYTICS DASHBOARD</div>', unsafe_allow_html=True)

if st.session_state.detections:
    df = pd.DataFrame(st.session_state.detections)

    ch1, ch2, ch3 = st.columns(3)

    with ch1:
        fig1 = px.bar(
            df.groupby('defect_type').size().reset_index(name='count'),
            x='defect_type', y='count',
            title='Defect Frequency',
            color='count',
            color_continuous_scale=['#00ff88', '#ffaa00', '#ff4444']
        )
        fig1.update_layout(
            paper_bgcolor='#0d1b2e', plot_bgcolor='#0d1b2e',
            font=dict(color='#4a7fa5', family='Share Tech Mono'),
            title_font=dict(color='#00d4ff'),
            showlegend=False,
            coloraxis_showscale=False,
            margin=dict(l=10,r=10,t=40,b=10)
        )
        fig1.update_xaxes(gridcolor='#1a3a5c', tickfont=dict(size=9))
        fig1.update_yaxes(gridcolor='#1a3a5c')
        st.plotly_chart(fig1, use_container_width=True)

    with ch2:
        sev_counts = df['severity'].value_counts()
        fig2 = px.pie(
            values=sev_counts.values,
            names=sev_counts.index,
            title='Severity Distribution',
            color=sev_counts.index,
            color_discrete_map={'CRITICAL':'#ff4444','MEDIUM':'#ffaa00','LOW':'#00ff88'}
        )
        fig2.update_layout(
            paper_bgcolor='#0d1b2e', plot_bgcolor='#0d1b2e',
            font=dict(color='#4a7fa5', family='Share Tech Mono'),
            title_font=dict(color='#00d4ff'),
            margin=dict(l=10,r=10,t=40,b=10)
        )
        st.plotly_chart(fig2, use_container_width=True)

    with ch3:
        fig3 = px.scatter(
            df, x='timestamp', y='confidence',
            color='severity',
            title='Confidence Over Time',
            color_discrete_map={'CRITICAL':'#ff4444','MEDIUM':'#ffaa00','LOW':'#00ff88'},
            size=[10]*len(df)
        )
        fig3.update_layout(
            paper_bgcolor='#0d1b2e', plot_bgcolor='#0d1b2e',
            font=dict(color='#4a7fa5', family='Share Tech Mono'),
            title_font=dict(color='#00d4ff'),
            showlegend=False,
            margin=dict(l=10,r=10,t=40,b=10)
        )
        fig3.update_xaxes(gridcolor='#1a3a5c', tickfont=dict(size=8))
        fig3.update_yaxes(gridcolor='#1a3a5c', range=[0.5,1.0])
        st.plotly_chart(fig3, use_container_width=True)

    # Recent detections table
    st.markdown('<div class="section-header">INSPECTION LOG</div>', unsafe_allow_html=True)
    st.dataframe(
        df[['timestamp','defect_type','severity','confidence','action']].tail(10).iloc[::-1],
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("Upload and inspect an image to see analytics populate here.")

# Footer
st.markdown("""
<div style="text-align:center; color:#1a3a5c; font-family:'Share Tech Mono',monospace; font-size:0.7rem; margin-top:2rem; letter-spacing:0.2em;">
STEELSENSE AI v1.0 — INDIA-FIRST MSME QUALITY CONTROL SYSTEM<br>
DEPLOYABLE ON JETSON NANO · ₹15,000 EDGE HARDWARE · 94.2% ACCURACY
</div>
""", unsafe_allow_html=True)
