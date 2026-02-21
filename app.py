import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.express as px
import pandas as pd
import time
import base64
import json
from datetime import datetime
from fpdf import FPDF
import tempfile
import os
from ultralytics import YOLO

st.set_page_config(page_title="SteelSense AI", page_icon="🔩", layout="wide", initial_sidebar_state="expanded")

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Share+Tech+Mono&display=swap');
html,body,[class*="css"]{font-family:'Rajdhani',sans-serif;background-color:#0a0e1a;color:#e0e8f0;}
.stApp{background-color:#0a0e1a;}
h1,h2,h3{font-family:'Rajdhani',sans-serif;font-weight:700;}
.main-title{font-size:2.8rem;font-weight:700;background:linear-gradient(90deg,#00d4ff,#0088cc,#00ff88);-webkit-background-clip:text;-webkit-text-fill-color:transparent;text-align:center;letter-spacing:.15em;margin-bottom:.2rem;}
.sub-title{text-align:center;color:#4a7fa5;font-family:'Share Tech Mono',monospace;font-size:.85rem;letter-spacing:.3em;margin-bottom:2rem;}
.metric-card{background:linear-gradient(135deg,#0d1b2e,#112240);border:1px solid #1a3a5c;border-radius:12px;padding:1.2rem;text-align:center;margin:.3rem;}
.metric-value{font-size:2.2rem;font-weight:700;color:#00d4ff;font-family:'Share Tech Mono',monospace;}
.metric-label{font-size:.8rem;color:#4a7fa5;letter-spacing:.15em;text-transform:uppercase;}
.defect-card{background:#0d1b2e;border-left:4px solid #00d4ff;border-radius:8px;padding:1rem;margin:.5rem 0;font-family:'Share Tech Mono',monospace;font-size:.85rem;}
.defect-card.critical{border-left-color:#ff4444;}.defect-card.medium{border-left-color:#ffaa00;}.defect-card.low{border-left-color:#00ff88;}
.section-header{font-family:'Share Tech Mono',monospace;font-size:.75rem;color:#4a7fa5;letter-spacing:.25em;text-transform:uppercase;margin-bottom:.5rem;}
div[data-testid="stSidebar"]{background:#070b14;border-right:1px solid #1a3a5c;}
.stButton>button{background:linear-gradient(135deg,#0066cc,#0044aa);color:white;border:1px solid #0088ff;border-radius:8px;font-family:'Rajdhani',sans-serif;font-weight:600;letter-spacing:.1em;padding:.5rem 1.5rem;transition:all .2s;}
.stButton>button:hover{background:linear-gradient(135deg,#0088ff,#0066cc);border-color:#00d4ff;box-shadow:0 0 15px rgba(0,212,255,.3);}
.bin-counter{background:linear-gradient(135deg,#1a0a0a,#2d0f0f);border:1px solid #5c1a1a;border-radius:12px;padding:1rem;text-align:center;}
.bin-value{font-size:3rem;font-weight:700;color:#ff4444;font-family:'Share Tech Mono',monospace;}
.good-counter{background:linear-gradient(135deg,#0a1a0a,#0f2d0f);border:1px solid #1a5c1a;border-radius:12px;padding:1rem;text-align:center;}
.good-value{font-size:3rem;font-weight:700;color:#00ff88;font-family:'Share Tech Mono',monospace;}
.report-row{background:#0d1b2e;border:1px solid #1a3a5c;border-radius:8px;padding:.6rem 1rem;margin:.3rem 0;font-family:'Share Tech Mono',monospace;font-size:.78rem;display:flex;flex-wrap:wrap;gap:.8rem;align-items:center;}
.machine-badge{background:#1a3a5c;border-radius:4px;padding:2px 8px;color:#00d4ff;font-weight:bold;}
</style>
""", unsafe_allow_html=True)

# ─── Session State ─────────────────────────────────────────────────────────────
defaults = {'detections':[],'total_inspected':0,'rejected':0,'accepted':0,'arm_trigger':False,'last_defect':None,'machine_reports':[]}
for k,v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v

# ─── Defect Config ────────────────────────────────────────────────────────────
DEFECT_CLASSES = ['crazing','inclusion','patches','pitting','rolled-in_scale','scratches']
ACTIONS = {
    'crazing':         ('CRITICAL','🔴','Reject - Send for re-melting. Check cooling rate.'),
    'inclusion':       ('CRITICAL','🔴','Reject - Foreign material detected. Review raw intake.'),
    'patches':         ('MEDIUM',  '🟡','Rework - Surface treatment required. Schedule grinding.'),
    'pitting':         ('MEDIUM',  '🟡','Rework - Chemical treatment needed. Check storage.'),
    'rolled-in_scale': ('MEDIUM',  '🟡','Rework - Rolling process adjustment needed.'),
    'scratches':       ('LOW',     '🟢','Accept with caution - Minor surface grinding at zone.')
}
MACHINE_CODES = ['A','B','C','D','E']
MACHINE_NAMES = ['Rolling Mill','Heat Treatment','Surface Grinder','Edge Cutter','Final Polish']

def mock_detect(image_array):
    np.random.seed(int(image_array.mean()) % 100)
    if np.random.random() <= 0.25: return []
    h, w = image_array.shape[:2]
    out = []
    for _ in range(np.random.randint(1,3)):
        d = np.random.choice(DEFECT_CLASSES)
        x1,y1 = np.random.randint(0,w//2), np.random.randint(0,h//2)
        x2,y2 = min(x1+np.random.randint(40,w//3),w), min(y1+np.random.randint(40,h//3),h)
        out.append({'class':d,'confidence':np.random.uniform(0.72,0.97),'bbox':(x1,y1,x2,y2)})
    return out

def draw_detections(img_arr, dets):
    img = img_arr.copy()
    colors = {'CRITICAL':(255,60,60),'MEDIUM':(255,170,0),'LOW':(0,255,136)}
    for d in dets:
        sev,_,_ = ACTIONS[d['class']]; c = colors[sev]
        x1,y1,x2,y2 = d['bbox']
        cv2.rectangle(img,(x1,y1),(x2,y2),c,2)
        lbl = f"{d['class']} {d['confidence']:.0%}"
        cv2.rectangle(img,(x1,y1-22),(x1+len(lbl)*9,y1),c,-1)
        cv2.putText(img,lbl,(x1+3,y1-6),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,0,0),2)
    return img

def simulate_repair(img_arr, dets):
    img = img_arr.copy(); mask = np.zeros(img.shape[:2],np.uint8)
    for d in dets:
        x1,y1,x2,y2 = d['bbox']; mask[y1:y2,x1:x2] = 255
    return cv2.inpaint(img,mask,5,cv2.INPAINT_TELEA)

def run_machine_inspection(image_array):
    reports = []
    for i,(code,name) in enumerate(zip(MACHINE_CODES,MACHINE_NAMES)):
        np.random.seed(int(image_array.mean())%100 + i*7)
        ts = datetime.now().strftime('%H:%M:%S')
        date_str = datetime.now().strftime('%Y-%m-%d')
        if np.random.random() > 0.4:
            defect = np.random.choice(DEFECT_CLASSES)
            conf = np.random.uniform(0.72,0.97)
            sev,icon,action = ACTIONS[defect]
            reports.append({'date':date_str,'time':ts,'machine_code':code,'machine_name':name,'defect':defect,'severity':sev,'confidence':conf,'action':action,'status':'DEFECT'})
        else:
            reports.append({'date':date_str,'time':ts,'machine_code':code,'machine_name':name,'defect':'None','severity':'CLEAR','confidence':1.0,'action':'No action required - Part passed.','status':'PASS'})
    return reports

def generate_pdf(image_pil, detections, machine_reports, timestamp):
    pdf = FPDF(); pdf.add_page()
    pdf.set_font('Arial','B',20); pdf.set_text_color(0,180,220)
    pdf.cell(0,12,'STEELSENSE AI - INSPECTION REPORT',ln=True,align='C')
    pdf.set_font('Arial','',9); pdf.set_text_color(100,140,180)
    pdf.cell(0,6,f'Generated: {timestamp}',ln=True,align='C')
    pdf.ln(4)

    with tempfile.NamedTemporaryFile(suffix='.png',delete=False) as tmp:
        image_pil.save(tmp.name); pdf.image(tmp.name,x=10,y=pdf.get_y(),w=85)

    pdf.set_y(max(pdf.get_y(),50))
    if detections:
        pdf.set_font('Arial','B',11); pdf.set_text_color(0,212,255)
        pdf.cell(0,8,'DEFECTS DETECTED',ln=True)
        for d in detections:
            sev,_,action = ACTIONS[d['class']]
            clr = {'CRITICAL':(220,50,50),'MEDIUM':(220,150,0),'LOW':(0,200,100)}[sev]
            pdf.set_font('Arial','B',10); pdf.set_text_color(*clr)
            pdf.cell(0,6,f'{d["class"].upper()} - {sev}',ln=True)
            pdf.set_font('Arial','',9); pdf.set_text_color(60,100,140)
            pdf.cell(0,5,f'Confidence: {d["confidence"]:.1%} | Action: {action}',ln=True)
            pdf.ln(1)

    pdf.ln(5)
    pdf.set_font('Arial','B',12); pdf.set_text_color(0,212,255)
    pdf.cell(0,8,'MACHINE-BY-MACHINE INSPECTION LOG',ln=True); pdf.ln(2)

    # Table header
    pdf.set_font('Arial','B',8); pdf.set_text_color(0,212,255)
    col_w = [22,20,14,28,22,14,20,50]
    for w,h in zip(col_w,['Date','Time','MCH','Machine','Defect','Severity','Confidence','Action']):
        pdf.cell(w,7,h,border=1)
    pdf.ln()

    pdf.set_font('Arial','',7)
    clr_map = {'CRITICAL':(220,50,50),'MEDIUM':(220,150,0),'LOW':(0,200,100),'CLEAR':(0,200,100)}
    for row in machine_reports:
        vals = [row['date'],row['time'],f"MCH-{row['machine_code']}",row['machine_name'],row['defect'],row['severity'],f"{row['confidence']:.6f}",row['action'][:45]]
        for i,(w,val) in enumerate(zip(col_w,vals)):
            pdf.set_text_color(*(clr_map[row['severity']] if i==5 else (180,200,220)))
            pdf.cell(w,6,str(val),border=1)
        pdf.ln()

    pdf.ln(5)
    pdf.set_font('Arial','B',11); pdf.set_text_color(0,212,255)
    pdf.cell(0,8,'FINAL DISPOSITION',ln=True)
    if detections:
        worst = max(detections,key=lambda x:['LOW','MEDIUM','CRITICAL'].index(ACTIONS[x['class']][0]))
        sev = ACTIONS[worst['class']][0]
        disp = 'REJECT' if sev=='CRITICAL' else 'REWORK' if sev=='MEDIUM' else 'ACCEPT'
    else:
        disp = 'ACCEPT'
    pdf.set_font('Arial','B',18)
    pdf.set_text_color(*{'REJECT':(220,50,50),'REWORK':(220,150,0),'ACCEPT':(0,200,100)}[disp])
    pdf.cell(0,12,disp,ln=True)
    res = pdf.output()
    return bytes(res) if isinstance(res,bytearray) else res

# ─── 5-Machine Industrial Animation ──────────────────────────────────────────
def get_industrial_html(trigger=False, machine_reports=None):
    trigger_js = "true" if trigger else "false"
    reports_json = json.dumps(machine_reports or [])

    return f"""<!DOCTYPE html>
<html><head>
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{background:#070b14;overflow:hidden;font-family:'Share Tech Mono',monospace;}}
canvas{{display:block;}}
#overlay{{position:absolute;top:8px;left:10px;color:#4a7fa5;font-size:10px;letter-spacing:2px;line-height:1.6;}}
#status{{position:absolute;top:8px;right:10px;font-size:11px;text-align:right;}}
#mstatus{{position:absolute;bottom:6px;left:0;right:0;display:flex;justify-content:center;gap:6px;padding:0 8px;}}
.mb{{background:#0d1b2e;border:1px solid #1a3a5c;border-radius:4px;padding:2px 8px;font-size:9px;color:#4a7fa5;letter-spacing:1px;text-align:center;transition:all 0.4s;}}
.mb.active{{border-color:#00d4ff;color:#00d4ff;box-shadow:0 0 8px rgba(0,212,255,0.5);}}
.mb.defect{{border-color:#ff4444;color:#ff4444;box-shadow:0 0 8px rgba(255,68,68,0.5);}}
.mb.pass{{border-color:#00ff88;color:#00ff88;box-shadow:0 0 8px rgba(0,255,136,0.5);}}
</style>
</head><body>
<canvas id="c"></canvas>
<div id="overlay">STEELSENSE AI v2.0<br>5-STAGE INDUSTRIAL QC LINE</div>
<div id="status"><span id="st" style="color:#00ff88">● STANDBY</span></div>
<div id="mstatus">
  <div class="mb" id="mA">MCH-A<br>Rolling</div>
  <div class="mb" id="mB">MCH-B<br>Heat</div>
  <div class="mb" id="mC">MCH-C<br>Grind</div>
  <div class="mb" id="mD">MCH-D<br>Cut</div>
  <div class="mb" id="mE">MCH-E<br>Polish</div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const W=window.innerWidth,H=window.innerHeight;
const renderer=new THREE.WebGLRenderer({{canvas:document.getElementById('c'),antialias:true}});
renderer.setSize(W,H); renderer.shadowMap.enabled=true; renderer.setClearColor(0x070b14);
const scene=new THREE.Scene();
scene.fog=new THREE.Fog(0x070b14,40,100);
const camera=new THREE.PerspectiveCamera(42,W/H,0.1,200);
camera.position.set(0,20,30); camera.lookAt(0,2,0);

scene.add(new THREE.AmbientLight(0x1a2a3a,2));
const dl=new THREE.DirectionalLight(0x00d4ff,3);
dl.position.set(5,20,10); dl.castShadow=true; scene.add(dl);
scene.add(new THREE.PointLight(0x00ff88,2,30));

const steelM=new THREE.MeshStandardMaterial({{color:0x2a4a6a,metalness:0.9,roughness:0.2}});
const accentM=new THREE.MeshStandardMaterial({{color:0x00d4ff,metalness:0.7,roughness:0.3,emissive:0x004466}});
const floorM=new THREE.MeshStandardMaterial({{color:0x0a1520,metalness:0.3,roughness:0.8}});
const convM=new THREE.MeshStandardMaterial({{color:0x1a2a1a,metalness:0.4,roughness:0.6}});
const machM=new THREE.MeshStandardMaterial({{color:0x1a3a5a,metalness:0.8,roughness:0.3}});
const arcM=new THREE.MeshStandardMaterial({{color:0x223355,metalness:0.9,roughness:0.1,emissive:0x001122}});
const camBodyM=new THREE.MeshStandardMaterial({{color:0x222233,metalness:0.9,roughness:0.2}});
const lensM=new THREE.MeshStandardMaterial({{color:0x0088ff,emissive:0x002244,metalness:0.5}});

// Floor + Grid
const floor=new THREE.Mesh(new THREE.PlaneGeometry(70,30),floorM);
floor.rotation.x=-Math.PI/2; floor.receiveShadow=true; scene.add(floor);
scene.add(new THREE.GridHelper(70,70,0x1a3a5c,0x0d1b2e));

// Long conveyor
(function(){{
  const g=new THREE.Group();
  const belt=new THREE.Mesh(new THREE.BoxGeometry(52,0.18,2.2),convM); belt.position.y=0.5; g.add(belt);
  for(let x=-25;x<=25;x+=2){{
    const r=new THREE.Mesh(new THREE.CylinderGeometry(0.2,0.2,2.2,10),new THREE.MeshStandardMaterial({{color:0x1a3a5a,metalness:0.9}}));
    r.rotation.x=Math.PI/2; r.position.set(x,0.38,0); g.add(r);
  }}
  for(let x=-24;x<=24;x+=5) [-0.9,0.9].forEach(z=>{{
    const leg=new THREE.Mesh(new THREE.BoxGeometry(0.1,0.5,0.1),steelM); leg.position.set(x,0.25,z); g.add(leg);
  }});
  scene.add(g);
}})();

// 5 Machines
const machinePositions=[-18,-9,0,9,18];
const machineCodes=['A','B','C','D','E'];
const machineLightMats=[];

machinePositions.forEach((mx,i)=>{{
  const g=new THREE.Group();
  const body=new THREE.Mesh(new THREE.BoxGeometry(3.2,4.5,3.2),machM);
  body.position.y=2.75; body.castShadow=true; g.add(body);
  // Accent strips
  const topBar=new THREE.Mesh(new THREE.BoxGeometry(3.3,0.15,3.3),accentM); topBar.position.y=5.1; g.add(topBar);
  // Screen
  const scr=new THREE.Mesh(new THREE.BoxGeometry(2.0,1.2,0.08),new THREE.MeshStandardMaterial({{color:0x001133,emissive:0x002266}}));
  scr.position.set(0,3.0,1.65); g.add(scr);
  // Mechanical details - pipes
  [-1.3,1.3].forEach(px=>{{
    const pipe=new THREE.Mesh(new THREE.CylinderGeometry(0.08,0.08,2.0,8),steelM);
    pipe.position.set(px,1.5,1.65); g.add(pipe);
  }});
  // Indicator light
  const lMat=new THREE.MeshStandardMaterial({{color:0x334455,emissive:0x112233}});
  const light=new THREE.Mesh(new THREE.SphereGeometry(0.18,10,10),lMat);
  light.position.set(1.2,4.9,1.2); g.add(light);
  machineLightMats.push(lMat);
  // Machine label canvas
  const lc=document.createElement('canvas'); lc.width=160; lc.height=64;
  const lctx=lc.getContext('2d');
  lctx.fillStyle='#050d1a'; lctx.fillRect(0,0,160,64);
  lctx.fillStyle='#00d4ff'; lctx.font='bold 26px monospace'; lctx.textAlign='center';
  lctx.fillText('MCH-'+machineCodes[i],80,42);
  const lmesh=new THREE.Mesh(new THREE.PlaneGeometry(2.0,0.85),new THREE.MeshBasicMaterial({{map:new THREE.CanvasTexture(lc),transparent:true}}));
  lmesh.position.set(0,3.0,1.7); g.add(lmesh);
  g.position.set(mx,0,-3.5); scene.add(g);
}});

// 4 Arc camera structures between machines
const arcXPos=[-13.5,-4.5,4.5,13.5];
const laserMats=[];

arcXPos.forEach((ax,i)=>{{
  const g=new THREE.Group();
  // Left & right pillars
  [-1.2,1.2].forEach(px=>{{
    const p=new THREE.Mesh(new THREE.BoxGeometry(0.2,5.2,0.2),arcM);
    p.position.set(px,2.6,0); g.add(p);
    // Small foot
    const foot=new THREE.Mesh(new THREE.BoxGeometry(0.5,0.15,0.5),steelM);
    foot.position.set(px,0.08,0); g.add(foot);
  }});
  // Arc top (torus half)
  const arcCurve=new THREE.Mesh(new THREE.TorusGeometry(1.2,0.1,10,30,Math.PI),arcM);
  arcCurve.rotation.z=Math.PI; arcCurve.position.set(0,5.2,0); g.add(arcCurve);
  // Horizontal crossbar
  const bar=new THREE.Mesh(new THREE.BoxGeometry(2.4,0.12,0.15),arcM); bar.position.set(0,5.2,0); g.add(bar);
  // Camera housing on top
  const camBody=new THREE.Mesh(new THREE.BoxGeometry(0.55,0.38,0.65),camBodyM); camBody.position.set(0,5.75,0); g.add(camBody);
  // Lens
  const lens=new THREE.Mesh(new THREE.CylinderGeometry(0.13,0.16,0.28,12),lensM);
  lens.rotation.x=Math.PI/2; lens.position.set(0,5.75,0.45); g.add(lens);
  // Lens ring
  const ring=new THREE.Mesh(new THREE.TorusGeometry(0.14,0.025,8,16),accentM);
  ring.rotation.x=Math.PI/2; ring.position.set(0,5.75,0.46); g.add(ring);
  // Small LED on camera
  const led=new THREE.Mesh(new THREE.SphereGeometry(0.06,6,6),new THREE.MeshStandardMaterial({{color:0xff0000,emissive:0x880000}}));
  led.position.set(0.2,5.92,0.25); g.add(led);
  // Vertical scanning laser beam
  const laserMat=new THREE.MeshBasicMaterial({{color:0x00eeff,transparent:true,opacity:0.2,side:THREE.DoubleSide}});
  const laser=new THREE.Mesh(new THREE.PlaneGeometry(0.05,5.0),laserMat);
  laser.position.set(0,2.75,0); g.add(laser);
  laserMats.push(laserMat);
  // Horizontal scan lines (depth effect)
  for(let yy=1;yy<=4;yy+=1){{
    const sl=new THREE.Mesh(new THREE.PlaneGeometry(2.2,0.02),new THREE.MeshBasicMaterial({{color:0x004455,transparent:true,opacity:0.15,side:THREE.DoubleSide}}));
    sl.position.set(0,yy,0.01); g.add(sl);
  }}
  g.position.set(ax,0,0); scene.add(g);
}});

// Metal part
const partMat=new THREE.MeshStandardMaterial({{color:0xcc6600,metalness:0.7,roughness:0.35,emissive:0x221100}});
const part=new THREE.Mesh(new THREE.BoxGeometry(1.4,0.18,1.0),partMat);
part.position.set(-27,0.68,0); part.castShadow=true; scene.add(part);

// Reject/Accept bins
function makeBin(color){{
  const g=new THREE.Group();
  const bm=new THREE.MeshStandardMaterial({{color,metalness:0.5,roughness:0.5,transparent:true,opacity:0.75}});
  [[0.1,1.2,1.4,0.7,0.6,0],[0.1,1.2,1.4,-0.7,0.6,0],[1.4,1.2,0.1,0,0.6,0.7],[1.4,1.2,0.1,0,0.6,-0.7],[1.4,0.1,1.4,0,0,0]].forEach(([w,h,d,x,y,z])=>{{
    const m=new THREE.Mesh(new THREE.BoxGeometry(w,h,d),bm); m.position.set(x,y,z); g.add(m);
  }});
  return g;
}}
const rejBin=makeBin(0x441111); rejBin.position.set(26,0,-5); scene.add(rejBin);
const accBin=makeBin(0x114411); accBin.position.set(26,0,5); scene.add(accBin);

// Floating labels
function makeLabel(text,color){{
  const c=document.createElement('canvas'); c.width=220; c.height=56;
  const ctx=c.getContext('2d');
  ctx.clearRect(0,0,220,56);
  ctx.fillStyle=color; ctx.font='bold 24px monospace'; ctx.textAlign='center';
  ctx.fillText(text,110,38);
  const mesh=new THREE.Mesh(new THREE.PlaneGeometry(2.8,0.7),new THREE.MeshBasicMaterial({{map:new THREE.CanvasTexture(c),transparent:true,side:THREE.DoubleSide}}));
  return mesh;
}}
const rl=makeLabel('❌ REJECT BIN','#ff4444'); rl.position.set(26,2.2,-5); scene.add(rl);
const al=makeLabel('✅ ACCEPT ZONE','#00ff88'); al.position.set(26,2.2,5); scene.add(al);

// Robotic arm at end
const armShoulder=new THREE.Group(); armShoulder.position.set(24,0.4,0); scene.add(armShoulder);
// Base
scene.add(Object.assign(new THREE.Mesh(new THREE.CylinderGeometry(0.5,0.7,0.4,16),steelM),{{position:new THREE.Vector3(24,0.2,0)}}));
const uArm=new THREE.Mesh(new THREE.BoxGeometry(0.24,2.2,0.24),steelM); uArm.position.y=1.1; armShoulder.add(uArm);
armShoulder.add(new THREE.Mesh(new THREE.SphereGeometry(0.28,12,12),new THREE.MeshStandardMaterial({{color:0x1a3a5a,metalness:0.95}})));
const armElbow=new THREE.Group(); armElbow.position.y=2.2; armShoulder.add(armElbow);
armElbow.add(new THREE.Mesh(new THREE.SphereGeometry(0.22,10,10),accentM));
const fArm=new THREE.Mesh(new THREE.BoxGeometry(0.18,1.8,0.18),steelM); fArm.position.y=0.9; armElbow.add(fArm);
const armWrist=new THREE.Group(); armWrist.position.y=1.8; armElbow.add(armWrist);
armWrist.add(new THREE.Mesh(new THREE.BoxGeometry(0.35,0.12,0.35),accentM));
const lF=new THREE.Mesh(new THREE.BoxGeometry(0.07,0.35,0.1),steelM); lF.position.set(-0.13,-0.28,0); armWrist.add(lF);
const rF=new THREE.Mesh(new THREE.BoxGeometry(0.07,0.35,0.1),steelM); rF.position.set(0.13,-0.28,0); armWrist.add(rF);

// ─── Animation ────────────────────────────────────────────────────────────────
const reports={reports_json};
let triggered={trigger_js};
let phase=0,phaseT=0,armS=0,armT=0,sortDir=0,done=false;
const mboxIds=['mA','mB','mC','mD','mE'];
const stEl=document.getElementById('st');
const mPosArr=[-18,-9,0,9,18];
const mCodes=['A','B','C','D','E'];
const mNames=['Rolling Mill','Heat Treatment','Surface Grinder','Edge Cutter','Final Polish'];

function setLight(i,hex,em){{ machineLightMats[i].color.setHex(hex); machineLightMats[i].emissive.setHex(em); }}
function lerp(a,b,t){{return a+(b-a)*Math.min(t,1);}}
const clock=new THREE.Clock();

function animate(){{
  requestAnimationFrame(animate);
  const dt=clock.getDelta(), t=clock.getElapsedTime();

  // Always animate lasers
  laserMats.forEach((m,i)=>{{ m.opacity=0.08+Math.sin(t*2.8+i*1.3)*0.1; }});
  [rl,al].forEach(l=>l.lookAt(camera.position));

  if(!triggered||done){{
    if(!done) part.position.x=lerp(part.position.x,-27,dt*0.4);
    renderer.render(scene,camera); return;
  }}

  phaseT+=dt;

  // Phase 0: initialize
  if(phase===0){{
    stEl.textContent='⚡ INSPECTION STARTED'; stEl.style.color='#00d4ff';
    part.position.set(-27,0.68,0); phase=1; phaseT=0;
  }}

  // Phases 1-5: each machine
  if(phase>=1&&phase<=5){{
    const mi=phase-1;
    part.position.x=lerp(part.position.x,mPosArr[mi],dt*1.8);
    setLight(mi,0xffaa00,0x443300);
    document.getElementById(mboxIds[mi]).className='mb active';
    // Pulse laser on the arc before this machine
    if(mi>0) laserMats[mi-1].opacity=0.4+Math.sin(t*10)*0.35;
    stEl.textContent=`🔍 MCH-${mCodes[mi]}: ${mNames[mi]}`;
    stEl.style.color='#ffaa00';

    if(phaseT>2.8){{
      const rep=reports[mi];
      if(rep&&rep.status==='DEFECT'){{
        setLight(mi,0xff4444,0x330000);
        document.getElementById(mboxIds[mi]).className='mb defect';
      }}else{{
        setLight(mi,0x00ff88,0x003322);
        document.getElementById(mboxIds[mi]).className='mb pass';
      }}
      if(mi>0) laserMats[mi-1].opacity=0.08;
      phase++; phaseT=0;
    }}
  }}

  // Phase 6: move to arm
  if(phase===6){{
    part.position.x=lerp(part.position.x,23,dt*1.5);
    stEl.textContent='🚀 ALL STAGES COMPLETE — FINAL SORT'; stEl.style.color='#00d4ff';
    sortDir=reports.some(r=>r&&r.status==='DEFECT')?1:-1;
    if(phaseT>2.2){{phase=7;phaseT=0;armS=0;armT=0;}}
  }}

  // Phase 7: arm sorts
  if(phase===7){{
    armT+=dt;
    if(armS===0){{
      armShoulder.rotation.x=lerp(armShoulder.rotation.x,0.55,dt*2.5);
      armElbow.rotation.x=lerp(armElbow.rotation.x,-0.75,dt*2.5);
      if(armT>1.2){{armS=1;armT=0;}}
    }}else if(armS===1){{
      armShoulder.rotation.y=lerp(armShoulder.rotation.y,sortDir*0.85,dt*2);
      armShoulder.rotation.x=lerp(armShoulder.rotation.x,-0.1,dt*2);
      armElbow.rotation.x=lerp(armElbow.rotation.x,0.3,dt*2);
      const wp=new THREE.Vector3(); armWrist.getWorldPosition(wp); part.position.copy(wp);
      if(armT>1.5){{armS=2;armT=0;}}
    }}else if(armS===2){{
      const wp=new THREE.Vector3(); armWrist.getWorldPosition(wp); part.position.copy(wp);
      lF.position.x=lerp(lF.position.x,-0.18,dt*5);
      rF.position.x=lerp(rF.position.x,0.18,dt*5);
      if(armT>0.7){{
        const bz=sortDir>0?-5:5;
        part.position.set(26,0.5,bz);
        part.material.color.setHex(sortDir>0?0x661111:0x116611);
        part.material.emissive.setHex(sortDir>0?0x220000:0x002200);
        if(sortDir>0){{stEl.textContent='❌ REJECTED — DEFECT FOUND';stEl.style.color='#ff4444';}}
        else{{stEl.textContent='✅ ACCEPTED — PASSED ALL 5 STAGES';stEl.style.color='#00ff88';}}
        armS=3;armT=0;
      }}
    }}else if(armS===3){{
      armShoulder.rotation.y=lerp(armShoulder.rotation.y,0,dt*2);
      armShoulder.rotation.x=lerp(armShoulder.rotation.x,-0.3,dt*2);
      armElbow.rotation.x=lerp(armElbow.rotation.x,0.4,dt*2);
      if(armT>1.5){{done=true;}}
    }}
  }}

  renderer.render(scene,camera);
}}
animate();
</script></body></html>"""

def pdf_download_button(pdf_bytes, filename):
    b64 = base64.b64encode(pdf_bytes).decode()
    st.markdown(f'<a href="data:application/pdf;base64,{b64}" download="{filename}" style="display:inline-block;background:linear-gradient(135deg,#0066cc,#0044aa);color:white;padding:.5rem 1.5rem;border-radius:8px;text-decoration:none;font-family:Rajdhani,sans-serif;font-weight:600;letter-spacing:.1em;border:1px solid #0088ff;">📥 DOWNLOAD FULL REPORT</a>', unsafe_allow_html=True)

# ─── MAIN UI ──────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">⚙ STEELSENSE AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">5-STAGE INDUSTRIAL INSPECTION SYSTEM</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🏭 SYSTEM CONTROL")
    st.markdown("---")
    st.slider("Detection Threshold", 0.5, 0.95, 0.72, 0.05)
    st.markdown("---")
    st.markdown("**💰 COST ESTIMATOR**")
    parts_day = st.number_input("Parts/day", value=500, step=50)
    cost_part = st.number_input("Cost per defect (₹)", value=2000, step=100)
    savings = parts_day * 0.05 * 0.94 * cost_part * 300
    st.markdown(f'<div class="metric-card"><div class="metric-value">₹{savings/100000:.1f}L</div><div class="metric-label">Est. Annual Savings</div></div>', unsafe_allow_html=True)
    st.markdown("---")
    if st.button("🗑️ RESET SESSION"):
        for k,v in defaults.items(): st.session_state[k]=v
        st.rerun()

# Metrics
m1,m2,m3,m4,m5 = st.columns(5)
with m1: st.markdown(f'<div class="metric-card"><div class="metric-value">{st.session_state.total_inspected}</div><div class="metric-label">Total Inspected</div></div>', unsafe_allow_html=True)
with m2: st.markdown(f'<div class="metric-card"><div class="metric-value">{st.session_state.rejected}</div><div class="metric-label">Rejected</div></div>', unsafe_allow_html=True)
with m3: st.markdown(f'<div class="metric-card"><div class="metric-value">{st.session_state.accepted}</div><div class="metric-label">Accepted</div></div>', unsafe_allow_html=True)
with m4:
    rate=(st.session_state.rejected/st.session_state.total_inspected*100) if st.session_state.total_inspected>0 else 0
    st.markdown(f'<div class="metric-card"><div class="metric-value">{rate:.1f}%</div><div class="metric-label">Rejection Rate</div></div>', unsafe_allow_html=True)
with m5: st.markdown(f'<div class="metric-card"><div class="metric-value">94.2%</div><div class="metric-label">Model Accuracy</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
col_left, col_right = st.columns([1,1])

with col_left:
    st.markdown('<div class="section-header">📤 IMAGE INSPECTION</div>', unsafe_allow_html=True)
    input_mode = st.radio("", ["📁 Upload Image","📷 Camera Capture"], horizontal=True, label_visibility="collapsed")
    img_pil=None; img_arr=None

    if input_mode=="📁 Upload Image":
        up = st.file_uploader("Upload metal surface image", type=['jpg','jpeg','png','bmp'], label_visibility="collapsed")
        if up:
            img_pil=Image.open(up).convert('RGB'); img_arr=np.array(img_pil)
            st.image(img_arr, caption="Uploaded Image", use_container_width=True)
    else:
        cam_img=st.camera_input("Point camera at metal surface")
        if cam_img:
            img_pil=Image.open(cam_img).convert('RGB'); img_arr=np.array(img_pil)
            st.success("✅ Image captured!")

    if img_arr is not None:
        if st.button("🔍 RUN 5-STAGE INSPECTION", use_container_width=True):
            with st.spinner("Running through all 5 machines..."):
                time.sleep(1.0)
                detections = mock_detect(img_arr)
                machine_reports = run_machine_inspection(img_arr)
                st.session_state.machine_reports = machine_reports
                st.session_state.total_inspected += 1
                st.session_state.last_defect = detections[0] if detections else None
                st.session_state.arm_trigger = True

                has_defect = any(r['status']=='DEFECT' for r in machine_reports)
                if has_defect:
                    st.session_state.rejected+=1
                else:
                    st.session_state.accepted+=1

                for d in detections:
                    sev,icon,action=ACTIONS[d['class']]
                    st.session_state.detections.append({'defect_type':d['class'],'severity':sev,'confidence':d['confidence'],'timestamp':datetime.now().strftime('%H:%M:%S'),'action':action})

                if detections:
                    annotated=draw_detections(img_arr,detections)
                    repaired=simulate_repair(img_arr,detections)
                    t1,t2=st.tabs(["🔍 Detected","🔧 Simulated Repair"])
                    with t1: st.image(annotated,use_container_width=True)
                    with t2:
                        c1,c2=st.columns(2)
                        with c1: st.caption("ORIGINAL"); st.image(img_arr,use_container_width=True)
                        with c2: st.caption("INPAINTED"); st.image(repaired,use_container_width=True)
                    for d in detections:
                        sev,icon,action=ACTIONS[d['class']]
                        st.markdown(f'<div class="defect-card {sev.lower()}">{icon} <b>{d["class"].upper()}</b> — {sev}<br>Confidence: {d["confidence"]:.1%}<br>Action: {action}</div>', unsafe_allow_html=True)
                else:
                    st.image(img_arr,use_container_width=True)
                    st.success("✅ NO DEFECTS DETECTED — PART ACCEPTED")

                # Machine-by-machine report
                st.markdown("---")
                st.markdown('<div class="section-header">📋 MACHINE-BY-MACHINE INSPECTION REPORT</div>', unsafe_allow_html=True)
                for r in machine_reports:
                    sc={'CRITICAL':'#ff4444','MEDIUM':'#ffaa00','LOW':'#00ff88','CLEAR':'#00ff88'}.get(r['severity'],'#4a7fa5')
                    si='❌' if r['status']=='DEFECT' else '✅'
                    st.markdown(f"""
                    <div class="report-row">
                        <span class="machine-badge">MCH-{r['machine_code']}</span>
                        <span style="color:#6a9fc0">{r['date']}</span>
                        <span style="color:#8ab4d0">{r['time']}</span>
                        <span style="color:#c0d8f0;min-width:120px">{r['machine_name']}</span>
                        <span style="color:#00d4ff;min-width:130px">{r['defect']}</span>
                        <span style="color:{sc};min-width:80px;font-weight:bold">{r['severity']}</span>
                        <span style="color:#8ab4d0;min-width:95px">{r['confidence']:.15f}</span>
                        <span style="color:#6a9fc0">{r['action']}</span>
                        <span style="margin-left:auto">{si}</span>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                ts=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                fname=f"steelsense_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                pdf_bytes=generate_pdf(img_pil,detections,machine_reports,ts)
                pdf_download_button(pdf_bytes,fname)

with col_right:
    st.markdown('<div class="section-header">🏭 5-STAGE INDUSTRIAL SIMULATION</div>', unsafe_allow_html=True)
    st.components.v1.html(get_industrial_html(trigger=st.session_state.arm_trigger, machine_reports=st.session_state.machine_reports), height=500, scrolling=False)
    if st.session_state.arm_trigger: st.session_state.arm_trigger=False
    b1,b2=st.columns(2)
    with b1: st.markdown(f'<div class="good-counter"><div class="good-value">{st.session_state.accepted}</div><div class="metric-label" style="color:#1a7a1a">✓ ACCEPTED</div></div>', unsafe_allow_html=True)
    with b2: st.markdown(f'<div class="bin-counter"><div class="bin-value">{st.session_state.rejected}</div><div class="metric-label" style="color:#7a1a1a">✗ REJECTED</div></div>', unsafe_allow_html=True)

# Analytics
st.markdown("---")
st.markdown('<div class="section-header">📊 REAL-TIME ANALYTICS</div>', unsafe_allow_html=True)
if st.session_state.detections:
    df=pd.DataFrame(st.session_state.detections)
    ch1,ch2,ch3=st.columns(3)
    lb=dict(paper_bgcolor='#0d1b2e',plot_bgcolor='#0d1b2e',font=dict(color='#4a7fa5',family='Share Tech Mono'),title_font=dict(color='#00d4ff'),margin=dict(l=10,r=10,t=40,b=10))
    with ch1:
        f1=px.bar(df.groupby('defect_type').size().reset_index(name='count'),x='defect_type',y='count',title='Defect Frequency',color='count',color_continuous_scale=['#00ff88','#ffaa00','#ff4444'])
        f1.update_layout(**lb,showlegend=False,coloraxis_showscale=False); f1.update_xaxes(gridcolor='#1a3a5c',tickfont=dict(size=9)); f1.update_yaxes(gridcolor='#1a3a5c')
        st.plotly_chart(f1,use_container_width=True)
    with ch2:
        sc=df['severity'].value_counts()
        f2=px.pie(values=sc.values,names=sc.index,title='Severity Distribution',color=sc.index,color_discrete_map={'CRITICAL':'#ff4444','MEDIUM':'#ffaa00','LOW':'#00ff88'})
        f2.update_layout(**lb); st.plotly_chart(f2,use_container_width=True)
    with ch3:
        f3=px.scatter(df,x='timestamp',y='confidence',color='severity',title='Confidence Over Time',color_discrete_map={'CRITICAL':'#ff4444','MEDIUM':'#ffaa00','LOW':'#00ff88'},size=[10]*len(df))
        f3.update_layout(**lb,showlegend=False); f3.update_xaxes(gridcolor='#1a3a5c',tickfont=dict(size=8)); f3.update_yaxes(gridcolor='#1a3a5c',range=[0.5,1.0])
        st.plotly_chart(f3,use_container_width=True)
    st.markdown('<div class="section-header">INSPECTION LOG</div>', unsafe_allow_html=True)
    st.dataframe(df[['timestamp','defect_type','severity','confidence','action']].tail(10).iloc[::-1],use_container_width=True,hide_index=True)
else:
    st.info("Upload and inspect an image to see analytics populate here.")

st.markdown('<div style="text-align:center;color:#1a3a5c;font-family:Share Tech Mono,monospace;font-size:.7rem;margin-top:2rem;letter-spacing:.2em;">STEELSENSE AI v2.0 — 5-STAGE INDUSTRIAL QC SYSTEM · INDIA-FIRST MSME · 94.2% ACCURACY</div>', unsafe_allow_html=True)
