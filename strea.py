import os
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
import base64
from io import BytesIO
from PIL import Image
from predictor import CascadePredictor

# ==========================================
# 1. CONFIGURACIÓN DE PÁGINA
# ==========================================
# Load favicon if available
try:
    ICON_PATH = os.path.join(os.path.dirname(__file__), "assets", "122402012.png")
    favicon = Image.open(ICON_PATH)
except:
    favicon = "🧬"

st.set_page_config(
    page_title="Structural Bioinformatics Interface",
    page_icon=favicon,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 2. ESTILOS CSS (LIGHT SCIENTIFIC)
# ==========================================
LIGHT_SCI_CSS = """
<style>
:root{
  --bg: #F7F9FC;
  --panel: #FFFFFF;
  --panel2: #FBFCFE;
  --text: #111827;
  --muted: #6B7280;
  --line: rgba(17,24,39,0.10);
  --accent: #2563EB;
  --accentSoft: rgba(37,99,235,0.10);
  --radius: 16px;
}

/* Page background */
.stApp{
  background: var(--bg);
}

/* Spacing */
.block-container{
  padding-top: 2.2rem;
  padding-bottom: 2.6rem;
  padding-left: 2.0rem;
  padding-right: 2.0rem;
  max-width: 1400px;
}
div[data-testid="stHorizontalBlock"]{ gap: 1.25rem; }

/* Sidebar */
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, #FFFFFF, #FAFBFF);
  border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] *{ color: var(--text); }

/* Headers */
h1, h2, h3, h4 { letter-spacing: -0.02em; color: var(--text) !important; }
h1{ margin-bottom: 0.6rem; font-weight: 760; }
h2{ margin-top: 0.2rem; margin-bottom: 0.6rem; }
h3{ margin-top: 0.2rem; margin-bottom: 0.6rem; }

/* Inputs */
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea,
div[data-baseweb="select"] > div{
  background: #FFFFFF !important;
  border: 1px solid var(--line) !important;
  border-radius: 12px !important;
  color: var(--text) !important;
}
div[data-baseweb="textarea"] textarea:focus {
  border-color: var(--accent) !important;
}

/* Card utility */
/* Card utility (Invisible as per request) */
.card{
  background: transparent;
  border: none;
  padding: 0;
  box-shadow: none;
  margin-bottom: 1.0rem;
}

/* Card utility (Invisible as per request) */
.card{
  background: transparent;
  border: none;
  padding: 0;
  box-shadow: none;
  margin-bottom: 1.0rem;
}


/* Header Logos */
.header-logo img, .header-icon img {
  max-height: 56px;
  width: auto;
  object-fit: contain;
  display: block;
  margin: 0 auto;
}
.header-title{
  display:flex; flex-direction:column; align-items:flex-start; justify-content:center; gap: 6px;
}

/* Badge */
.badge{
  display:inline-flex; align-items:center; gap:0.45rem; padding:0.25rem 0.65rem;
  border:1px solid rgba(37,99,235,0.25); background: rgba(37,99,235,0.08);
  border-radius: 999px; color: var(--text); font-size: 0.85rem; font-weight: 500;
}
.badge.success { background: rgba(16,185,129,0.1); border-color: rgba(16,185,129,0.3); color: #047857; }
.badge.warning { background: rgba(245,158,11,0.1); border-color: rgba(245,158,11,0.3); color: #B45309; }
.badge.error { background: rgba(239,68,68,0.1); border-color: rgba(239,68,68,0.3); color: #B91C1C; }

/* Status Classes (User Requested) */
/* ===== Tech Status Badges (User Requested) ===== */
.badge-ok{
  display:inline-flex;
  align-items:center;
  gap:6px;
  padding:3px 10px;
  border-radius:999px;
  font-size:12px;
  font-weight:600;
  color:#065F46;
  background:#ECFDF5;
  border:1px solid #A7F3D0;
}

.badge-warn{
  display:inline-flex;
  align-items:center;
  gap:6px;
  padding:3px 10px;
  border-radius:999px;
  font-size:12px;
  font-weight:600;
  color:#92400E;
  background:#FFF7ED;
  border:1px solid #FCD34D;
}

.badge-error{
  display:inline-flex;
  align-items:center;
  gap:6px;
  padding:3px 10px;
  border-radius:999px;
  font-size:12px;
  font-weight:600;
  color:#7C2D12;
  background:#FEF2F2;
  border:1px solid #FCA5A5;
}

.badge-off{
  display:inline-flex;
  align-items:center;
  gap:6px;
  padding:3px 10px;
  border-radius:999px;
  font-size:12px;
  font-weight:600;
  color:#374151;
  background:#F3F4F6;
  border:1px solid #E5E7EB;
}

.badge-info{
  display:inline-flex;
  align-items:center;
  gap:6px;
  padding:3px 10px;
  border-radius:999px;
  font-size:12px;
  font-weight:600;
  color:#1E40AF;
  background:#EFF6FF;
  border:1px solid #BFDBFE;
}
.badge-domain { background:#EFF6FF; color:#1E40AF; }
.badge-tox    { background:#ECFDF5; color:#065F46; }
.badge-bio    { background:#F3F4F6; color:#374151; }

/* ===== Result Card Design ===== */
.result-card{
  border: 1px solid rgba(37,99,235,0.18);
  background: linear-gradient(180deg, #FFFFFF, #FBFCFE);
  border-radius: 16px;
  padding: 18px 18px;
  box-shadow: 0 10px 24px rgba(17,24,39,0.06);
  margin-bottom: 20px;
}

.result-top{
  display:flex;
  justify-content:space-between;
  align-items:center;
  gap:12px;
}

.result-kicker{
  font-size: 12px;
  color: #6B7280;
  letter-spacing: 0.02em;
  margin: 0;
}

.result-value{
  font-size: 26px;
  font-weight: 850;
  letter-spacing: -0.02em;
  margin: 4px 0 0 0;
  color: #111827;
  line-height: 1.2;
}

.pill{
  display:inline-flex;
  align-items:center;
  gap:6px;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 650;
  border: 1px solid rgba(17,24,39,0.10);
  background: #F3F4F6;
  color: #374151;
  white-space: nowrap;
}

.pill-warn{
  background: #FFF7ED;
  border-color: #FCD34D;
  color: #92400E;
}

.pill-ok{
  background: #ECFDF5;
  border-color: #A7F3D0;
  color: #065F46;
}

.result-icon{
  width: 42px;
  height: 42px;
  border-radius: 12px;
  display:flex;
  align-items:center;
  justify-content:center;
  background: rgba(37,99,235,0.10);
  border: 1px solid rgba(37,99,235,0.20);
  font-size: 18px;
  flex-shrink: 0;
}


/* Buttons */
.stButton > button{
  border-radius: 999px; border: 1px solid rgba(37,99,235,0.35);
  background: var(--accentSoft); color: var(--text); padding: 0.55rem 1.2rem;
  font-weight: 600; width: 100%; transition: all .15s ease;
}
.stButton > button:hover{
  transform: translateY(-1px); background: rgba(37,99,235,0.16);
}

.muted{ color: var(--muted); font-size: 0.9rem; }
.hero-title{
  font-weight: 800;
  letter-spacing: -0.025em;
}
.hero-sub{
  font-size: 14px;
  color: #6B7280;
}
hr{ border: none; height: 1px; background: linear-gradient(90deg, transparent, var(--line), transparent); margin: 1.5rem 0; }
</style>
"""
st.markdown(LIGHT_SCI_CSS, unsafe_allow_html=True)

# ==========================================
# 3. LÓGICA DE NEGOCIO
# ==========================================
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
@st.cache_resource
def get_predictor():
    return CascadePredictor(MODELS_DIR)

predictor = get_predictor()

# ==========================================
# 4. HEADER (Logos)
# ==========================================
UCM_LOGO = os.path.join("assets", "logo_ucm.png")
BIO_ICON = os.path.join("assets", "122402012.png")

st.markdown('<div class="card">', unsafe_allow_html=True)
cL, cC, cR = st.columns([0.22, 1.0, 0.22], vertical_alignment="center")

with cL:
    if os.path.exists(UCM_LOGO):
        st.markdown('<div class="header-logo">', unsafe_allow_html=True)
        st.image(UCM_LOGO, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
with cC:
    st.markdown("""
      <div class="header-title">
        <div class="badge">🧬 QSAR & ADMET Pipeline</div>
        <h1 class="hero-title" style="margin:0;">Bioinformatics and Computational Chemistry Laboratory</h1>
        <p class="hero-sub" style="margin:0;">Molecular Modeling · Docking · Molecular Dynamics · Structural Analysis</p>
      </div>
    """, unsafe_allow_html=True)
with cR:
    pass

st.markdown("""
<p class="muted" style="margin-top:6px; font-size:13px;">
Research software · Academic use · Universidad Católica del Maule
</p>
""", unsafe_allow_html=True)

st.markdown("""
<div style="height:1px; background:linear-gradient(90deg,#E5E7EB,#CBD5E1,#E5E7EB);
margin-top:16px;"></div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# 5. MAIN GRID
# ==========================================
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

left, right = st.columns([1.35, 1.0], gap="large")

# --- COLUMNA DERECHA (Estado + Input + Vis) ---
with right:
    # 1. System Status
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧠 System Status")
    m1, m2, m3 = st.columns(3)
    m1.metric("🧩 Models", "3")
    m2.metric("🧬 FP Bits", "2048")
    m3.metric("📐 Validation", "MCC")
    st.markdown('</div>', unsafe_allow_html=True)

    # 2. Input
    st.markdown('<div class="card">', unsafe_allow_html=True)
    # st.subheader("🧬 Entrada Molecular")
    
    # User requested Input Style
    smiles_input = st.text_area("SMILES", height=90, placeholder="Ex: CC(=O)Oc1ccccc1C(=O)O")
    
    if smiles_input:
        st.caption(f"{len(smiles_input)} characters")
        if len(smiles_input) < 10:
            st.warning("SMILES too short — check structure")
        else:
            st.success("Valid SMILES for analysis")

    analyze_btn = st.button("🚀 Run Analysis")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 3. Visualization
    if analyze_btn and smiles_input:
        with st.spinner("Processing..."):
            st.session_state.prediction_result = predictor.predict(smiles_input)
            
    if st.session_state.prediction_result and st.session_state.prediction_result.get("ok"):
        res = st.session_state.prediction_result
        mol = Chem.MolFromSmiles(res["smiles_canonical"])
        if mol:
            # Generate Base64 Image
            img = Draw.MolToImage(mol, size=(450, 350))
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode()

            st.markdown(f"""
            <div class="card">
              <p class="section-label">2D Visualization</p>
              <img src="data:image/png;base64,{img_b64}" style="width:100%; border-radius:12px;">
              
              <div style="height:1px; background:linear-gradient(90deg,#E5E7EB,#2563EB,#E5E7EB); margin:10px 0 14px 0;"></div>
              
              <p class="muted" style="font-size:12px; margin-top:8px;">
                2D molecular depiction generated from SMILES representation (RDKit)
              </p>
            </div>
            """, unsafe_allow_html=True)
    else:
        # Placeholder or previous state
        pass


# --- COLUMNA IZQUIERDA (Resultados) ---
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Pipeline Results")
    st.caption("Stage Summary")
    
    if st.session_state.prediction_result:
        res = st.session_state.prediction_result
        if not res.get("ok"):
            st.error(res.get("error"))
        else:
            # Info General
            final_lbl = res['final_label']
            
            # Map for display
            display_lbl = (final_lbl
                           .replace("CANDIDATO", "CANDIDATE")
                           .replace("ALERTA", "ALERT")
                           .replace("TOXICIDAD", "TOXICITY")
                           .replace("BIOACTIVIDAD", "BIOACTIVITY"))

            # Definitions
            icon = "🔎"
            pill_class = "pill-warn"
            pill_text = "⚠️ Review"
            
            if "CANDIDATE" in display_lbl:
                icon = "🧪"
                pill_class = "pill-ok"
                pill_text = "✅ Accepted"
            elif "ALERT" in display_lbl:
                icon = "⚠️"
                pill_class = "pill-warn"
                pill_text = "🚨 Toxic"

            # Template
            template = """
            <div class="result-card">
              <div class="result-top">
                <div style="display:flex; align-items:center; gap:12px;">
                  <div class="result-icon">__ICON__</div>
                  <div>
                    <p class="result-kicker">Final Compound Classification</p>
                    <p class="result-value">__LABEL__</p>
                  </div>
                </div>
                <span class="pill __PILLCLASS__">__PILL__</span>
              </div>
            </div>
            """

            html = (template
                    .replace("__ICON__", icon)
                    .replace("__LABEL__", display_lbl.replace("_", " "))
                    .replace("__PILLCLASS__", pill_class)
                    .replace("__PILL__", pill_text))

            st.markdown(html, unsafe_allow_html=True)
            
            # Tabla Detalle
            doa = res["doa"]
            s1 = res["stage1_dsstox_like"]
            s2 = res["stage2_clue_like"]
            s3 = res["stage3_uiref_like"]
            
            # Helpers for span classes
            def get_span(condition, true_text, false_text, true_cls="badge-ok", false_cls="badge-warn"):
                cls_ = true_cls if condition else false_cls
                txt = true_text if condition else false_text
                return f'<span class="{cls_}">{txt}</span>'

            # 1. DOA
            doa_span = get_span(doa['in_domain'], "✅ IN DOMAIN", "⚠️ OUT OF DOMAIN", "badge-ok", "badge-warn")
            
            # 2. DSSTOX
            dsstox_span = get_span(not s1['alert'], "✅ OK", "⚠️ ALERT", "badge-ok", "badge-warn")
            
            # 3. CLUE
            clue_span = get_span(s2['pass'], "✅ PASS", "⏸ FAIL", "badge-ok", "badge-off")

            # 4. UIREF
            uiref_span = get_span(s3['candidate'], "✅ CANDIDATE", "⏸ NOT CANDIDATE", "badge-ok", "badge-off")

            st.markdown(f"""
            <div style="margin-bottom: 12px; padding-bottom: 12px; border-bottom: 1px solid #eee;">
                <strong>1. Applicability Domain</strong><br>
                {doa_span}
                <div style="margin-top:4px;"><small class="muted">Mean Similarity: <code>{doa['kNN_mean_sim']:.3f}</code></small></div>
            </div>
            
            <div style="margin-bottom: 12px; padding-bottom: 12px; border-bottom: 1px solid #eee;">
                <strong>2. Toxicity (DSSTOX)</strong><br>
                {dsstox_span}
                <div style="margin-top:4px;"><small class="muted">Probability: <code>{s1['p']:.3f}</code></small></div>
            </div>
            
            <div style="margin-bottom: 12px; padding-bottom: 12px; border-bottom: 1px solid #eee;">
                <strong>3. Bioactivity (CLUE)</strong><br>
                {clue_span}
                <div style="margin-top:4px;"><small class="muted">Probability: <code>{s2['p']:.3f}</code></small></div>
            </div>

            <div>
                <strong>4. UIREF Selection</strong><br>
                {uiref_span}
                <div style="margin-top:4px;"><small class="muted">Probability: <code>{s3['p']:.3f}</code></small></div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.info("Waiting for analysis...")
        st.markdown('<div style="height:200px"></div>', unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

    # Notas
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📝 Lab Notes")
    obs_text = st.text_area("Observations", height=120, label_visibility="collapsed", placeholder="Enter laboratory observations or findings here...")
    
    # Logic to prepare the report content
    report_content = "=== QSAR & ADMET PIPELINE REPORT ===\n\n"
    if st.session_state.prediction_result:
        res = st.session_state.prediction_result
        report_content += f"SMILES: {res.get('smiles_input', 'N/A')}\n"
        report_content += f"Canonical: {res.get('smiles_canonical', 'N/A')}\n"
        report_content += f"Final Classification: {res.get('final_label', 'N/A')}\n\n"
        
        report_content += "--- Stage Details ---\n"
        report_content += f"1. Applicability Domain: {'IN DOMAIN' if res['doa']['in_domain'] else 'OUT OF DOMAIN'} (Sim: {res['doa']['kNN_mean_sim']:.3f})\n"
        report_content += f"2. Toxicity (DSSTOX): {'ALERT' if res['stage1_dsstox_like']['alert'] else 'OK'} (Prob: {res['stage1_dsstox_like']['p']:.3f})\n"
        report_content += f"3. Bioactivity (CLUE): {'PASS' if res['stage2_clue_like']['pass'] else 'FAIL'} (Prob: {res['stage2_clue_like']['p']:.3f})\n"
        report_content += f"4. UIREF Selection: {'CANDIDATE' if res['stage3_uiref_like']['candidate'] else 'NOT CANDIDATE'} (Prob: {res['stage3_uiref_like']['p']:.3f})\n\n"
    else:
        report_content += "No prediction data available.\n\n"
        
    report_content += "--- Laboratory Observations ---\n"
    report_content += obs_text if obs_text else "No observations recorded."
    report_content += "\n\n© 2026 - Bioinformatics and Computational Chemistry Laboratory (UCM)"

    # Download button
    st.download_button(
        label="📥 Download Laboratory Report (.txt)",
        data=report_content,
        file_name="molecular_report.txt",
        mime="text/plain",
        help="Export all results and notes to a text file for your records."
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()
st.markdown(
"""
<p class="muted" style="font-size:12px;">
© 2026 — University of Maule ·
Bioinformatics and Computational Chemistry Laboratory<br>
Academic and research use
</p>
""",
unsafe_allow_html=True
)
