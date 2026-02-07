import streamlit as st

# ----------------------------
# Config
# ----------------------------
st.set_page_config(
    page_title="Interfaz Bioinformática Estructural",
    page_icon="🧬",
    layout="wide",
)

# ----------------------------
# CSS (igual al anterior: claro y prolijo)
# ----------------------------
CSS = """
<style>
:root{
  --bg: #F6F8FB;
  --panel: #FFFFFF;
  --line: rgba(17, 24, 39, 0.14);
  --text: #111827;
  --muted: #6B7280;
  --radius: 16px;
  --shadow: 0 10px 30px rgba(17, 24, 39, 0.06);
}

.stApp{
  background: linear-gradient(180deg, var(--bg), #ffffff);
}

.block-container{
  padding-top: 1.4rem;
  padding-bottom: 2.0rem;
  max-width: 1200px;
}

/* Contenedor general */
.shell{
  border: 1px solid var(--line);
  border-radius: calc(var(--radius) + 6px);
  background: rgba(255,255,255,0.55);
  padding: 18px;
}

/* Card base */
.card{
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 18px 18px;
}

.title-card{
  padding: 22px 18px;
  text-align: center;
}

/* Tipografía */
.h1{
  margin: 0;
  font-size: 34px;
  font-weight: 750;
  letter-spacing: -0.02em;
  color: var(--text);
}
.sub{
  margin: 8px 0 0 0;
  color: var(--muted);
  font-size: 14px;
}

/* Labels internos */
.section-label{
  font-size: 14px;
  font-weight: 650;
  color: var(--text);
  margin: 0 0 10px 0;
  letter-spacing: 0.01em;
}
.helper{
  color: var(--muted);
  font-size: 13px;
  margin: 0 0 12px 0;
}

/* Bloques de contenido */
.box{
  border: 1px dashed rgba(17,24,39,0.22);
  border-radius: 14px;
  padding: 14px;
  background: rgba(249,250,251,0.60);
}

/* Inputs */
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea,
div[data-baseweb="select"] > div{
  border-radius: 12px !important;
  border: 1px solid rgba(17,24,39,0.14) !important;
}

.stButton > button{
  border-radius: 999px;
  padding: 0.55rem 0.95rem;
  border: 1px solid rgba(37,99,235,0.35);
  background: rgba(37,99,235,0.10);
}

hr{
  border: none;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(17,24,39,0.14), transparent);
  margin: 14px 0;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ----------------------------
# Layout (según tus 2 referencias)
# ----------------------------
st.markdown('<div class="shell">', unsafe_allow_html=True)

# Título
st.markdown(
    """
    <div class="card title-card">
      <p class="h1">TÍTULO</p>
      <p class="sub">Interfaz clara y prolija · Streamlit (puro) · Bioinformática Estructural</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# Abajo:
# IZQUIERDA: Toxicidad (arriba) + Información adicional (abajo)
# DERECHA: input "Molécula a escribir" (arriba) + Estructura grande (abajo)
# (La derecha ahora es más grande)
col_left, col_right = st.columns([1.0, 1.25], gap="large")

with col_left:
    # TOXICIDAD (arriba, izquierda)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Toxicidad</p>', unsafe_allow_html=True)
    st.markdown('<p class="helper">Resultados / predicción (ADMET, flags, puntajes).</p>', unsafe_allow_html=True)

    st.markdown('<div class="box">', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Riesgo", "Bajo")
    m2.metric("Score", "0.18")
    m3.metric("Confianza", "0.82")
    st.divider()
    st.selectbox("Modelo", ["ADMET básico", "Clasificador (QSAR)", "Regresión (LD50)"])
    st.button("Evaluar toxicidad", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")

    # INFORMACIÓN ADICIONAL (abajo, izquierda)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Información adicional</p>', unsafe_allow_html=True)
    st.markdown('<p class="helper">Notas del experimento, parámetros, metadata.</p>', unsafe_allow_html=True)

    st.markdown('<div class="box">', unsafe_allow_html=True)
    st.text_input("ID / Nombre del ligando", placeholder="Ej: LIG-023")
    st.text_input("Target / Proteína", placeholder="Ej: AChE (PDB: 1EVE)")
    st.text_area("Notas", placeholder="Observaciones, parámetros, versión del modelo, dataset, etc.", height=170)
    c1, c2, _ = st.columns([1, 1, 2])
    with c1:
        st.button("Guardar", use_container_width=True)
    with c2:
        st.button("Limpiar", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    # MOLECULA A ESCRIBIR (arriba derecha)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Molécula a escribir</p>', unsafe_allow_html=True)
    st.markdown('<p class="helper">Ingresa SMILES o sube un archivo.</p>', unsafe_allow_html=True)

    st.markdown('<div class="box">', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["SMILES", "Archivo"])
    with tab1:
        st.text_area("SMILES", placeholder="Ej: CC(=O)Oc1ccccc1C(=O)O", height=120)
        st.button("Procesar SMILES", use_container_width=True)
    with tab2:
        st.file_uploader("Subir archivo (SDF / MOL / PDB)", type=["sdf", "mol", "pdb"])
        st.button("Cargar estructura", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")

    # ESTRUCTURA (más grande)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Estructura</p>', unsafe_allow_html=True)
    st.markdown('<p class="helper">Área de visualización / render. (Placeholder sin dependencias)</p>', unsafe_allow_html=True)

    st.markdown('<div class="box">', unsafe_allow_html=True)
    # Placeholder grande (sin dependencias)
    st.markdown(
        """
        <div style="
          height: 520px;
          border-radius: 14px;
          border: 1px solid rgba(17,24,39,0.14);
          background: linear-gradient(180deg, rgba(255,255,255,0.85), rgba(249,250,251,0.85));
          display: flex;
          align-items: center;
          justify-content: center;
          color: rgba(17,24,39,0.55);
          font-weight: 650;
          letter-spacing: 0.02em;">
          ÁREA DE ESTRUCTURA (más grande)
        </div>
        """,
        unsafe_allow_html=True
    )
    st.caption("Si después quieres vista 3D real, se puede integrar (requiere librería extra).")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)  # cierre shell

