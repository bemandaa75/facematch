import streamlit as st
import streamlit.components.v1 as components
import os, tempfile, base64, json
import numpy as np
from PIL import Image
from io import BytesIO
from deepface import DeepFace
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title="FaceMatch", page_icon="🔍", layout="wide")

# ── Hide ALL Streamlit chrome + sidebar ──────────────────────
st.html("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{
    --fm-pink:#ec4f7f; --fm-pink-light:#fde7ee; --fm-bg:#f7f7fa; --fm-dark:#2b2d3a;
    --fm-muted:#8c8fa3; --fm-border:#ececf3; --fm-green:#2bb673; --fm-green-bg:#e8f8f0;
}
html, body, [class*="css"]{ font-family:'Inter','Segoe UI',sans-serif; }

/* Force light mode regardless of OS/browser dark mode */
html, body { background-color:#f7f7fa !important; color:#2b2d3a !important; color-scheme: light !important; }
[data-testid="stAppViewContainer"], [data-testid="stApp"], .main, section.main { background-color:#f7f7fa !important; }
* { color-scheme: light !important; }

#MainMenu,footer,header{visibility:hidden}
[data-testid="stSidebar"]{display:none}
[data-testid="collapsedControl"]{display:none}
.block-container{padding:1.5rem 2rem 3rem !important;max-width:1400px !important}

div[data-testid="column"]:nth-of-type(1) > div[data-testid="stVerticalBlockBorderWrapper"],
div[data-testid="column"]:nth-of-type(1) div[data-testid="stVerticalBlock"]{
    background:#fff;
}
div[data-testid="stVerticalBlockBorderWrapper"]{
    border:1px solid var(--fm-border) !important;
    border-radius:18px !important;
    background:#fff !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] > div{
    padding:1.5rem 1.5rem 1.75rem !important;
}

.fm-card-title{
    display:flex;align-items:center;gap:0.6rem;font-weight:700;color:var(--fm-dark);
    font-size:1.05rem;margin-bottom:1.1rem;font-family:'Poppins',sans-serif;
}
.fm-card-title i{color:var(--fm-pink);}
.fm-field-label{font-size:0.88rem;font-weight:600;color:var(--fm-dark);margin:0.9rem 0 0.35rem;}
.fm-field-label:first-of-type{margin-top:0;}

[data-testid="stFileUploaderDropzone"]{
    border:1.5px dashed var(--fm-border) !important;
    border-radius:14px !important;
    background:#fafafa !important;
    padding:0.85rem !important;
}
[data-testid="stFileUploaderDropzone"]:hover{
    border-color:var(--fm-pink) !important;
    background:var(--fm-pink-light) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] span{ font-size:0.82rem !important; }
[data-testid="stFileUploaderDropzoneInstructions"] svg{ width:1.4rem !important; height:1.4rem !important; }

div[data-testid="stRadio"] > div[role="radiogroup"]{
    display:flex; gap:0.6rem; flex-wrap:nowrap;
}
div[data-testid="stRadio"] label{
    flex:1;
    border:2px solid var(--fm-border) !important;
    border-radius:14px !important;
    padding:0.65rem 0.5rem !important;
    margin:0 !important;
    cursor:pointer;
    transition:all 0.2s;
    text-align:center;
    background:#fff;
}
div[data-testid="stRadio"] label:has(input:checked){
    border-color:var(--fm-pink) !important;
    background:var(--fm-pink-light) !important;
}
div[data-testid="stRadio"] label > div:first-child{ display:none !important; }
div[data-testid="stRadio"] label p{
    font-size:0.82rem !important; font-weight:700 !important; color:var(--fm-dark) !important;
    margin:0 !important;
}

div[data-testid="stButton"] > button{
    background:linear-gradient(135deg,var(--fm-pink),#f47b9d) !important;
    color:#fff !important;
    border:none !important;
    border-radius:12px !important;
    font-weight:700 !important;
    font-size:0.95rem !important;
    padding:0.7rem 1.2rem !important;
    width:100% !important;
    margin-top:0.5rem !important;
    font-family:'Poppins',sans-serif !important;
}
div[data-testid="stButton"] > button:hover{
    background:linear-gradient(135deg,#d43e6e,var(--fm-pink)) !important;
    box-shadow:0 4px 14px rgba(236,79,127,0.35) !important;
}
div[data-testid="stButton"] > button p{ color:#fff !important; font-weight:700 !important; }

iframe{ border:none !important; }
</style>""")



# ── Konfigurasi PCA ───────────────────────────────────────────
DATASET_FOLDER = "dataset"
N_COMPONENTS   = 50
THRESHOLD_PCA  = 0.60

# ── Load model PCA (cached) ───────────────────────────────────
@st.cache_resource(show_spinner="⏳ Melatih model PCA dari dataset LFW...")
def load_pca_model():
    if not os.path.exists(DATASET_FOLDER):
        return None, None, None, 0
    X, labels = [], []
    for nama in sorted(os.listdir(DATASET_FOLDER)):
        folder = os.path.join(DATASET_FOLDER, nama)
        if not os.path.isdir(folder): continue
        for file in sorted(os.listdir(folder)):
            if not file.lower().endswith((".jpg",".jpeg",".png")): continue
            try:
                r = DeepFace.represent(img_path=os.path.join(folder,file),
                    model_name="Facenet512", enforce_detection=False, detector_backend="opencv")
                if r:
                    X.append(np.array(r[0]["embedding"])); labels.append(nama)
            except: pass
    if len(X) < N_COMPONENTS: return None, None, None, 0
    X = np.array(X)
    me = np.mean(X, axis=0); Xc = X - me
    sc = StandardScaler(with_mean=False); Xcs = sc.fit_transform(Xc)
    k  = min(N_COMPONENTS, len(X)-1, 512)
    pca = PCA(n_components=k); pca.fit(Xcs)
    return pca, sc, me, len(X)

# ── Fungsi analisis ───────────────────────────────────────────
def analisis_pca(path1, path2, threshold=0.60):
    pca, sc, me, total = load_pca_model()
    if pca is None: return {"status":"error","pesan":"Model PCA tidak tersedia."}
    try:
        def emb(p):
            r = DeepFace.represent(img_path=p,model_name="Facenet512",
                enforce_detection=False,detector_backend="opencv")
            return np.array(r[0]["embedding"])
        e1=emb(path1).reshape(1,-1); e2=emb(path2).reshape(1,-1)
        z1=pca.transform(sc.transform(e1-me)).flatten()
        z2=pca.transform(sc.transform(e2-me)).flatten()
        dot=float(np.dot(z1,z2)); n1=float(np.linalg.norm(z1)); n2=float(np.linalg.norm(z2))
        sim=dot/(n1*n2) if n1>0 and n2>0 else 0.0
        ed =float(np.linalg.norm(z1-z2))
        if sim>=threshold:
            pct=round(min((sim+0.20)*100,99.0),2); kes="Kemungkinan besar orang yang sama (Terverifikasi via PCA/Eigenfaces)"
        elif sim>=threshold-0.20:
            pct=round(sim*100,2); kes="Kemungkinan orang yang sama (Kemiripan sedang via PCA)"
        else:
            pct=round(max(sim*100,0.0),2); kes="Kemungkinan bukan orang yang sama (PCA/Eigenfaces)"
        return {"status":"ok","dot_product":round(dot,4),"norma_a":round(n1,4),"norma_b":round(n2,4),
                "cosine_similarity":round(sim,4),"euclidean_distance":round(ed,4),
                "persentase":pct,"kesimpulan":kes,"n_components":pca.n_components_,"total_dataset":total}
    except Exception as e: return {"status":"error","pesan":str(e)}



def img_to_b64(path_or_bytes):
    if isinstance(path_or_bytes, str):
        with open(path_or_bytes,"rb") as f: data=f.read()
    else: data=path_or_bytes
    return base64.b64encode(data).decode()

# ── State ─────────────────────────────────────────────────────
for k,v in [("hasil",None),("metode",None),("img1_b64",None),("img2_b64",None),
            ("nama_kecil",""),("nama_dewasa",""),("pca_aktif",None)]:
    if k not in st.session_state: st.session_state[k] = v

# ── CSS untuk komponen HTML ────────────────────────────────────
CSS = """
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root{
  --fm-pink:#ec4f7f;--fm-pink-2:#f47b9d;--fm-pink-light:#fde7ee;--fm-pink-border:#fbd2e0;
  --fm-bg:#f4f5f9;--fm-dark:#1e2030;--fm-muted:#8c8fa3;--fm-border:#e8eaf0;
  --fm-green:#2bb673;--fm-green-bg:#e6f9f1;--fm-green-border:#b8edd4;
  --fm-blue:#4a6cf7;--fm-blue-bg:#eef0fe;
  --fm-shadow:0 2px 12px rgba(44,47,73,0.07);
  --fm-shadow-lg:0 8px 32px rgba(44,47,73,0.12);
}
html,body{background:var(--fm-bg) !important;font-family:'Inter',sans-serif;color:var(--fm-dark) !important;margin:0;padding:0;color-scheme:light !important;}
h1,h2,h3,h4,h5,h6{font-family:'Poppins',sans-serif;}

/* === CARDS === */
.fm-card{background:#fff;border:1px solid var(--fm-border);border-radius:20px;padding:1.6rem;box-shadow:var(--fm-shadow);}
.fm-card-title{display:flex;align-items:center;gap:0.6rem;font-weight:700;font-size:1.05rem;margin-bottom:1.4rem;color:var(--fm-dark);}
.fm-card-title .ti{width:34px;height:34px;border-radius:10px;background:var(--fm-pink-light);color:var(--fm-pink);display:flex;align-items:center;justify-content:center;font-size:0.95rem;flex-shrink:0;}

/* === PHOTOS === */
.photo-frame{border-radius:16px;overflow:hidden;border:2px solid var(--fm-border);box-shadow:var(--fm-shadow);aspect-ratio:3/4;display:flex;align-items:center;justify-content:center;background:#f9f9fc;}
.photo-frame img{width:100%;height:100%;object-fit:cover;}
.photo-label{font-size:0.78rem;font-weight:600;color:var(--fm-muted);margin-top:0.6rem;text-transform:uppercase;letter-spacing:0.05em;}

/* === GAUGE === */
.gauge-wrap{position:relative;width:160px;height:160px;margin:0 auto;}
.gauge-wrap svg{transform:rotate(-90deg);}
.gauge-bg{fill:none;stroke:var(--fm-border);stroke-width:13;}
.gauge-fg{fill:none;stroke:var(--fm-pink);stroke-width:13;stroke-linecap:round;}
.gauge-center{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;}
.gauge-pct{font-size:1.5rem;font-weight:800;font-family:'Poppins',sans-serif;color:var(--fm-dark);line-height:1;}
.gauge-lbl{font-size:0.7rem;color:var(--fm-muted);font-weight:600;letter-spacing:0.06em;text-transform:uppercase;margin-top:0.2rem;}

/* === RESULT BADGE === */
.result-badge{display:inline-flex;align-items:center;gap:0.5rem;padding:0.6rem 1.1rem;border-radius:30px;font-weight:700;font-size:0.85rem;line-height:1.3;text-align:center;}
.result-badge.match{background:var(--fm-green-bg);color:var(--fm-green);border:1px solid var(--fm-green-border);}
.result-badge.nomatch{background:var(--fm-pink-light);color:var(--fm-pink);border:1px solid var(--fm-pink-border);}

/* === VECTOR BOX === */
.vector-box{background:linear-gradient(135deg,#fff5f8 0%,#fde7ee 100%);border:1px solid var(--fm-pink-border);border-radius:16px;padding:1rem 1.2rem;margin-top:1.1rem;}
.vector-row{display:flex;justify-content:space-between;align-items:center;padding:0.42rem 0;border-bottom:1px solid rgba(236,79,127,0.1);font-size:0.88rem;}
.vector-row:last-child{border-bottom:none;}
.vector-row .lbl{color:var(--fm-muted);font-weight:500;}
.vector-row .val{font-weight:700;font-family:'Poppins',sans-serif;color:var(--fm-dark);}
.vector-row.hl .lbl{color:var(--fm-pink);font-weight:600;}
.vector-row.hl .val{color:var(--fm-pink);font-size:1.05rem;}

/* === METHOD BADGE === */
.method-chip{display:inline-flex;align-items:center;gap:0.4rem;padding:0.3rem 0.75rem;border-radius:20px;font-size:0.75rem;font-weight:700;margin-bottom:1rem;}
.method-chip.pca{background:var(--fm-green-bg);color:var(--fm-green);border:1px solid var(--fm-green-border);}
.method-chip.arcface{background:var(--fm-blue-bg);color:var(--fm-blue);border:1px solid #c5cef7;}

/* === META PILLS === */
.meta-row{display:flex;gap:0.6rem;margin-top:1rem;flex-wrap:wrap;}
.meta-pill{display:flex;align-items:center;gap:0.5rem;background:var(--fm-bg);border:1px solid var(--fm-border);border-radius:12px;padding:0.5rem 0.8rem;flex:1;min-width:140px;}
.meta-pill .mic{width:28px;height:28px;border-radius:8px;background:var(--fm-pink-light);color:var(--fm-pink);display:flex;align-items:center;justify-content:center;font-size:0.78rem;flex-shrink:0;}
.meta-pill .mv{font-weight:700;font-family:'Poppins',sans-serif;font-size:0.88rem;line-height:1.1;color:var(--fm-dark);}
.meta-pill .ml{font-size:0.72rem;color:var(--fm-muted);}

/* === PEMBUKTIAN === */
.pembuktian-btn{width:100%;display:flex;align-items:center;justify-content:space-between;background:#fff;border:1px solid var(--fm-border);border-radius:12px;padding:0.6rem 1rem;cursor:pointer;font-size:0.85rem;font-weight:600;color:var(--fm-dark);margin-top:1rem;transition:border-color 0.2s,background 0.2s;}
.pembuktian-btn:hover{border-color:var(--fm-pink);background:var(--fm-pink-light);}
.pembuktian-btn .chev{font-size:0.78rem;transition:transform 0.25s;color:var(--fm-muted);}
.pembuktian-box{display:none;margin-top:0.5rem;background:#f8f9fc;border:1px solid var(--fm-border);border-radius:12px;padding:1rem 1.2rem;font-size:0.82rem;font-family:'Courier New',monospace;line-height:1.9;color:var(--fm-dark);}

/* === EMPTY STATE === */
.empty-wrap{display:flex;flex-direction:column;align-items:center;justify-content:center;padding:3rem 1rem;text-align:center;}
.empty-icon{width:80px;height:80px;border-radius:50%;background:var(--fm-bg);border:2px dashed var(--fm-border);display:flex;align-items:center;justify-content:center;margin:0 auto 1.2rem;font-size:2rem;color:#c8cad8;}
.empty-title{font-weight:700;font-size:1rem;color:var(--fm-dark);margin-bottom:0.3rem;}
.empty-sub{font-size:0.85rem;color:var(--fm-muted);}

/* === TABLE === */
.fm-table thead th{font-size:0.75rem;text-transform:uppercase;letter-spacing:0.05em;color:var(--fm-muted);font-weight:700;border-bottom:2px solid var(--fm-border);padding:0.6rem 0.8rem;background:transparent;}
.fm-table tbody td{font-size:0.88rem;vertical-align:middle;border-bottom:1px solid var(--fm-border);padding:0.7rem 0.8rem;}
.fm-table tbody tr:last-child td{border-bottom:none;}
.fm-table tbody tr:hover td{background:#fafbff;}
.badge-status{font-size:0.76rem;font-weight:700;padding:0.28rem 0.65rem;border-radius:8px;}
.badge-status.same{background:var(--fm-green-bg);color:var(--fm-green);}
.badge-status.diff{background:var(--fm-pink-light);color:var(--fm-pink);}
.badge-metode{font-size:0.74rem;font-weight:700;padding:0.22rem 0.6rem;border-radius:8px;}
.badge-metode.pca{background:var(--fm-green-bg);color:var(--fm-green);}
.badge-metode.arcface{background:var(--fm-blue-bg);color:var(--fm-blue);}

/* === CHART === */
.chart-wrap{position:relative;height:220px;}
.chart-stats{display:flex;gap:0.65rem;margin-top:1rem;flex-wrap:wrap;}
.chart-stat{flex:1;min-width:120px;background:var(--fm-bg);border:1px solid var(--fm-border);border-radius:14px;padding:0.8rem 1rem;text-align:center;}
.chart-stat .sv{font-family:'Poppins',sans-serif;font-weight:800;font-size:1.3rem;color:var(--fm-pink);}
.chart-stat .sl{color:var(--fm-muted);font-size:0.74rem;font-weight:600;text-transform:uppercase;letter-spacing:0.03em;margin-top:0.1rem;}

/* === CONCEPT PILLS === */
.concept-pill{display:inline-flex;align-items:center;gap:0.4rem;font-size:0.82rem;color:var(--fm-dark);font-weight:500;background:var(--fm-bg);border:1px solid var(--fm-border);border-radius:30px;padding:0.35rem 0.85rem;margin:0.18rem;}
.concept-pill i{color:var(--fm-green);font-size:0.75rem;}

/* === HISTORY SECTION === */
.section-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:1.2rem;flex-wrap:wrap;gap:0.5rem;}
.history-chip{background:var(--fm-pink-light);color:var(--fm-pink);font-weight:700;border-radius:30px;padding:0.28rem 0.85rem;font-size:0.82rem;}
.btn-hapus{background:#fff;border:1px solid var(--fm-border);color:var(--fm-muted);font-weight:600;font-size:0.82rem;border-radius:10px;padding:0.38rem 0.85rem;cursor:pointer;display:flex;align-items:center;gap:0.4rem;}
.btn-hapus:hover{border-color:#e74c3c;color:#e74c3c;}

/* === INFO CARDS === */
.info-card{background:#fff;border:1px solid var(--fm-border);border-radius:16px;padding:1.4rem;}
.info-card h6{font-weight:700;display:flex;align-items:center;gap:0.5rem;margin-bottom:1rem;font-size:0.95rem;}
.info-icon{width:34px;height:34px;border-radius:10px;background:var(--fm-pink-light);color:var(--fm-pink);display:flex;align-items:center;justify-content:center;font-size:0.9rem;flex-shrink:0;}
.info-note{border-radius:12px;padding:0.7rem 0.9rem;font-size:0.82rem;line-height:1.6;margin-bottom:0.65rem;}
.info-note:last-child{margin-bottom:0;}
.info-note.pca-note{background:var(--fm-green-bg);border:1px solid var(--fm-green-border);}
.info-note.arc-note{background:var(--fm-blue-bg);border:1px solid #c5cef7;}
.note-title{font-weight:700;font-size:0.8rem;margin-bottom:0.2rem;}
.note-title.pca{color:var(--fm-green);}
.note-title.arc{color:var(--fm-blue);}

@keyframes spin{to{transform:rotate(360deg)}}
</style>
"""

# ── Build Hasil HTML ───────────────────────────────────────────
hasil   = st.session_state.hasil
metode  = st.session_state.metode
img1_b64= st.session_state.img1_b64
img2_b64= st.session_state.img2_b64

if hasil and hasil.get("status") == "ok":
    d = hasil
    pct = d["persentase"]
    is_match = "besar" in d["kesimpulan"] or "sedang" in d["kesimpulan"]
    badge_cls = "match" if is_match else "nomatch"
    badge_icon = "fa-circle-check" if is_match else "fa-circle-xmark"
    circ = 427.26
    offset = circ - (circ * (pct / 100))
    metode_badge = '<span class="method-badge pca"><i class="fa-solid fa-layer-group me-1"></i>Metode: PCA / Eigenfaces</span>'
    eucl_row = f'<div class="vector-row"><span class="lbl">Euclidean Distance</span><span class="val">{d.get("euclidean_distance","—")}</span></div>'
    sim = d["cosine_similarity"]
    thr = st.session_state.get("threshold", 0.60)
    if sim >= thr: sp = f'<span style="color:var(--fm-green);font-weight:700;">Mirip ✅</span>'
    elif sim >= thr-0.20: sp = f'<span style="color:var(--fm-amber);font-weight:700;">Kemungkinan Mirip ⚠️</span>'
    else: sp = f'<span style="color:#e74c3c;font-weight:700;">Tidak Mirip ❌</span>'
    pembuktian_html = f"""
    Matriks X &nbsp;&nbsp;: {d.get('total_dataset','—')} × 512<br>
    Komponen &nbsp;: {d.get('n_components','—')} eigenfaces (SVD)<br>
    Xc = X − X̄ &nbsp;(centering mean embedding)<br>
    Z = Xc × Vk &nbsp;(proyeksi ke ruang PCA)<br>
    Cos Sim &nbsp;&nbsp;&nbsp;: {sim}<br>
    Threshold &nbsp;: {thr} → {sp}"""
    meta_html = f"""
    <div class="meta-pill"><span class="mic"><i class="fa-solid fa-vector-square"></i></span>
    <div><div class="mv">{d.get('n_components','—')} Komponen</div><div class="ml">PCA Eigenfaces</div></div></div>
    <div class="meta-pill"><span class="mic"><i class="fa-solid fa-database"></i></span>
    <div><div class="mv">{d.get('total_dataset','—')}</div><div class="ml">Dataset Latih</div></div></div>
    <div class="meta-pill"><span class="mic"><i class="fa-solid fa-sliders"></i></span>
    <div><div class="mv">{thr}</div><div class="ml">Threshold</div></div></div>"""

    hasil_html = f"""
    <div class="method-chip {'pca' if metode=='pca' else 'arcface'}">
      <i class="fa-solid {'fa-layer-group' if metode=='pca' else 'fa-robot'}"></i>
      Metode: {'PCA / Eigenfaces' if metode=='pca' else 'ArcFace (Deep Learning)'}
    </div>

    <div class="row g-3 align-items-stretch mb-0">
      <div class="col-4 text-center d-flex flex-column align-items-center">
        <div class="photo-frame w-100"><img src="data:image/jpeg;base64,{img1_b64}" alt="Kecil"></div>
        <div class="photo-label">Foto Masa Kecil</div>
      </div>
      <div class="col-4 d-flex flex-column align-items-center justify-content-center gap-3">
        <div class="gauge-wrap">
          <svg viewBox="0 0 160 160">
            <circle class="gauge-bg" cx="80" cy="80" r="68"></circle>
            <circle id="gaugeFg" class="gauge-fg" cx="80" cy="80" r="68"
              stroke-dasharray="427.26" stroke-dashoffset="427.26"></circle>
          </svg>
          <div class="gauge-center">
            <div class="gauge-pct" id="pctCounter" data-target="{pct}">0%</div>
            <div class="gauge-lbl">Kemiripan</div>
          </div>
        </div>
        <div class="result-badge {badge_cls} text-center" style="max-width:180px;">
          <i class="fa-solid {badge_icon} flex-shrink-0"></i>
          <span>{d['kesimpulan']}</span>
        </div>
      </div>
      <div class="col-4 text-center d-flex flex-column align-items-center">
        <div class="photo-frame w-100"><img src="data:image/jpeg;base64,{img2_b64}" alt="Dewasa"></div>
        <div class="photo-label">Foto Masa Dewasa</div>
      </div>
    </div>

    <div class="vector-box">
      <div class="vector-row"><span class="lbl">Dot Product (z₁ · z₂)</span><span class="val">{d['dot_product']}</span></div>
      <div class="vector-row"><span class="lbl">Norma Vektor A (‖z₁‖)</span><span class="val">{d['norma_a']}</span></div>
      <div class="vector-row"><span class="lbl">Norma Vektor B (‖z₂‖)</span><span class="val">{d['norma_b']}</span></div>
      <div class="vector-row hl"><span class="lbl">Cosine Similarity</span><span class="val">{d['cosine_similarity']}</span></div>
      {eucl_row}
    </div>

    <button class="pembuktian-btn" onclick="togglePembuktian(this)">
      <span><i class="fa-solid fa-flask me-2" style="color:var(--fm-pink);"></i>{"Lihat Pembuktian PCA/Eigenfaces" if metode=="pca" else "Lihat Pembuktian ArcFace"}</span>
      <i class="fa-solid fa-chevron-down chev"></i>
    </button>
    <div id="boxPembuktian" class="pembuktian-box">
      {pembuktian_html}
    </div>

    <div class="meta-row">{meta_html}</div>
    """
else:
    hasil_html = """
    <div class="empty-wrap">
      <div class="empty-icon"><i class="fa-solid fa-face-viewfinder"></i></div>
      <div class="empty-title">Belum ada foto yang dianalisis</div>
      <div class="empty-sub">Upload foto masa kecil &amp; masa dewasa,<br>lalu klik <b>Analisis Kemiripan</b>.</div>
    </div>"""

error_html = ""
if hasil and hasil.get("status") == "error":
    error_html = f'<div class="alert alert-danger mt-3 rounded-3 d-flex align-items-center" role="alert"><i class="fa-solid fa-circle-exclamation me-2 fs-5"></i><div>{hasil.get("pesan","Terjadi kesalahan.")}</div></div>'

_hasil_json = json.dumps({
    "status": hasil.get("status") if hasil else None,
    "persentase": hasil.get("persentase") if hasil else None,
    "kesimpulan": hasil.get("kesimpulan","") if hasil else "",
    "metode": metode
})
_nama_kecil  = st.session_state.get("nama_kecil","")
_nama_dewasa = st.session_state.get("nama_dewasa","")

# ── Cek PCA ───────────────────────────────────────────────────
if st.session_state.get("pca_aktif") is None:
    pca,sc,me,tot = load_pca_model()
    st.session_state.pca_aktif = pca is not None

# ── Logo ─────────────────────────────────────────────────────
def _get_logo():
    p = os.path.join("static","logo.png")
    if os.path.exists(p):
        with open(p,"rb") as f: return base64.b64encode(f.read()).decode()
    return None
_logo_b64 = _get_logo()
if _logo_b64:
    _logo_html = f'<img src="data:image/png;base64,{_logo_b64}" style="width:56px;height:56px;border-radius:16px;object-fit:cover;box-shadow:0 8px 20px rgba(236,79,127,0.25);">'
else:
    _logo_html = '<div style="width:56px;height:56px;border-radius:16px;background:linear-gradient(135deg,#ec4f7f,#f47b9d);display:flex;align-items:center;justify-content:center;color:#fff;font-size:1.6rem;box-shadow:0 8px 20px rgba(236,79,127,0.25);"><i class="fa-solid fa-face-viewfinder"></i></div>'

# ── Header ─────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;padding:0.5rem 0 1.75rem;font-family:'Inter',sans-serif;">
    <div style="display:inline-flex;align-items:center;justify-content:center;gap:0.85rem;">
        {_logo_html}
        <h1 style="font-family:'Poppins',sans-serif;font-weight:800;font-size:2.5rem;letter-spacing:-0.02em;margin:0;color:#2b2d3a;">FaceMatch</h1>
    </div>
    <p style="color:#8c8fa3;font-size:1rem;margin-top:0.35rem;">Deteksi Kemiripan Wajah Masa Kecil dan Dewasa Berdasarkan Prinsip Aljabar Linear</p>
    <div style="width:90px;height:3px;border-radius:3px;background:linear-gradient(90deg,transparent,#ec4f7f,transparent);margin:1rem auto 0;"></div>
</div>
""", unsafe_allow_html=True)

# ── Layout: kiri (upload) | kanan (hasil) ─────────────────────
col_left, col_right = st.columns([5, 7], gap="medium")

with col_left:
    with st.container(border=True):
        st.markdown('<div class="fm-card-title"><i class="fa fa-cloud-arrow-up"></i> Upload Foto</div>', unsafe_allow_html=True)

        st.markdown('<div class="fm-field-label">👶 Foto Masa Kecil</div>', unsafe_allow_html=True)
        foto_kecil = st.file_uploader("foto_kecil", type=["jpg","jpeg","png","jfif","webp"], label_visibility="collapsed", key="up_kecil")

        st.markdown('<div class="fm-field-label">🧑 Foto Masa Dewasa</div>', unsafe_allow_html=True)
        foto_dewasa = st.file_uploader("foto_dewasa", type=["jpg","jpeg","png","jfif","webp"], label_visibility="collapsed", key="up_dewasa")

        st.markdown('<div class="fm-field-label">⚙️ Threshold Kemiripan PCA</div>', unsafe_allow_html=True)
        threshold = st.slider("threshold", min_value=0.30, max_value=0.90, value=0.60, step=0.05, label_visibility="collapsed")
        st.caption(f"Default: 0.60 | Saat ini: {threshold:.2f} | Makin tinggi = makin ketat")

        analisis_btn = st.button("🔍 Analisis Kemiripan", use_container_width=True)

        if not st.session_state.pca_aktif:
            st.warning("⚠️ Dataset PCA tidak ditemukan")

        if error_html:
            st.markdown(error_html, unsafe_allow_html=True)

# ── Proses analisis ───────────────────────────────────────────
if analisis_btn and foto_kecil and foto_dewasa:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f1:
        f1.write(foto_kecil.read()); p1=f1.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f2:
        f2.write(foto_dewasa.read()); p2=f2.name
    with col_right:
        with st.spinner("⏳ Menganalisis kemiripan wajah dengan PCA/Eigenfaces..."):
            h = analisis_pca(p1, p2, threshold); m = "pca"
    st.session_state.hasil = h
    st.session_state.metode = m
    st.session_state.threshold = threshold
    st.session_state.img1_b64 = img_to_b64(p1)
    st.session_state.img2_b64 = img_to_b64(p2)
    st.session_state.nama_kecil = foto_kecil.name
    st.session_state.nama_dewasa = foto_dewasa.name
    os.unlink(p1); os.unlink(p2)
    st.rerun()
elif analisis_btn:
    st.warning("⚠️ Upload kedua foto dulu!")

# ── HTML panel kanan: Hasil Analisis ──────────────────────────
with col_right:
    with st.container(border=True):
        st.markdown('<div class="fm-card-title"><i class="fa-solid fa-square-poll-horizontal"></i> Hasil Analisis</div>', unsafe_allow_html=True)
        PAGE_HTML = f"""
<!DOCTYPE html><html lang="id"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<meta name="color-scheme" content="light">
{CSS}
<style>body{{background:transparent !important;}} .main-card{{border:none !important;}}</style>
</head><body style="background:transparent;padding:0;margin:0;">
<div id="fm-resize-root">
    {hasil_html}
</div>
<script>
const pc=document.getElementById('pctCounter'),gf=document.getElementById('gaugeFg');
if(pc&&gf){{
  const tg=parseFloat(pc.dataset.target)||0,ci=427.26,dur=1400,st0=performance.now();
  function anim(now){{
    const el=now-st0,pr=Math.min(el/dur,1),ea=1-Math.pow(1-pr,3),cv=tg*ea;
    pc.textContent=cv.toFixed(2)+'%';gf.style.strokeDashoffset=ci-(ci*(cv/100));
    if(pr<1)requestAnimationFrame(anim);
    else{{pc.textContent=tg.toFixed(2)+'%';gf.style.strokeDashoffset=ci-(ci*(tg/100));}}
  }}
  requestAnimationFrame(anim);
}}
function fmReportHeight(){{
  const root=document.getElementById('fm-resize-root');
  if(!root)return;
  // Use max of getBoundingClientRect and scrollHeight for reliability
  const h=Math.max(root.getBoundingClientRect().height, root.scrollHeight)+48;
  window.parent.postMessage({{type:'streamlit:setFrameHeight',height:Math.ceil(h)}},'*');
}}
function togglePembuktian(btn){{
  const box=document.getElementById('boxPembuktian');
  const ic=btn.querySelector('.chev');
  if(!box)return;
  const opening=box.style.display==='none';
  box.style.display=opening?'block':'none';
  if(ic)ic.style.transform=opening?'rotate(180deg)':'rotate(0deg)';
  // Force multiple resize calls to ensure Streamlit catches it
  [50,150,300,600,1000].forEach(t=>setTimeout(fmReportHeight,t));
}}
document.addEventListener('DOMContentLoaded',()=>{{setTimeout(fmReportHeight,120);setTimeout(fmReportHeight,600);}});
window.addEventListener('load',fmReportHeight);
const _obs=new ResizeObserver(()=>setTimeout(fmReportHeight,50));
_obs.observe(document.getElementById('fm-resize-root')||document.body);
</script>
</body></html>"""
        _hasil_ada = hasil and hasil.get("status") == "ok"
        _h = 750 if _hasil_ada else 260
        components.html(PAGE_HTML, height=_h, scrolling=True)

# ── Section bawah: Riwayat & Info ─────────────────────────────
_hasil_json2 = json.dumps({
    "status": hasil.get("status") if hasil else None,
    "persentase": hasil.get("persentase") if hasil else None,
    "kesimpulan": hasil.get("kesimpulan","") if hasil else "",
    "metode": metode
})

BOTTOM_HTML = f"""
<!DOCTYPE html><html lang="id"><head><meta charset="UTF-8">
<meta name="color-scheme" content="light">
{CSS}
</head><body style="background:transparent;padding:0;margin:0;">
<div class="container-fluid px-0">

  <!-- RIWAYAT -->
  <div class="fm-card mb-4">
    <div class="section-head">
      <div class="fm-card-title mb-0">
        <span class="ti"><i class="fa-solid fa-clock-rotate-left"></i></span>Riwayat Analisis
      </div>
      <div class="d-flex align-items-center gap-2">
        <span class="history-chip" id="historyCount">0 Data Tersimpan</span>
        <button class="btn-hapus" id="btnClearHistory" style="display:none;" onclick="clearHistory()">
          <i class="fa-solid fa-trash-can"></i>Hapus
        </button>
      </div>
    </div>
    <div class="table-responsive">
      <table class="fm-table w-100">
        <thead><tr>
          <th>ID</th><th>File Kecil</th><th>File Dewasa</th>
          <th>Similarity</th><th>Metode</th><th>Status</th><th>Waktu</th>
        </tr></thead>
        <tbody id="historyTableBody">
          <tr><td colspan="7" class="text-center py-4" style="color:#c8cad8;">
            <i class="fa-solid fa-inbox d-block mb-2 fs-3"></i>Belum ada riwayat.
          </td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- CHART -->
  <div class="fm-card mb-4" id="chartContainer" style="display:none;">
    <div class="fm-card-title">
      <span class="ti"><i class="fa-solid fa-chart-line"></i></span>Tren Similarity dari Riwayat
    </div>
    <div class="chart-wrap"><canvas id="trendChart"></canvas></div>
    <div class="chart-stats">
      <div class="chart-stat"><div class="sv" id="statAvg">0%</div><div class="sl">Rata-rata</div></div>
      <div class="chart-stat"><div class="sv" id="statMax">0%</div><div class="sl">Tertinggi</div></div>
      <div class="chart-stat"><div class="sv" id="statSame">0</div><div class="sl">Hasil Sama</div></div>
      <div class="chart-stat"><div class="sv" id="statDiff">0</div><div class="sl">Tidak Mirip</div></div>
    </div>
  </div>

  <!-- INFO -->
  <div class="row g-4">
    <div class="col-lg-5">
      <div class="info-card h-100">
        <h6><span class="info-icon"><i class="fa-solid fa-lightbulb"></i></span>Konsep Aljabar Linear</h6>
        <div>
          <span class="concept-pill"><i class="fa-solid fa-circle-check"></i>Matriks (Pixel)</span>
          <span class="concept-pill"><i class="fa-solid fa-circle-check"></i>Vektor (Embedding)</span>
          <span class="concept-pill"><i class="fa-solid fa-circle-check"></i>Dot Product</span>
          <span class="concept-pill"><i class="fa-solid fa-circle-check"></i>Norma Vektor</span>
          <span class="concept-pill"><i class="fa-solid fa-circle-check"></i>Cosine Similarity</span>
          <span class="concept-pill"><i class="fa-solid fa-circle-check"></i>PCA / Eigenfaces</span>
          <span class="concept-pill"><i class="fa-solid fa-circle-check"></i>SVD (Xc=UΣVᵀ)</span>
          <span class="concept-pill"><i class="fa-solid fa-circle-check"></i>Reduksi Dimensi</span>
          <span class="concept-pill"><i class="fa-solid fa-circle-check"></i>Euclidean Distance</span>
        </div>
      </div>
    </div>
    <div class="col-lg-7">
      <div class="info-card h-100">
        <h6><span class="info-icon"><i class="fa-solid fa-circle-exclamation"></i></span>Catatan Penggunaan</h6>
        <div class="info-note pca-note">
          <div class="note-title pca"><i class="fa-solid fa-layer-group me-1"></i>PCA / Eigenfaces</div>
          <div style="color:#333;font-size:0.82rem;">Metode Aljabar Linear klasik berbasis reduksi dimensi (SVD). Mendeteksi kemiripan melalui proyeksi ke ruang eigenfaces.
          <br><span style="color:#c0392b;font-weight:600;">⚠ Sensitif</span> terhadap perubahan pencahayaan, posisi, dan ekspresi.</div>
        </div>
        <div class="info-note" style="background:#f8f9fc;border:1px solid var(--fm-border);">
          <div class="note-title" style="color:#6b7280;"><i class="fa-solid fa-sliders me-1"></i>Threshold PCA</div>
          <div style="color:#333;font-size:0.82rem;">Default: <strong>0.60</strong>. Nilai lebih tinggi = lebih ketat. Atur lewat slider di panel upload.</div>
        </div>
      </div>
    </div>
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bundle.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script>
let trendChartInstance=null;
function dapatkanRiwayat(){{const d=localStorage.getItem('facematch_history');return d?JSON.parse(d):[];}}
function simpanRiwayat(d){{localStorage.setItem('facematch_history',JSON.stringify(d));}}
function renderRiwayat(){{
  const r=dapatkanRiwayat(),tb=document.getElementById('historyTableBody'),
    hc=document.getElementById('historyCount'),bc=document.getElementById('btnClearHistory'),
    cc=document.getElementById('chartContainer');
  hc.textContent=r.length+' Data Tersimpan';
  if(r.length===0){{
    tb.innerHTML='<tr><td colspan="7" class="text-center py-4" style="color:#c8cad8;"><i class="fa-solid fa-inbox d-block mb-2 fs-3"></i>Belum ada riwayat.</td></tr>';
    bc.style.display='none';cc.style.display='none';bottomResize();return;
  }}
  bc.style.display='flex';cc.style.display='block';
  tb.innerHTML=r.map(i=>`<tr>
    <td><span style="color:#8c8fa3;font-weight:700;">#${{i.id}}</span></td>
    <td><small style="color:#8c8fa3;">${{i.foto_kecil}}</small></td>
    <td><small style="color:#8c8fa3;">${{i.foto_dewasa}}</small></td>
    <td><span style="font-weight:800;font-family:'Poppins',sans-serif;color:#ec4f7f;">${{i.persentase}}%</span></td>
    <td><span class="badge-metode ${{i.metode==='pca'?'pca':'arcface'}}">${{i.metode==='pca'?'PCA':'ArcFace'}}</span></td>
    <td><span class="badge-status ${{i.status==='Sama'?'same':'diff'}}">${{i.status}}</span></td>
    <td><small style="color:#c8cad8;">${{i.tanggal}}</small></td>
  </tr>`).join('');
  const dk=[...r].reverse(),lbl=dk.map(i=>'#'+i.id),val=dk.map(i=>parseFloat(i.persentase));
  const tc=document.getElementById('trendChart');
  if(tc){{
    if(trendChartInstance)trendChartInstance.destroy();
    trendChartInstance=new Chart(tc,{{type:'line',data:{{labels:lbl,datasets:[{{label:'Similarity (%)',data:val,
      borderColor:'#ec4f7f',backgroundColor:'rgba(236,79,127,0.08)',borderWidth:2.5,
      pointBackgroundColor:'#ec4f7f',pointRadius:4,tension:0.35,fill:true}}]}},
      options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}}}},
      scales:{{y:{{beginAtZero:true,max:100,ticks:{{callback:v=>v+'%'}},grid:{{color:'#e8eaf0'}}}},x:{{grid:{{display:false}}}}}}}}}});
  }}
  const tot=r.length,avg=(val.reduce((a,b)=>a+b,0)/tot).toFixed(2),mx=Math.max(...val).toFixed(2),sm=r.filter(i=>i.status==='Sama').length;
  document.getElementById('statAvg').textContent=avg+'%';
  document.getElementById('statMax').textContent=mx+'%';
  document.getElementById('statSame').textContent=sm;
  document.getElementById('statDiff').textContent=tot-sm;
  setTimeout(bottomResize,200);
}}
function clearHistory(){{
  if(confirm('Hapus seluruh riwayat?')){{localStorage.removeItem('facematch_history');renderRiwayat();}}
}}
const _hasil={_hasil_json2};
const _fk="{_nama_kecil}";const _fd="{_nama_dewasa}";
if(_hasil.status==='ok'&&_fk&&_fd){{
  const r=dapatkanRiwayat();
  const pct=parseFloat(_hasil.persentase),kes=_hasil.kesimpulan,mt=_hasil.metode;
  const st2=(kes.includes('besar')||kes.includes('sedang'))?'Sama':'Tidak Mirip';
  const nid=r.length>0?Math.max(...r.map(o=>o.id))+1:1;
  const now=new Date();
  const wkt=now.toLocaleDateString('id-ID',{{day:'numeric',month:'short'}})+' '+now.toLocaleTimeString('id-ID',{{hour:'2-digit',minute:'2-digit'}});
  const nb={{id:nid,foto_kecil:_fk,foto_dewasa:_fd,persentase:pct.toFixed(2),metode:mt,status:st2,tanggal:wkt}};
  const ada=r.some(i=>i.foto_kecil===nb.foto_kecil&&i.foto_dewasa===nb.foto_dewasa&&i.persentase===nb.persentase&&i.metode===nb.metode);
  if(!ada){{r.unshift(nb);simpanRiwayat(r);}}
}}
function bottomResize(){{
  window.parent.postMessage({{type:'streamlit:setFrameHeight',height:document.body.scrollHeight+40}},'*');
}}
document.addEventListener('DOMContentLoaded',()=>{{renderRiwayat();setTimeout(bottomResize,300);setTimeout(bottomResize,900);}});
window.addEventListener('load',bottomResize);
new ResizeObserver(()=>setTimeout(bottomResize,100)).observe(document.body);
</script>
</body></html>"""

components.html(BOTTOM_HTML, height=1100, scrolling=True)