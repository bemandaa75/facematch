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
st.set_page_config(page_title="FaceMatch", page_icon="🔍", layout="wide",
    initial_sidebar_state="collapsed")

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

/* ── FORCE WHITE / ANTI DARK MODE ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
.stApp,
.main,
section[data-testid="stSidebar"] ~ div,
[data-testid="block-container"] {
    background-color: #ffffff !important;
    color: #2b2d3a !important;
}
[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stVerticalBlock"],
div[class*="stMarkdown"],
div[class*="stText"],
p, span, label, div {
    color: #2b2d3a !important;
}
/* Override Streamlit dark theme variables */
:root, [data-theme="dark"], [data-theme="light"] {
    --background-color: #ffffff !important;
    --secondary-background-color: #f7f7fa !important;
    --text-color: #2b2d3a !important;
}

#MainMenu,footer,header{visibility:hidden}
[data-testid="stSidebar"]{display:none}
[data-testid="collapsedControl"]{display:none}
.block-container{padding:1rem 1rem 3rem !important;max-width:1400px !important}

/* Mobile responsive columns */
@media (max-width: 768px) {
    .block-container{padding:0.5rem 0.5rem 2rem !important;}
    [data-testid="column"] {
        min-width: 100% !important;
        width: 100% !important;
    }
    [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
        gap: 0.75rem !important;
    }
}

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
N_COMPONENTS   = 150
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
        gray1 = grayscale_grid_b64(path1)
        gray2 = grayscale_grid_b64(path2)
        fitur1 = fitur_pca_stats(z1)
        fitur2 = fitur_pca_stats(z2)
        if sim>=threshold:
            pct=round(min((sim+0.20)*100,99.0),2); kes="Kemungkinan besar orang yang sama (Terverifikasi via PCA/Eigenfaces)"
        elif sim>=threshold-0.20:
            pct=round(sim*100,2); kes="Kemungkinan orang yang sama (Kemiripan sedang via PCA)"
        else:
            pct=round(max(sim*100,0.0),2); kes="Kemungkinan bukan orang yang sama (PCA/Eigenfaces)"
        return {"status":"ok","dot_product":round(dot,4),"norma_a":round(n1,4),"norma_b":round(n2,4),
                "cosine_similarity":round(sim,4),"euclidean_distance":round(ed,4),
                "persentase":pct,"kesimpulan":kes,"n_components":pca.n_components_,"total_dataset":total,
                "gray1":gray1,"gray2":gray2,"fitur1":fitur1,"fitur2":fitur2}
    except Exception as e: return {"status":"error","pesan":str(e)}



def fitur_bar_block(judul, f):
    """Render blok bar horizontal sederhana untuk komposisi fitur PCA satu foto."""
    if not f: return ""
    items = [("PC Dominan (PC1)", f.get("pc1",0)), ("Variasi PC2", f.get("pc2",0)),
             ("Variasi PC3", f.get("pc3",0)), ("Energi Total (Σz²)", f.get("energi",0)),
             ("Kemiringan", f.get("skew",0))]
    maxval = max([abs(v) for _,v in items] + [1])
    rows = ""
    for label, val in items:
        widthpct = min(100, abs(val) / maxval * 100)
        rows += f"""
        <div style="margin-bottom:0.55rem;">
            <div style="display:flex;justify-content:space-between;font-size:0.78rem;color:var(--fm-muted);margin-bottom:0.2rem;">
                <span>{label}</span><span style="font-weight:700;color:var(--fm-dark);">{val}</span>
            </div>
            <div style="background:var(--fm-bg);border-radius:6px;height:8px;overflow:hidden;">
                <div style="background:var(--fm-pink);height:100%;width:{widthpct:.1f}%;border-radius:6px;"></div>
            </div>
        </div>"""
    return f"""<div style="border:1px solid var(--fm-border);border-radius:12px;padding:0.9rem 1rem;height:100%;">
    <div style="font-weight:700;font-size:0.85rem;color:var(--fm-dark);margin-bottom:0.7rem;">{judul}</div>
    {rows}</div>"""

def img_to_b64(path_or_bytes):
    if isinstance(path_or_bytes, str):
        with open(path_or_bytes,"rb") as f: data=f.read()
    else: data=path_or_bytes
    return base64.b64encode(data).decode()

def grayscale_grid_b64(path, size=100):
    """Konversi foto ke grayscale persegi (mirip ilustrasi matriks piksel input PCA)."""
    img = Image.open(path).convert("L").resize((size, size))
    buf = BytesIO(); img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def fitur_pca_stats(z):
    """Statistik ringkas dari vektor hasil proyeksi PCA (z), buat visualisasi 'komposisi fitur'."""
    z = np.asarray(z, dtype=float)
    energi = float(np.sum(z**2))
    pc1 = float(abs(z[0])) if len(z) > 0 else 0.0
    pc2 = float(abs(z[1])) if len(z) > 1 else 0.0
    pc3 = float(abs(z[2])) if len(z) > 2 else 0.0
    mean_z = float(np.mean(z)); std_z = float(np.std(z)) if np.std(z) > 0 else 1e-9
    skew = float(np.mean(((z - mean_z) / std_z) ** 3))
    return {"pc1": round(pc1,2), "pc2": round(pc2,2), "pc3": round(pc3,2),
            "energi": round(energi,2), "skew": round(skew,2)}

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
:root{--fm-pink:#ec4f7f;--fm-pink-light:#fde7ee;--fm-bg:#f7f7fa;--fm-dark:#2b2d3a;--fm-muted:#8c8fa3;--fm-border:#ececf3;--fm-green:#2bb673;--fm-green-bg:#e8f8f0;--fm-amber:#e0a13a;--fm-amber-bg:#fdf2e2;}
body{background-color:#ffffff;font-family:'Inter','Segoe UI',sans-serif;color:var(--fm-dark);margin:0;overflow-x:hidden;}
/* Mobile responsive inside iframe */
@media (max-width: 600px) {
    .row.g-3.align-items-center { gap: 0.5rem !important; }
    .col-4 { width: 33.33% !important; }
    .gauge-wrap { width: 120px !important; height: 120px !important; }
    .gauge-center .pct { font-size: 1rem !important; }
    .meta-info-row { flex-direction: column !important; }
    .meta-info-pill { min-width: unset !important; }
    .preview-img { max-height: 140px !important; }
    .result-badge { font-size: 0.78rem !important; padding: 0.5rem 0.75rem !important; }
    .chart-stats { gap: 0.5rem !important; }
    .chart-stat { min-width: 100px !important; }
    .col-lg-5, .col-lg-7 { width: 100% !important; }
    .table { font-size: 0.78rem !important; }
    th, td { padding: 0.4rem 0.3rem !important; }
}
h1,h2,h3,h4,h5,h6{font-family:'Poppins',sans-serif;}
.header-section{text-align:center;margin-bottom:2rem;}
.header-logo{display:inline-flex;align-items:center;justify-content:center;gap:0.85rem;}
.header-icon-badge{width:56px;height:56px;border-radius:16px;background:linear-gradient(135deg,var(--fm-pink) 0%,#f47b9d 100%);display:flex;align-items:center;justify-content:center;color:#fff;font-size:1.6rem;box-shadow:0 8px 20px rgba(236,79,127,0.25);}
.header-section h1{font-weight:800;font-size:2.5rem;letter-spacing:-0.02em;margin-bottom:0;color:var(--fm-dark);}
.header-section p{color:var(--fm-muted);font-size:1rem;margin-top:0.35rem;}
.header-divider{width:90px;height:3px;border-radius:3px;background:linear-gradient(90deg,transparent,var(--fm-pink),transparent);margin:1.25rem auto 0;}
.main-card{border:1px solid var(--fm-border);border-radius:18px;background:#ffffff;}
.card-title-row{display:flex;align-items:center;gap:0.6rem;font-weight:700;color:var(--fm-dark);font-size:1.05rem;margin-bottom:1.5rem;}
.card-title-row i{color:var(--fm-pink);}
.preview-img{max-height:200px;width:100%;object-fit:cover;border-radius:14px;box-shadow:0 4px 14px rgba(0,0,0,0.06);border:1px solid var(--fm-border);}
.preview-label{font-size:0.8rem;font-weight:600;color:var(--fm-muted);margin-top:0.5rem;}
.gauge-wrap{position:relative;width:170px;height:170px;margin:0 auto;}
.gauge-wrap svg{transform:rotate(-90deg);}
.gauge-bg{fill:none;stroke:var(--fm-border);stroke-width:14;}
.gauge-fg{fill:none;stroke:var(--fm-pink);stroke-width:14;stroke-linecap:round;transition:stroke-dashoffset 0.8s ease;}
.gauge-center{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;}
.gauge-center .pct{font-size:1.35rem;font-weight:800;font-family:'Poppins',sans-serif;color:var(--fm-dark);}
.gauge-center .pct-label{font-size:0.78rem;color:var(--fm-muted);font-weight:600;letter-spacing:0.04em;text-transform:uppercase;}
.result-badge{font-size:0.95rem;font-weight:700;padding:0.7rem 1.25rem;border-radius:30px;display:inline-flex;align-items:center;gap:0.5rem;}
.result-badge.match{background:var(--fm-green-bg);color:var(--fm-green);border:1px solid #c7ecdb;}
.result-badge.nomatch{background:var(--fm-pink-light);color:var(--fm-pink);border:1px solid #fbd2e0;}
.vector-box{border-radius:14px;padding:1.1rem 1.25rem;background:var(--fm-pink-light);border:1px solid #fbd2e0;}
.vector-row{display:flex;justify-content:space-between;align-items:center;padding:0.5rem 0;border-bottom:1px solid rgba(0,0,0,0.05);font-size:0.92rem;}
.vector-row:last-child{border-bottom:none;}
.vector-row .label{color:var(--fm-muted);font-weight:500;}
.vector-row .value{font-weight:700;color:var(--fm-dark);font-family:'Poppins',sans-serif;}
.vector-row.highlight .label,.vector-row.highlight .value{color:var(--fm-pink);font-size:1.05rem;}
.meta-info-row{display:flex;gap:0.75rem;margin-top:1rem;flex-wrap:wrap;}
.meta-info-pill{display:flex;align-items:center;gap:0.6rem;background:var(--fm-bg);border:1px solid var(--fm-border);border-radius:12px;padding:0.6rem 0.9rem;font-size:0.85rem;flex:1;min-width:160px;}
.meta-info-pill .icon-circle-sm{width:32px;height:32px;border-radius:9px;background:var(--fm-pink-light);color:var(--fm-pink);display:flex;align-items:center;justify-content:center;font-size:0.85rem;flex-shrink:0;}
.meta-info-pill .meta-value{font-weight:700;font-family:'Poppins',sans-serif;color:var(--fm-dark);font-size:0.95rem;line-height:1.1;}
.meta-info-pill .meta-label{color:var(--fm-muted);font-size:0.75rem;}
.empty-state{display:flex;flex-direction:column;align-items:center;justify-content:center;padding:3.5rem 1rem;color:var(--fm-muted);text-align:center;}
.empty-state i{font-size:2.6rem;margin-bottom:1rem;color:#d8dae6;}
.info-strip{border-radius:16px;border:1px solid var(--fm-border);background:#ffffff;padding:1.4rem 1.6rem;}
.info-strip h6{font-weight:700;display:flex;align-items:center;gap:0.5rem;margin-bottom:0.9rem;color:var(--fm-dark);}
.info-strip .icon-circle{width:36px;height:36px;border-radius:10px;background:var(--fm-pink-light);color:var(--fm-pink);display:flex;align-items:center;justify-content:center;font-size:0.95rem;}
.concept-pill{display:inline-flex;align-items:center;gap:0.45rem;font-size:0.85rem;color:var(--fm-dark);font-weight:500;background:var(--fm-bg);border:1px solid var(--fm-border);border-radius:30px;padding:0.4rem 0.9rem;margin:0.2rem;}
.concept-pill i{color:var(--fm-green);}
.method-badge{font-size:0.75rem;font-weight:700;padding:0.2rem 0.6rem;border-radius:20px;display:inline-block;}
.method-badge.arcface{background:#e8eeff;color:#4a6cf7;}
.method-badge.pca{background:#e8f8f0;color:var(--fm-green);}
.btn-clear{background:#fff;border:1px solid var(--fm-border);color:var(--fm-muted);font-weight:600;font-size:0.85rem;border-radius:10px;padding:0.45rem 1rem;cursor:pointer;}
.btn-clear:hover{border-color:var(--fm-pink);color:var(--fm-pink);}
.history-badge-count{background:var(--fm-pink-light);color:var(--fm-pink);font-weight:700;border-radius:30px;padding:0.35rem 1rem;font-size:0.85rem;}
.table thead th{font-size:0.78rem;text-transform:uppercase;letter-spacing:0.04em;color:var(--fm-muted);font-weight:700;border-bottom:1px solid var(--fm-border);background:transparent;}
.table tbody td{font-size:0.9rem;vertical-align:middle;border-bottom:1px solid var(--fm-border);}
.badge-status{font-size:0.78rem;font-weight:700;padding:0.35rem 0.75rem;border-radius:8px;}
.badge-status.same{background:var(--fm-green-bg);color:var(--fm-green);}
.badge-status.diff{background:var(--fm-pink-light);color:var(--fm-pink);}
.chart-wrap{position:relative;height:240px;}
.chart-stats{display:flex;gap:0.75rem;margin-top:1rem;flex-wrap:wrap;}
.chart-stat{flex:1;min-width:130px;background:var(--fm-bg);border:1px solid var(--fm-border);border-radius:12px;padding:0.75rem 1rem;text-align:center;}
.chart-stat .stat-value{font-family:'Poppins',sans-serif;font-weight:800;font-size:1.3rem;color:var(--fm-pink);}
.chart-stat .stat-label{color:var(--fm-muted);font-size:0.78rem;font-weight:600;text-transform:uppercase;letter-spacing:0.03em;}
/* Upload card */
.upload-section-label{font-size:0.88rem;font-weight:600;color:var(--fm-dark);margin-bottom:0.45rem;}
.upload-box{border:1.5px dashed var(--fm-border);border-radius:14px;padding:1.1rem 1rem 0.85rem;background:#fafafa;text-align:center;transition:border-color 0.2s,background 0.2s;cursor:pointer;}
.upload-box:hover,.upload-box.has-file{border-color:var(--fm-pink);background:var(--fm-pink-light);}
.upload-box .icon-circle-upload{width:48px;height:48px;border-radius:50%;background:var(--fm-pink-light);display:inline-flex;align-items:center;justify-content:center;margin-bottom:0.6rem;}
.upload-file-row{display:inline-flex;align-items:center;border-radius:7px;overflow:hidden;border:1px solid #d1d5db;font-size:0.8rem;background:#fff;margin-top:0.5rem;}
.upload-file-row .choose-btn{background:#f3f4f6;color:#374151;font-weight:600;padding:0.35rem 0.8rem;border-right:1px solid #d1d5db;cursor:pointer;white-space:nowrap;}
.upload-file-row .file-name{color:#6b7280;padding:0.35rem 0.7rem;max-width:150px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
/* Method select */
.method-cards{display:flex;gap:0.75rem;margin-bottom:1.1rem;}
.method-card{flex:1;border:2px solid var(--fm-border);border-radius:14px;padding:0.85rem 0.5rem;text-align:center;cursor:pointer;background:#fff;transition:border-color 0.2s,background 0.2s;user-select:none;}
.method-card.active{border-color:var(--fm-pink);background:var(--fm-pink-light);}
.method-card .mc-icon{font-size:1.3rem;margin-bottom:0.3rem;}
.method-card .mc-title{font-weight:700;font-size:0.82rem;color:var(--fm-dark);}
.method-card .mc-sub{font-size:0.72rem;color:var(--fm-muted);}
/* Analisis button */
.btn-analisis{width:100%;background:linear-gradient(135deg,#ec4f7f,#f47b9d);color:#fff;border:none;border-radius:12px;font-weight:700;font-size:0.95rem;padding:0.8rem 1.2rem;cursor:pointer;display:flex;align-items:center;justify-content:center;gap:0.5rem;transition:opacity 0.2s,box-shadow 0.2s;font-family:'Poppins',sans-serif;}
.btn-analisis:hover{opacity:0.92;box-shadow:0 4px 16px rgba(236,79,127,0.35);}
.btn-analisis:disabled{opacity:0.55;cursor:not-allowed;}
/* Loading overlay */
.loading-overlay{display:none;text-align:center;padding:2rem;color:var(--fm-pink);}
.spinner{width:40px;height:40px;border:4px solid var(--fm-pink-light);border-top-color:var(--fm-pink);border-radius:50%;animation:spin 0.8s linear infinite;margin:0 auto 1rem;}
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
    eucl_row = f'<div class="vector-row"><span class="label">Euclidean Distance</span><span class="value">{d.get("euclidean_distance","—")}</span></div>'
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
    <div class="meta-info-pill"><span class="icon-circle-sm"><i class="fa-solid fa-vector-square"></i></span>
    <div><div class="meta-value">{d.get('n_components','—')} Komponen</div><div class="meta-label">PCA Eigenfaces</div></div></div>
    <div class="meta-info-pill"><span class="icon-circle-sm"><i class="fa-solid fa-database"></i></span>
    <div><div class="meta-value">{d.get('total_dataset','—')}</div><div class="meta-label">Foto Dataset Latih</div></div></div>
    <div class="meta-info-pill"><span class="icon-circle-sm"><i class="fa-solid fa-sliders"></i></span>
    <div><div class="meta-value">{thr}</div><div class="meta-label">Threshold Digunakan</div></div></div>"""

    hasil_html = f"""
    {metode_badge}
    <div class="row g-3 align-items-center mb-4 mt-2">
        <div class="col-4 text-center">
            <img src="data:image/jpeg;base64,{img1_b64}" class="preview-img" alt="Kecil">
            <div class="preview-label">Foto Masa Kecil</div>
        </div>
        <div class="col-4 text-center">
            <div class="gauge-wrap">
                <svg viewBox="0 0 160 160">
                    <circle class="gauge-bg" cx="80" cy="80" r="68"></circle>
                    <circle id="gaugeFg" class="gauge-fg" cx="80" cy="80" r="68"
                        stroke-dasharray="427.26" stroke-dashoffset="427.26" data-target="{pct}"></circle>
                </svg>
                <div class="gauge-center">
                    <div class="pct" id="pctCounter" data-target="{pct}">0%</div>
                    <div class="pct-label">Kemiripan</div>
                </div>
            </div>
            <div class="mt-3">
                <div class="result-badge {badge_cls}">
                    <i class="fa-solid {badge_icon}"></i>{d['kesimpulan']}
                </div>
            </div>
        </div>
        <div class="col-4 text-center">
            <img src="data:image/jpeg;base64,{img2_b64}" class="preview-img" alt="Dewasa">
            <div class="preview-label">Foto Masa Dewasa</div>
        </div>
    </div>
    <div class="vector-box">
        <div class="vector-row"><span class="label">Dot Product (z₁ · z₂)</span><span class="value">{d['dot_product']}</span></div>
        <div class="vector-row"><span class="label">Norma Vektor A (‖z₁‖)</span><span class="value">{d['norma_a']}</span></div>
        <div class="vector-row"><span class="label">Norma Vektor B (‖z₂‖)</span><span class="value">{d['norma_b']}</span></div>
        <div class="vector-row highlight"><span class="label">Cosine Similarity</span><span class="value">{d['cosine_similarity']}</span></div>
        {eucl_row}
    </div>
    <div class="mt-3">
        <button class="btn-clear w-100 text-start" type="button" onclick="togglePembuktian(this)"
            style="display:flex;align-items:center;justify-content:space-between;">
            <span><i class="fa-solid fa-flask me-2" style="color:var(--fm-pink);"></i>
            {"Lihat Pembuktian PCA/Eigenfaces" if metode=="pca" else "Lihat Pembuktian ArcFace"}</span>
            <i class="fa-solid fa-chevron-down" style="font-size:0.8rem;transition:transform 0.25s;"></i>
        </button>
        <div id="boxPembuktian" style="display:none;margin-top:0.5rem;">
            <div class="p-3 rounded-3" style="background:#f7f7fa;border:1px solid var(--fm-border);font-size:0.83rem;font-family:monospace;line-height:2;">
                {pembuktian_html}
            </div>

            <div class="mt-3 pt-3" style="border-top:1px solid var(--fm-border);">
                <div class="card-title-row" style="margin-bottom:1rem;"><i class="fa-solid fa-image"></i>Grid Grayscale (Input PCA)</div>
                <div class="row g-3">
                    <div class="col-6 text-center">
                        <img src="data:image/png;base64,{d.get('gray1','')}" style="width:100%;max-width:160px;border-radius:10px;border:1px solid var(--fm-border);" alt="Grayscale Kecil">
                        <div class="preview-label">Foto Masa Kecil (grayscale 100×100)</div>
                    </div>
                    <div class="col-6 text-center">
                        <img src="data:image/png;base64,{d.get('gray2','')}" style="width:100%;max-width:160px;border-radius:10px;border:1px solid var(--fm-border);" alt="Grayscale Dewasa">
                        <div class="preview-label">Foto Masa Dewasa (grayscale 100×100)</div>
                    </div>
                </div>
            </div>

            <div class="mt-3 pt-3" style="border-top:1px solid var(--fm-border);">
                <div class="card-title-row" style="margin-bottom:1rem;"><i class="fa-solid fa-chart-simple"></i>Komposisi Fitur PCA (per Foto)</div>
                <div class="row g-3">
                    <div class="col-md-6">{fitur_bar_block("Foto Masa Kecil", d.get('fitur1',{}))}</div>
                    <div class="col-md-6">{fitur_bar_block("Foto Masa Dewasa", d.get('fitur2',{}))}</div>
                </div>
                <div style="font-size:0.78rem;color:var(--fm-muted);margin-top:0.6rem;">
                    <i class="fa-solid fa-circle-info me-1"></i>PC1–PC3 menunjukkan kontribusi komponen utama hasil reduksi dimensi (SVD); Energi Total = Σz², Kemiringan = ukuran asimetri distribusi nilai vektor z.
                </div>
            </div>
        </div>
    </div>
    <div class="meta-info-row">{meta_html}</div>
    """
else:
    hasil_html = """
    <div class="empty-state">
        <i class="fa-solid fa-circle-user"></i>
        <p class="mb-1 fw-semibold text-dark">Belum ada foto yang dianalisis</p>
        <small>Upload foto masa kecil dan masa dewasa lalu klik Analisis Kemiripan.</small>
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
col_left, col_right = st.columns([1, 1], gap="medium")

with col_left:
    with st.container(border=True):
        st.markdown('<div class="fm-card-title"><i class="fa fa-cloud-arrow-up"></i> Upload Foto</div>', unsafe_allow_html=True)

        st.markdown('<div class="fm-field-label">👶 Foto Masa Kecil</div>', unsafe_allow_html=True)
        foto_kecil = st.file_uploader("foto_kecil", type=["jpg","jpeg","png","jfif","webp"], label_visibility="collapsed", key="up_kecil")

        st.markdown('<div class="fm-field-label">🧑 Foto Masa Dewasa</div>', unsafe_allow_html=True)
        foto_dewasa = st.file_uploader("foto_dewasa", type=["jpg","jpeg","png","jfif","webp"], label_visibility="collapsed", key="up_dewasa")

        st.markdown('''<div class="fm-field-label" style="display:flex;align-items:center;justify-content:space-between;">
            <span>⚙️ Threshold Kemiripan PCA</span>
            <span style="background:var(--fm-bg);color:var(--fm-muted);font-size:0.7rem;font-weight:700;padding:0.15rem 0.55rem;border-radius:20px;border:1px solid var(--fm-border);">Opsional</span>
        </div>''', unsafe_allow_html=True)
        st.caption("Default 0.60 sudah dioptimalkan. Centang di bawah hanya jika ingin menyesuaikan.")
        atur_manual = st.checkbox("Atur threshold secara manual", value=False)
        if atur_manual:
            threshold = st.slider("threshold", min_value=0.30, max_value=0.90, value=0.60, step=0.05, label_visibility="collapsed")
            st.caption(f"Threshold aktif: {threshold:.2f} | Makin tinggi = makin ketat")
        else:
            threshold = 0.60
            st.caption(f"Menggunakan nilai otomatis: {threshold:.2f}")

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
{CSS}
<style>body{{background:transparent !important;}} .main-card{{border:none !important;}}</style>
</head><body style="background:transparent;padding:0;margin:0;">
<div id="fm-resize-root">
    {hasil_html}
</div>
<script>
const pc=document.getElementById('pctCounter'),gf=document.getElementById('gaugeFg');
if(pc&&gf){{const tg=parseFloat(pc.dataset.target)||0,ci=427.26,dur=1200,st0=performance.now();
function anim(now){{const el=now-st0,pr=Math.min(el/dur,1),ea=1-Math.pow(1-pr,3),cv=tg*ea;
pc.textContent=cv.toFixed(2)+'%';gf.style.strokeDashoffset=ci-(ci*(cv/100));
if(pr<1)requestAnimationFrame(anim);else{{pc.textContent=tg.toFixed(2)+'%';gf.style.strokeDashoffset=ci-(ci*(tg/100));}}}}
requestAnimationFrame(anim);}}
function togglePembuktian(btn){{const box=document.getElementById('boxPembuktian'),ic=btn.querySelector('.fa-chevron-down');
if(!box)return;const bk=box.style.display==='none';box.style.display=bk?'block':'none';
ic.style.transform=bk?'rotate(180deg)':'rotate(0deg)';}}
</script>
</body></html>"""
        _page_height = 770 if (hasil and hasil.get("status") == "ok") else 380
        components.html(PAGE_HTML, height=_page_height, scrolling=True)

# ── Section bawah: Riwayat & Info ─────────────────────────────
_hasil_json2 = json.dumps({
    "status": hasil.get("status") if hasil else None,
    "persentase": hasil.get("persentase") if hasil else None,
    "kesimpulan": hasil.get("kesimpulan","") if hasil else "",
    "metode": metode
})

BOTTOM_HTML = f"""
<!DOCTYPE html><html lang="id"><head><meta charset="UTF-8">
{CSS}
</head><body style="background:transparent;padding:0;margin:0;">
<div class="container-fluid px-0">

    <div class="row mb-4">
        <div class="col-12">
            <div class="card main-card shadow-sm p-4 bg-white">
                <div class="d-flex justify-content-between align-items-center mb-4 flex-wrap gap-2">
                    <div class="card-title-row mb-0"><i class="fa-solid fa-clock-rotate-left"></i>Riwayat Analisis</div>
                    <div class="d-flex align-items-center gap-2">
                        <span class="history-badge-count" id="historyCount">0 Data Tersimpan</span>
                        <button type="button" class="btn-clear" id="btnClearHistory" style="display:none;">
                            <i class="fa-solid fa-trash-can me-1"></i>Hapus Riwayat
                        </button>
                    </div>
                </div>
                <div class="table-responsive">
                    <table class="table align-middle mb-0">
                        <thead><tr>
                            <th>ID</th><th>Nama File Kecil</th><th>Nama File Dewasa</th>
                            <th>Similarity</th><th>Metode</th><th>Status</th><th>Waktu</th>
                        </tr></thead>
                        <tbody id="historyTableBody">
                            <tr><td colspan="7" class="text-center py-4 text-muted">
                                <i class="fa-solid fa-inbox d-block mb-2 fs-3 text-black-50"></i>
                                Belum ada riwayat pengujian data.
                            </td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <div class="row mb-4" id="chartContainer" style="display:none;">
        <div class="col-12">
            <div class="card main-card shadow-sm p-4 bg-white">
                <div class="card-title-row"><i class="fa-solid fa-chart-line"></i>Tren Similarity dari Riwayat</div>
                <div class="chart-wrap"><canvas id="trendChart"></canvas></div>
                <div class="chart-stats">
                    <div class="chart-stat"><div class="stat-value" id="statAvg">0%</div><div class="stat-label">Rata-rata Similarity</div></div>
                    <div class="chart-stat"><div class="stat-value" id="statMax">0%</div><div class="stat-label">Tertinggi</div></div>
                    <div class="chart-stat"><div class="stat-value" id="statSame">0</div><div class="stat-label">Hasil "Sama"</div></div>
                    <div class="chart-stat"><div class="stat-value" id="statDiff">0</div><div class="stat-label">Hasil "Tidak Mirip"</div></div>
                </div>
            </div>
        </div>
    </div>

    <div class="row g-4">
        <div class="col-lg-5">
            <div class="info-strip h-100">
                <h6><span class="icon-circle"><i class="fa-solid fa-lightbulb"></i></span>Konsep Aljabar Linear yang Digunakan</h6>
                <div>
                    <span class="concept-pill"><i class="fa-solid fa-circle-check"></i>Matriks (Representasi Pixel)</span>
                    <span class="concept-pill"><i class="fa-solid fa-circle-check"></i>Vektor (Flatten / Embedding)</span>
                    <span class="concept-pill"><i class="fa-solid fa-circle-check"></i>Dot Product</span>
                    <span class="concept-pill"><i class="fa-solid fa-circle-check"></i>Norma Vektor</span>
                    <span class="concept-pill"><i class="fa-solid fa-circle-check"></i>Cosine Similarity</span>
                    <span class="concept-pill"><i class="fa-solid fa-circle-check"></i>PCA / Eigenfaces</span>
                    <span class="concept-pill"><i class="fa-solid fa-circle-check"></i>SVD (Xc = UΣVᵀ)</span>
                    <span class="concept-pill"><i class="fa-solid fa-circle-check"></i>Reduksi Dimensi</span>
                    <span class="concept-pill"><i class="fa-solid fa-circle-check"></i>Euclidean Distance</span>
                </div>
            </div>
        </div>
        <div class="col-lg-7">
            <div class="info-strip h-100">
                <h6><span class="icon-circle"><i class="fa-solid fa-circle-exclamation"></i></span>Catatan Penggunaan</h6>
                <div style="font-size:0.87rem;color:var(--fm-dark);line-height:1.7;">
                    <div style="background:var(--fm-green-bg);border:1px solid #c7ecdb;border-radius:10px;padding:0.65rem 0.85rem;margin-bottom:0.7rem;">
                        <div style="font-weight:700;color:var(--fm-green);font-size:0.82rem;margin-bottom:0.25rem;"><i class="fa-solid fa-layer-group me-1"></i>PCA / Eigenfaces</div>
                        <div style="color:#444;font-size:0.82rem;">Metode Aljabar Linear klasik berbasis reduksi dimensi (SVD). Tujuannya untuk mendeteksi kemiripan wajah melalui proyeksi ke ruang eigenfaces.<br><br>
                        <span style="color:#c0392b;font-weight:600;">⚠ Keterbatasan:</span> PCA sensitif terhadap perubahan pencahayaan, posisi, dan ekspresi wajah.</div>
                    </div>
                    <div style="background:#f7f7fa;border:1px solid #ececf3;border-radius:10px;padding:0.65rem 0.85rem;">
                        <div style="font-weight:700;color:#8c8fa3;font-size:0.82rem;margin-bottom:0.25rem;"><i class="fa-solid fa-sliders me-1"></i>Pengaturan Threshold</div>
                        <div style="color:#444;font-size:0.82rem;">Threshold default: <strong>0.60</strong>. Nilai lebih tinggi = lebih ketat. Nilai lebih rendah = lebih longgar. Atur melalui slider di panel upload.</div>
                    </div>
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
    if(r.length===0){{tb.innerHTML='<tr><td colspan="7" class="text-center py-4 text-muted"><i class="fa-solid fa-inbox d-block mb-2 fs-3 text-black-50"></i>Belum ada riwayat.</td></tr>';bc.style.display='none';cc.style.display='none';return;}}
    bc.style.display='inline-block';cc.style.display='block';
    tb.innerHTML=r.map(i=>`<tr><td><span class="text-secondary fw-semibold">#${{i.id}}</span></td>
    <td><small class="text-muted">${{i.foto_kecil}}</small></td><td><small class="text-muted">${{i.foto_dewasa}}</small></td>
    <td><span class="fw-bold" style="color:var(--fm-pink)">${{i.persentase}}%</span></td>
    <td><span class="method-badge ${{i.metode==='pca'?'pca':'arcface'}}">${{i.metode==='pca'?'PCA':'ArcFace'}}</span></td>
    <td><span class="badge-status ${{i.status==='Sama'?'same':'diff'}}">${{i.status}}</span></td>
    <td><small class="text-black-50">${{i.tanggal}}</small></td></tr>`).join('');
    const dk=[...r].reverse(),lbl=dk.map(i=>'#'+i.id),val=dk.map(i=>parseFloat(i.persentase));
    const tc=document.getElementById('trendChart');
    if(tc){{if(trendChartInstance)trendChartInstance.destroy();
    trendChartInstance=new Chart(tc,{{type:'line',data:{{labels:lbl,datasets:[{{label:'Similarity (%)',data:val,
    borderColor:'#ec4f7f',backgroundColor:'rgba(236,79,127,0.1)',borderWidth:2.5,
    pointBackgroundColor:'#ec4f7f',pointRadius:4,tension:0.35,fill:true}}]}},
    options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}}}},
    scales:{{y:{{beginAtZero:true,max:100,ticks:{{callback:function(v){{return v+'%'}}}},
    grid:{{color:'#ececf3'}}}},x:{{grid:{{display:false}}}}}}}}}});}}
    const tot=r.length,avg=(val.reduce((a,b)=>a+b,0)/tot).toFixed(2),mx=Math.max(...val).toFixed(2),
    sm=r.filter(i=>i.status==='Sama').length,df=tot-sm;
    document.getElementById('statAvg').textContent=avg+'%';
    document.getElementById('statMax').textContent=mx+'%';
    document.getElementById('statSame').textContent=sm;
    document.getElementById('statDiff').textContent=df;
}}
const _hasil={_hasil_json2};
const _fk="{_nama_kecil}";
const _fd="{_nama_dewasa}";
if(_hasil.status==='ok'&&_fk&&_fd){{
    const r=dapatkanRiwayat();
    const pct=parseFloat(_hasil.persentase);
    const kes=_hasil.kesimpulan;
    const mt=_hasil.metode;
    const st2=(kes.includes('besar')||kes.includes('sedang'))?'Sama':'Tidak Mirip';
    const nid=r.length>0?Math.max(...r.map(o=>o.id))+1:1;
    const now=new Date();
    const wkt=now.toLocaleDateString('id-ID',{{day:'numeric',month:'short'}})+' '+now.toLocaleTimeString('id-ID',{{hour:'2-digit',minute:'2-digit'}});
    const nb={{id:nid,foto_kecil:_fk,foto_dewasa:_fd,persentase:pct.toFixed(2),metode:mt,status:st2,tanggal:wkt}};
    const ada=r.some(i=>i.foto_kecil===nb.foto_kecil&&i.foto_dewasa===nb.foto_dewasa&&i.persentase===nb.persentase&&i.metode===nb.metode);
    if(!ada){{r.unshift(nb);simpanRiwayat(r);}}
}}
document.addEventListener('DOMContentLoaded',function(){{
    renderRiwayat();
    document.getElementById('btnClearHistory').addEventListener('click',function(){{
        if(confirm('Hapus seluruh riwayat?')){{localStorage.removeItem('facematch_history');renderRiwayat();}}
    }});
}});
</script>
</body></html>"""

components.html(BOTTOM_HTML, height=900, scrolling=True)