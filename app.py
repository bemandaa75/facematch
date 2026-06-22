from PIL import Image
from flask import Flask, render_template, request, send_from_directory
import os
import cv2
import numpy as np
from deepface import DeepFace
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
DATASET_FOLDER = "dataset"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# KONFIGURASI PCA

N_COMPONENTS = 50       # k komponen utama (sesuai PDF dosen)
THRESHOLD_PCA = 0.60    # threshold cosine similarity di ruang PCA

# Variabel global model PCA
pca_model    = None
scaler       = None     # StandardScaler untuk normalisasi embedding
mean_embed   = None     # mean embedding untuk centering (Xc = X - X̄)
labels_db    = None
X_pca_db     = None


# FUNGSI PCA BERBASIS EMBEDDING (bukan piksel mentah)

def ambil_embedding(path):
    """
    Langkah 1: Ekstrak embedding wajah menggunakan Facenet512.
    Menghasilkan vektor 512 dimensi yang merepresentasikan fitur wajah.
    Ini menggantikan flatten piksel di PDF dosen — hasilnya jauh lebih
    bermakna karena sudah memahami struktur wajah.
    """
    result = DeepFace.represent(
        img_path=path,
        model_name="Facenet512",
        enforce_detection=False,
        detector_backend="opencv"
    )
    if not result:
        raise ValueError("Wajah tidak terdeteksi.")
    return np.array(result[0]["embedding"])  # vektor 512 dimensi


def latih_pca_dari_dataset(dataset_path=DATASET_FOLDER):
    """
    Langkah 2-5 PDF Dosen (berbasis embedding, bukan piksel):
    - Bentuk matriks X (m x 512) dari embedding tiap foto di dataset
    - Centering data: Xc = X - X̄  (X̄ = mean embedding)
    - PCA via SVD: Xc = U Σ Vᵀ
    - Ambil k=50 komponen utama → ruang eigenfaces berdimensi rendah
    """
    global pca_model, scaler, mean_embed, labels_db, X_pca_db

    print("[PCA] Memulai pelatihan dari dataset (embedding-based)...")
    X, labels = [], []

    if not os.path.exists(dataset_path):
        print(f"[PCA] Folder dataset '{dataset_path}' tidak ditemukan!")
        return False

    for nama_orang in sorted(os.listdir(dataset_path)):
        folder = os.path.join(dataset_path, nama_orang)
        if not os.path.isdir(folder):
            continue
        for file in sorted(os.listdir(folder)):
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            try:
                emb = ambil_embedding(os.path.join(folder, file))
                X.append(emb)
                labels.append(nama_orang)
            except Exception:
                pass

    total_gambar = len(X)
    total_orang  = len(set(labels))

    if total_gambar < N_COMPONENTS:
        print(f"[PCA] Dataset terlalu kecil ({total_gambar}). Minimal {N_COMPONENTS}.")
        return False

    # Langkah 3: Matriks X (m x 512)
    X = np.array(X)
    print(f"[PCA] Matriks X: {X.shape}  ({total_orang} orang, {total_gambar} foto)")

    # Langkah 4: Centering — Xc = X - X̄
    mean_embed = np.mean(X, axis=0)
    Xc = X - mean_embed

    # Normalisasi skala antar dimensi embedding
    scaler = StandardScaler(with_mean=False)
    Xc_scaled = scaler.fit_transform(Xc)

    # Langkah 5: PCA via SVD → k=50 komponen utama
    k = min(N_COMPONENTS, total_gambar - 1, 512)
    pca_model = PCA(n_components=k)
    X_pca_db  = pca_model.fit_transform(Xc_scaled)
    labels_db = np.array(labels)

    explained = np.sum(pca_model.explained_variance_ratio_) * 100
    print(f"[PCA] Selesai! {k} komponen menjelaskan {explained:.1f}% variansi.")
    return True


def bandingkan_wajah_pca(path1, path2, threshold=THRESHOLD_PCA):
    """
    Langkah 6 + Keputusan Kemiripan (PDF Dosen):
    - Ekstrak embedding kedua foto (vektor 512 dimensi)
    - Centering: kurangi mean_embed
    - Proyeksi ke ruang PCA: Z = Xc × Vk
    - Hitung cosine similarity dan Euclidean distance di ruang PCA
    - Keputusan berdasarkan threshold
    """
    if pca_model is None or mean_embed is None:
        return {
            "status": "error",
            "pesan": "Model PCA belum dilatih. Pastikan folder dataset tersedia."
        }

    try:
        # Ekstrak embedding — vektor 512 dimensi per foto
        emb1 = ambil_embedding(path1).reshape(1, -1)
        emb2 = ambil_embedding(path2).reshape(1, -1)

        # Langkah 4: Centering — Xc = X - X̄
        emb1_c = emb1 - mean_embed
        emb2_c = emb2 - mean_embed

        # Normalisasi skala
        emb1_s = scaler.transform(emb1_c)
        emb2_s = scaler.transform(emb2_c)

        # Langkah 6: Proyeksi ke ruang PCA — Z = Xc × Vk
        z1 = pca_model.transform(emb1_s).flatten()
        z2 = pca_model.transform(emb2_s).flatten()

        # Cosine Similarity di ruang PCA
        # cos(θ) = (z1 · z2) / (‖z1‖ · ‖z2‖)
        dot_product    = float(np.dot(z1, z2))
        norm1          = float(np.linalg.norm(z1))
        norm2          = float(np.linalg.norm(z2))
        cosine_sim     = dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0

        # Euclidean Distance di ruang PCA
        euclidean_dist = float(np.linalg.norm(z1 - z2))

        # Keputusan kemiripan
        if cosine_sim >= threshold:
            persentase = round(min((cosine_sim + 0.20) * 100, 99.0), 2)
            kesimpulan = "Kemungkinan besar orang yang sama (Terverifikasi via PCA/Eigenfaces)"
        elif cosine_sim >= threshold - 0.20:
            persentase = round(cosine_sim * 100, 2)
            kesimpulan = "Kemungkinan orang yang sama (Kemiripan sedang via PCA)"
        else:
            persentase = round(max(cosine_sim * 100, 0.0), 2)
            kesimpulan = "Kemungkinan bukan orang yang sama (PCA/Eigenfaces)"

        return {
            "status"            : "ok",
            "metode"            : "PCA/Eigenfaces",
            "dot_product"       : round(dot_product, 4),
            "norma_a"           : round(norm1, 4),
            "norma_b"           : round(norm2, 4),
            "cosine_similarity" : round(cosine_sim, 4),
            "euclidean_distance": round(euclidean_dist, 4),
            "persentase"        : persentase,
            "kesimpulan"        : kesimpulan,
            "n_components"      : pca_model.n_components_,
            "total_dataset"     : len(labels_db) if labels_db is not None else 0,
        }

    except Exception as e:
        return {"status": "error", "pesan": str(e)}


# FUNGSI ARCFACE (deep learning)

def bandingkan_wajah(path_kecil, path_dewasa):
    try:
        embedding_kecil  = DeepFace.represent(img_path=path_kecil,  model_name="ArcFace", enforce_detection=False)
        embedding_dewasa = DeepFace.represent(img_path=path_dewasa, model_name="ArcFace", enforce_detection=False)

        if not embedding_kecil or not embedding_dewasa:
            return {"status": "error", "pesan": "Wajah tidak terdeteksi di salah satu foto."}

        if len(embedding_kecil) > 1:
            embedding_kecil = sorted(embedding_kecil, key=lambda x: x["facial_area"]["w"] * x["facial_area"]["h"])

        vektor1 = np.array(embedding_kecil[0]["embedding"])
        vektor2 = np.array(embedding_dewasa[0]["embedding"])

        dot_product  = np.dot(vektor1, vektor2)
        norm1        = np.linalg.norm(vektor1)
        norm2        = np.linalg.norm(vektor2)
        cosine_sim   = dot_product / (norm1 * norm2)

        vektor1_mean = vektor1 - np.mean(vektor1)
        vektor2_mean = vektor2 - np.mean(vektor2)
        norm_v1m     = np.linalg.norm(vektor1_mean)
        norm_v2m     = np.linalg.norm(vektor2_mean)

        korelasi_pola = (
            np.dot(vektor1_mean, vektor2_mean) / (norm_v1m * norm_v2m)
            if norm_v1m > 0 and norm_v2m > 0 else 0.0
        )

        if cosine_sim < 0.18:
            persentase_keyakinan = round(max(float(cosine_sim) * 100, 0.0), 2)
            kesimpulan = "Kemungkinan bukan orang yang sama (Terdeteksi sebagai orang yang berbeda)"
        elif cosine_sim < 0.35:
            if korelasi_pola >= 0.25:
                persentase_keyakinan = round(min((cosine_sim + 0.40) * 100, 95.0), 2)
                kesimpulan = "Kemungkinan besar orang yang sama (Terverifikasi lewat pola struktur lintas usia)"
            else:
                persentase_keyakinan = round(float(cosine_sim) * 100, 2)
                kesimpulan = "Kemungkinan bukan orang yang sama (Struktur geometri tidak selaras)"
        else:
            persentase_keyakinan = round(min((cosine_sim + 0.30) * 100, 99.0), 2)
            kesimpulan = "Kemungkinan besar orang yang sama (Terverifikasi secara Pola Geometri)"

        return {
            "status"           : "ok",
            "metode"           : "ArcFace",
            "dot_product"      : round(float(dot_product), 4),
            "norma_a"          : round(float(norm1), 4),
            "norma_b"          : round(float(norm2), 4),
            "cosine_similarity": round(float(cosine_sim), 4),
            "persentase"       : persentase_keyakinan,
            "kesimpulan"       : kesimpulan,
        }

    except Exception as e:
        return {"status": "error", "pesan": str(e)}

# FLASK ROUTES

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        foto_kecil  = request.files["foto_kecil"]
        foto_dewasa = request.files["foto_dewasa"]
        metode      = request.form.get("metode", "arcface")

        path_kecil  = os.path.join(app.config["UPLOAD_FOLDER"], foto_kecil.filename)
        path_dewasa = os.path.join(app.config["UPLOAD_FOLDER"], foto_dewasa.filename)
        foto_kecil.save(path_kecil)
        foto_dewasa.save(path_dewasa)

        if metode == "pca":
            # Baca threshold dari form kalau user atur, pakai default 0.60 kalau tidak
            try:
                threshold = float(request.form.get("threshold_pca", 0.60))
                threshold = max(0.30, min(0.90, threshold))
            except (ValueError, TypeError):
                threshold = 0.60
            hasil_dipilih = bandingkan_wajah_pca(path_kecil, path_dewasa, threshold=threshold)
        else:
            metode = "arcface"
            hasil_dipilih = bandingkan_wajah(path_kecil, path_dewasa)

        if hasil_dipilih.get("status") == "ok":
            hasil = {"status": "ok", "metode": metode, "data": hasil_dipilih}
        else:
            hasil = {
                "status": "error",
                "metode": metode,
                "pesan" : hasil_dipilih.get("pesan", "Terjadi kesalahan.")
            }

        return render_template(
            "index.html",
            hasil=hasil,
            foto_kecil=foto_kecil.filename,
            foto_dewasa=foto_dewasa.filename,
            riwayat=[]
        )

    return render_template("index.html", hasil=None, riwayat=[])


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# training pca saat aplikasi jalan
print("[APP] Memulai pelatihan PCA dari dataset...")
pca_berhasil = latih_pca_dari_dataset()
if pca_berhasil:
    print("[APP] Model PCA siap digunakan!")
else:
    print("[APP] PCA tidak aktif. Hanya ArcFace yang tersedia.")

if __name__ == "__main__":
    app.run(debug=True)