import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms
import time
import io
import base64
import textwrap
from pathlib import Path
import gdown

import plotly.graph_objects as go

# =========================================================
# 1. KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="NutriCocoa AI - Dashboard Deteksi Hara Kakao",
    page_icon="🍫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 2. DESIGN SYSTEM
# Catatan perbaikan dari versi sebelumnya:
#  - SEMUA blok HTML multi-baris di-dedent (textwrap.dedent) sebelum
#    dikirim ke st.markdown(). Tanpa ini, Streamlit/Markdown akan
#    membaca baris yang berindentasi >=4 spasi sebagai blok kode,
#    sehingga tag HTML muncul sebagai teks mentah (inilah penyebab
#    bug "kode HTML tampil sebagai teks" pada screenshot sebelumnya).
#  - Dependensi font "Material Symbols Outlined" dari Google Fonts
#    dihapus karena saat gagal dimuat (mis. tidak ada akses internet
#    di localhost) ia hanya menampilkan teks nama ikon mentah seperti
#    "agriculture", "warning", dsb. Diganti dengan ikon SVG inline
#    sederhana yang tidak bergantung pada koneksi eksternal.
#  - Tema dipaksa terang & lebih hidup agar tidak lagi hitam-putih.
# =========================================================
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">

<style>
:root{
  --primary:#0f6b3f;
  --primary-dark:#0a4a2c;
  --primary-light:#e6f6ee;
  --accent:#ffb703;
  --accent-dark:#7a4a00;
  --accent-light:#fff4dc;
  --danger:#d6453d;
  --danger-light:#fbe9e8;
  --surface:#ffffff;
  --surface-alt:#f6f8f6;
  --border:#e1e7e2;
  --text:#1a231d;
  --text-soft:#5b6960;
}

html, body, [class*="css"]{ font-family:'Plus Jakarta Sans', sans-serif; color:var(--text); }
.stApp{ background: linear-gradient(180deg, #f4faf6 0%, #f8f9f7 320px, #f8f9f7 100%); }

/* Streamlit membungkus HTML kustom kita di dalam stMarkdownContainer dengan
   CSS bawaannya sendiri (termasuk warna teks yang kadang pudar/transparan
   untuk elemen <p>, <span>, dsb). Selector generik di atas kalah spesifisitas,
   sehingga sebagian teks (mis. isi "Solusi", label sumbu grafik) jadi nyaris
   tak terbaca. Paksa semua teks di dalam kartu/komponen kustom kita memakai
   warna eksplisit dengan !important agar selalu kontras. */
[data-testid="stMarkdownContainer"] .nc-card,
[data-testid="stMarkdownContainer"] .nc-reco,
[data-testid="stMarkdownContainer"] .nc-step-card,
[data-testid="stMarkdownContainer"] .nc-hero-badge,
[data-testid="stMarkdownContainer"] .nc-info-box,
[data-testid="stMarkdownContainer"] .nc-header,
[data-testid="stMarkdownContainer"] .nc-empty-card{
  color: var(--text) !important;
}
[data-testid="stMarkdownContainer"] .nc-card p,
[data-testid="stMarkdownContainer"] .nc-card span,
[data-testid="stMarkdownContainer"] .nc-card div,
[data-testid="stMarkdownContainer"] .nc-reco p,
[data-testid="stMarkdownContainer"] .nc-reco span,
[data-testid="stMarkdownContainer"] .nc-reco div,
[data-testid="stMarkdownContainer"] .nc-step-card p,
[data-testid="stMarkdownContainer"] .nc-step-card span,
[data-testid="stMarkdownContainer"] .nc-step-card div{
  color: inherit !important;
}
[data-testid="stMarkdownContainer"] .nc-reco-body{ color: var(--text) !important; }
[data-testid="stMarkdownContainer"] .nc-reco-solution{ color: var(--text) !important; }
[data-testid="stMarkdownContainer"] .nc-reco-solution .lbl{ color: var(--primary-dark) !important; }
[data-testid="stMarkdownContainer"] .nc-label-caps,
[data-testid="stMarkdownContainer"] .nc-conf-meta,
[data-testid="stMarkdownContainer"] .nc-inference-time,
[data-testid="stMarkdownContainer"] .nc-empty-desc,
[data-testid="stMarkdownContainer"] .nc-step-desc,
[data-testid="stMarkdownContainer"] .nc-subtitle{ color: var(--text-soft) !important; }
[data-testid="stMarkdownContainer"] .nc-pred-title,
[data-testid="stMarkdownContainer"] .nc-empty-title,
[data-testid="stMarkdownContainer"] .nc-step-title,
[data-testid="stMarkdownContainer"] .nc-section-head h3,
[data-testid="stMarkdownContainer"] .nc-header h1{ color: var(--primary-dark) !important; }
[data-testid="stMarkdownContainer"] .nc-conf-badge,
[data-testid="stMarkdownContainer"] .nc-hero-badge{ color: #ffffff !important; }

[data-testid="stSidebar"]{
  background: linear-gradient(180deg, var(--primary-dark), var(--primary));
  border-right: none;
}
[data-testid="stSidebar"] * { color: #eafbf1 !important; }
[data-testid="stSidebar"] .nc-sidebar-sub{ color:#bfe6cf !important; }

.nc-ico{ display:inline-flex; align-items:center; justify-content:center; width:1em; height:1em; vertical-align:-0.15em; }
.nc-ico svg{ width:100%; height:100%; }

/* ---- Header ---- */
.nc-header{ display:flex; align-items:center; gap:12px; margin-bottom:4px; }
.nc-header-icon{ width:46px; height:46px; border-radius:14px; background:var(--primary); display:flex; align-items:center; justify-content:center; flex-shrink:0; }
.nc-header-icon svg{ width:26px; height:26px; }
.nc-header h1{ font-size:30px; font-weight:800; color:var(--primary-dark); margin:0; letter-spacing:-0.02em; }
.nc-subtitle{ color:var(--text-soft); font-size:15px; margin:10px 0 28px 0; line-height:1.6; max-width:780px; }

/* ---- Generic card ---- */
.nc-card{
  background:var(--surface);
  border:1px solid var(--border);
  border-radius:20px;
  padding:24px 26px;
  box-shadow:0 4px 18px rgba(15,107,63,0.06);
}

/* ---- Hero image ---- */
.nc-hero-wrap{ position:relative; border-radius:20px; overflow:hidden; border:1px solid var(--border); margin-bottom:22px; box-shadow:0 6px 20px rgba(15,107,63,0.08);}
.nc-hero-wrap img{ width:100%; max-height:380px; object-fit:cover; display:block; }
.nc-hero-badge{
  position:absolute; bottom:14px; left:14px;
  display:flex; align-items:center; gap:7px;
  background:rgba(10,74,44,0.85); color:#fff;
  padding:7px 16px; border-radius:999px;
  font-size:12.5px; font-weight:600; letter-spacing:0.02em;
  backdrop-filter: blur(6px);
}

/* ---- Prediction card ---- */
.nc-label-caps{ font-size:12px; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:var(--text-soft); }
.nc-pred-row{ display:flex; justify-content:space-between; align-items:flex-start; gap:14px; margin-bottom:18px; flex-wrap:wrap; }
.nc-pred-title{ font-size:24px; font-weight:800; color:var(--primary-dark); margin-top:4px; }
.nc-conf-badge{ background:var(--primary); color:#fff; padding:9px 20px; border-radius:14px; font-weight:800; font-size:18px; white-space:nowrap; }
.nc-conf-meta{ display:flex; justify-content:space-between; font-size:13.5px; color:var(--text-soft); margin-bottom:7px; }
.nc-progress-track{ width:100%; background:var(--surface-alt); height:12px; border-radius:999px; overflow:hidden; border:1px solid var(--border); }
.nc-progress-fill{ height:100%; border-radius:999px; background:linear-gradient(90deg, var(--primary), #2bb673); transition:width 0.6s ease; }
.nc-inference-time{ margin-top:14px; font-size:12.5px; color:var(--text-soft); display:flex; align-items:center; gap:6px;}

/* ---- Recommendation card ---- */
.nc-reco{ border-radius:18px; padding:22px 24px; border:1px solid var(--accent); background:var(--accent-light); margin-top:22px; }
.nc-reco.success{ border-color:var(--primary); background:var(--primary-light); }
.nc-reco-head{ display:flex; align-items:center; gap:9px; font-weight:800; font-size:17px; color:var(--accent-dark); margin-bottom:11px; }
.nc-reco.success .nc-reco-head{ color:var(--primary-dark); }
.nc-reco-body{ font-size:14.5px; color:var(--text); line-height:1.65; }
.nc-reco-solution{ background:rgba(255,255,255,0.8); border:1px solid rgba(0,0,0,0.06); border-radius:14px; padding:14px 18px; margin-top:14px; font-size:14px; line-height:1.6; }
.nc-reco-solution .lbl{ color:var(--primary-dark); font-weight:700; display:block; margin-bottom:4px; }

/* ---- Section heads ---- */
.nc-section-head{ display:flex; align-items:center; gap:8px; margin:34px 0 16px 0; }
.nc-section-head h3{ font-size:19px; font-weight:800; color:var(--primary-dark); margin:0; }

/* ---- Sidebar ---- */
.nc-sidebar-brand{ display:flex; align-items:center; gap:10px; font-size:20px; font-weight:800; margin-bottom:2px; }
.nc-sidebar-brand .nc-header-icon{ width:36px; height:36px; border-radius:11px; background:rgba(255,255,255,0.15); }
.nc-sidebar-brand .nc-header-icon svg{ width:20px; height:20px; }
.nc-sidebar-sub{ font-size:12.5px; margin-bottom:18px; }
.nc-info-box{ background:rgba(255,255,255,0.1); padding:16px 17px; border-radius:16px; font-size:13.5px; line-height:1.7; border:1px solid rgba(255,255,255,0.15); }
.nc-info-row{ display:flex; justify-content:space-between; margin-top:5px; }
.nc-info-row:first-child{ margin-top:0; }
.nc-info-row b{ color:#ffffff !important; }

/* ---- Empty state ---- */
.nc-empty-card{ display:flex; align-items:center; gap:16px; }
.nc-empty-icon{ width:48px; height:48px; border-radius:14px; background:var(--primary-light); display:flex; align-items:center; justify-content:center; flex-shrink:0; }
.nc-empty-icon svg{ width:24px; height:24px; }
.nc-empty-title{ font-weight:800; color:var(--primary-dark); font-size:16px; }
.nc-empty-desc{ font-size:13.5px; color:var(--text-soft); }

.nc-step-card{ background:var(--surface); border:1px solid var(--border); border-radius:18px; padding:20px 18px; height:100%; }
.nc-step-num{ display:inline-flex; align-items:center; justify-content:center; width:30px; height:30px; border-radius:9px; background:var(--primary-light); color:var(--primary-dark); font-size:13px; font-weight:800; }
.nc-step-title{ font-weight:800; margin:10px 0 6px 0; color:var(--primary-dark); font-size:15px; }
.nc-step-desc{ font-size:13px; color:var(--text-soft); line-height:1.55; }

.nc-footer{ text-align:center; color:var(--text-soft); font-size:12px; margin-top:40px; padding-top:20px; border-top:1px solid var(--border); }
</style>
""", unsafe_allow_html=True)

# Ikon SVG inline (tanpa dependensi font eksternal / tanpa "ikon AI" generik)
ICONS = {
    "leaf": '<svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 20A7 7 0 0 1 4 13c0-5 4-9 11-13 1 7-1 11-4 13"/><path d="M4 13c2-1 5-2 8-2"/></svg>',
    "upload": '<svg viewBox="0 0 24 24" fill="none" stroke="#0f6b3f" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>',
    "check": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>',
    "alert": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    "camera": '<svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg>',
}


def icon(name, size="1em"):
    return f'<span class="nc-ico" style="width:{size};height:{size};">{ICONS[name]}</span>'


def md(html: str):
    """st.markdown wrapper yang selalu dedent agar tidak terbaca sebagai code block."""
    st.markdown(textwrap.dedent(html), unsafe_allow_html=True)


# =========================================================
# 3. VARIABEL GLOBAL & DATA DOMAIN
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = ['kalium', 'kalsium', 'magnesium', 'nitrogen', 'normal', 'posfor']

CLASS_LABELS = {
    "kalium": "Kalium (K)",
    "kalsium": "Kalsium (Ca)",
    "magnesium": "Magnesium (Mg)",
    "nitrogen": "Nitrogen (N)",
    "normal": "Normal",
    "posfor": "Fosfor (P)",
}

CLASS_COLORS = {
    "kalium": "#7a4a00",
    "kalsium": "#0f6b3f",
    "magnesium": "#2bb673",
    "nitrogen": "#d6453d",
    "normal": "#0f6b3f",
    "posfor": "#ffb703",
}

TIMM_MODEL_NAME = "swin_tiny_patch4_window7_224"
MODEL_PATH = Path(__file__).resolve().parent / "swin_kakao.pth"
MODEL_VAL_ACCURACY = 0.976

# Bobot model (.pth) berukuran besar sehingga tidak disimpan di repo GitHub
# (lihat .gitignore). Sebagai gantinya, file di-hosting di Google Drive dan
# diunduh secara otomatis saat aplikasi pertama kali dijalankan.
GDRIVE_FILE_ID = "1smdt6bV4xQkbpCXQ653FJPLg-g0_R4Es"

REKOMENDASI = {
    "nitrogen": {
        "judul": "Kekurangan Nitrogen (N)",
        "deskripsi": "Daun menguning merata mulai dari bagian bawah tanaman, menandakan pasokan nitrogen untuk pertumbuhan vegetatif tidak mencukupi.",
        "solusi": "Berikan pupuk tinggi N seperti Urea atau ZA, atau tambahkan kompos organik secara berkala sesuai dosis anjuran.",
    },
    "posfor": {
        "judul": "Kekurangan Fosfor (P)",
        "deskripsi": "Pertumbuhan tanaman terhambat dengan daun berwarna hijau tua pekat hingga keunguan.",
        "solusi": "Aplikasikan pupuk SP-36, TSP, atau DAP di sekitar zona perakaran tanaman.",
    },
    "kalium": {
        "judul": "Kekurangan Kalium (K)",
        "deskripsi": "Pinggiran daun tampak seperti terbakar (scorching) atau berubah warna menjadi kecokelatan.",
        "solusi": "Tambahkan pupuk KCl atau ZK untuk memperkuat jaringan dan ketahanan tanaman.",
    },
    "kalsium": {
        "judul": "Kekurangan Kalsium (Ca)",
        "deskripsi": "Daun-daun muda tampak terdistorsi, menggulung, atau pertumbuhannya tidak normal.",
        "solusi": "Berikan kapur pertanian seperti Dolomit untuk memperbaiki pH dan kadar kalsium tanah.",
    },
    "magnesium": {
        "judul": "Kekurangan Magnesium (Mg)",
        "deskripsi": "Klorosis antar tulang daun — area daun menguning sementara tulang daun tetap berwarna hijau.",
        "solusi": "Berikan pupuk Magnesium Sulfat (Kieserit) atau Dolomit sesuai dosis anjuran untuk mengembalikan klorofil daun.",
    },
    "normal": {
        "judul": "Tanaman Normal & Sehat",
        "deskripsi": "Kondisi daun terlihat prima tanpa indikasi gejala defisiensi hara yang signifikan.",
        "solusi": "Lanjutkan pemeliharaan rutin dan jadwal pemupukan berimbang secara berkala.",
    },
}

# =========================================================
# 4. FUNGSI INTI
# =========================================================
def download_model_if_needed():
    """Unduh bobot model dari Google Drive jika belum ada secara lokal."""
    if MODEL_PATH.exists():
        return True
    try:
        with st.spinner("Mengunduh bobot model dari Google Drive (hanya sekali)..."):
            gdown.download(
                id=GDRIVE_FILE_ID,
                output=str(MODEL_PATH),
                quiet=False,
            )
        return MODEL_PATH.exists()
    except Exception as e:
        st.error(f"Gagal mengunduh model dari Google Drive: {e}")
        return False


@st.cache_resource(show_spinner="Memuat model Swin Transformer...")
def load_model():
    try:
        if not download_model_if_needed():
            return None
        model = timm.create_model(TIMM_MODEL_NAME, pretrained=False, num_classes=len(CLASSES))
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        return model
    except FileNotFoundError:
        st.error(
            f"File model **`{MODEL_PATH.name}`** tidak ditemukan di folder `{MODEL_PATH.parent}`.\n\n"
            "Pastikan GDRIVE_FILE_ID benar dan file dibagikan dengan akses "
            "\"Anyone with the link\", lalu muat ulang halaman."
        )
        return None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
        return None


def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return image, transform(image).unsqueeze(0)


def predict(model, image_tensor):
    start_time = time.time()
    with torch.no_grad():
        outputs = model(image_tensor.to(DEVICE))
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    inference_time = time.time() - start_time
    return probabilities.cpu(), inference_time


def confidence_level(conf: float):
    if conf >= 0.85:
        return "Tinggi", "#0f6b3f"
    elif conf >= 0.60:
        return "Sedang", "#a8710a"
    return "Rendah", "#d6453d"


def image_to_data_uri(image: Image.Image, max_width: int = 1000) -> str:
    img = image.copy()
    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, int(img.height * ratio)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


# =========================================================
# 5. SIDEBAR
# =========================================================
with st.sidebar:
    md(f"""
    <div class="nc-sidebar-brand">
        <div class="nc-header-icon">{icon('leaf')}</div>
        <span>NutriCocoa AI</span>
    </div>
    <div class="nc-sidebar-sub">Deteksi defisiensi hara daun kakao</div>
    """)

    uploaded_file = st.file_uploader("Unggah Citra Daun Kakao", type=["jpg", "jpeg", "png"])

    md("<div style='height:16px'></div>")
    md(f"""
    <div class="nc-info-box">
        <div class="nc-info-row"><span>Model</span><b>Swin Transformer</b></div>
        <div class="nc-info-row"><span>Arsitektur</span><b>{TIMM_MODEL_NAME}</b></div>
        <div class="nc-info-row"><span>Akurasi Validasi</span><b>{MODEL_VAL_ACCURACY*100:.1f}%</b></div>
        <div class="nc-info-row"><span>Jumlah Kelas</span><b>{len(CLASSES)}</b></div>
    </div>
    """)

# =========================================================
# 6. HEADER
# =========================================================
md(f"""
<div class="nc-header">
    <div class="nc-header-icon">{icon('leaf', '24px')}</div>
    <h1>NutriCocoa AI</h1>
</div>
<p class="nc-subtitle">
    Dashboard deteksi defisiensi unsur hara pada daun kakao menggunakan model Swin Transformer.
    Unggah citra daun melalui panel di samping untuk memulai analisis.
</p>
""")

model = load_model()

# =========================================================
# 7. KONTEN UTAMA
# =========================================================
if uploaded_file is None:
    md(f"""
    <div class="nc-card nc-empty-card">
        <div class="nc-empty-icon">{icon('upload', '22px')}</div>
        <div>
            <div class="nc-empty-title">Belum ada citra yang diunggah</div>
            <div class="nc-empty-desc">Silakan unggah citra daun kakao pada panel sebelah kiri untuk memulai deteksi.</div>
        </div>
    </div>
    """)

    md('<div class="nc-section-head"><h3>Cara Penggunaan</h3></div>')

    steps = [
        ("01", "Unggah Foto", "Ambil foto daun kakao dengan pencahayaan cukup dan fokus yang jelas."),
        ("02", "Model Menganalisis", "Model Swin Transformer memproses citra dan menghitung probabilitas tiap kelas hara."),
        ("03", "Lihat Rekomendasi", "Dapatkan diagnosis beserta rekomendasi pemupukan yang sesuai."),
    ]
    cols = st.columns(3)
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            md(f"""
            <div class="nc-step-card">
                <div class="nc-step-num">{num}</div>
                <div class="nc-step-title">{title}</div>
                <div class="nc-step-desc">{desc}</div>
            </div>
            """)

elif model is None:
    st.info("Lengkapi berkas model terlebih dahulu untuk menjalankan analisis.")

else:
    image, input_tensor = preprocess_image(uploaded_file)
    probs, inf_time = predict(model, input_tensor)
    top_idx = int(torch.argmax(probs).item())
    label = CLASSES[top_idx]
    confidence = float(probs[top_idx].item())
    level_label, level_color = confidence_level(confidence)
    reco = REKOMENDASI[label]
    is_normal = label == "normal"

    left, right = st.columns([1, 1.15], gap="large")

    with left:
        data_uri = image_to_data_uri(image)
        md(f"""
        <div class="nc-hero-wrap">
            <img src="{data_uri}" alt="Citra daun kakao yang diunggah">
            <div class="nc-hero-badge">{icon('camera', '14px')} {uploaded_file.name}</div>
        </div>
        """)

        md(f"""
        <div class="nc-reco {'success' if is_normal else ''}">
            <div class="nc-reco-head">{icon('check' if is_normal else 'alert', '18px')} Rekomendasi Tindakan</div>
            <div class="nc-reco-body">{reco['deskripsi']}</div>
            <div class="nc-reco-solution">
                <span class="lbl">Solusi</span>
                {reco['solusi']}
            </div>
        </div>
        """)

    with right:
        md(f"""
        <div class="nc-card">
            <div class="nc-pred-row">
                <div>
                    <div class="nc-label-caps">Hasil Prediksi</div>
                    <div class="nc-pred-title">{reco['judul']}</div>
                </div>
                <div class="nc-conf-badge">{confidence*100:.1f}%</div>
            </div>
            <div class="nc-conf-meta">
                <span>Tingkat Kepercayaan</span>
                <span style="font-weight:700; color:{level_color};">{level_label}</span>
            </div>
            <div class="nc-progress-track">
                <div class="nc-progress-fill" style="width:{confidence*100:.1f}%;"></div>
            </div>
            <div class="nc-inference-time">Waktu inferensi: {inf_time:.4f} detik</div>
        </div>
        """)

        md('<div class="nc-section-head"><h3>Distribusi Probabilitas</h3></div>')

        order = sorted(range(len(CLASSES)), key=lambda i: probs[i].item())
        y_labels = [CLASS_LABELS[CLASSES[i]] for i in order]
        x_vals = [probs[i].item() * 100 for i in order]
        bar_colors = [CLASS_COLORS[CLASSES[i]] for i in order]
        text_vals = [f"{v:.1f}%" for v in x_vals]

        fig = go.Figure(go.Bar(
            x=x_vals,
            y=y_labels,
            orientation="h",
            marker=dict(color=bar_colors, line=dict(width=0)),
            text=text_vals,
            textposition="outside",
            textfont=dict(family="Plus Jakarta Sans, sans-serif", color="#1a231d", size=14),
            cliponaxis=False,
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        ))
        fig.update_layout(
            height=260,
            margin=dict(l=4, r=40, t=8, b=8),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Plus Jakarta Sans, sans-serif", color="#1a231d", size=13),
            xaxis=dict(visible=False, range=[0, max(x_vals) * 1.22 if max(x_vals) > 0 else 100]),
            yaxis=dict(
                showgrid=False,
                tickfont=dict(family="Plus Jakarta Sans, sans-serif", color="#1a231d", size=13.5),
            ),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# =========================================================
# 8. FOOTER
# =========================================================
md('<div class="nc-footer">NutriCocoa AI · Dashboard Deteksi Hara Daun Kakao</div>')
