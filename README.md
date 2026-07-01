# NutriCocoa AI

Dashboard Deteksi Defisiensi Unsur Hara Daun Kakao menggunakan Swin Transformer (swin_tiny_patch4_window7_224).

## Features

- Upload gambar daun
- Prediksi 6 kelas
- Confidence Score
- Grafik Probabilitas
- Rekomendasi Pemupukan

## Bobot Model (.pth)

Berkas bobot model (`swin_kakao.pth`) **tidak** disimpan di repo ini karena
ukurannya melebihi batas file GitHub. Bobot di-hosting di Google Drive dan
akan otomatis diunduh oleh `app.py` (lewat `gdown`) saat aplikasi pertama
kali dijalankan, lalu di-cache oleh Streamlit (`@st.cache_resource`) sehingga
hanya diunduh sekali per sesi server.

ID file Google Drive diatur lewat variabel `GDRIVE_FILE_ID` di `app.py`.
Pastikan berkas di Drive dibagikan dengan akses **"Anyone with the link"**
(bukan private), jika tidak proses unduh otomatis akan gagal.

## Menjalankan Secara Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```
