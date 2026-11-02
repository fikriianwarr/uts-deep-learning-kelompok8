# Klasifikasi Tutupan Lahan Citra Satelit

Proyek UTS Pembelajaran Mendalam Kelompok 8, semester ganjil 2026/2027. Proyek ini membandingkan CNN dasar dan EfficientNet-B0 untuk mengklasifikasikan sepuluh kelas tutupan lahan pada citra EuroSAT RGB. Aplikasi Gradio menampilkan probabilitas kelas, visualisasi Grad-CAM, dan keputusan penerimaan berdasarkan confidence threshold.

## Anggota kelompok

| Nama | NIM |
|---|---|
| Jordanniel Saor Sihombing | 2311120148 |
| Jordani Brema Barus | 231112650 |
| Kevin Rorista Kaban | 231112119 |
| Fadillah Rasyd Barus | 231112885 |
| M. Alfi Syahrin | 231112853 |
| Fikri Anwar | 231112145 |

## Dataset dan metode

[EuroSAT RGB](https://zenodo.org/records/7711810) berisi 27.000 citra Sentinel-2 berukuran 64×64 piksel. Sepuluh kelasnya adalah AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, dan SeaLake.

Data dibagi secara stratified dengan seed 42 menjadi 18.900 citra training, 4.050 validation, dan 4.050 testing. Manifest menyimpan pembagian serta fingerprint isi data. Audit dataset tidak menemukan berkas rusak atau duplikat byte-identik.

Kedua model menggunakan input 64×64 dan normalisasi ImageNet. Flip horizontal, flip vertikal, serta rotasi ±20° hanya diterapkan pada training. CNN dasar memiliki 316.970 parameter. EfficientNet-B0 memiliki 4.020.358 parameter dan menggunakan bobot ImageNet, pelatihan head, kemudian fine-tuning. Checkpoint dipilih berdasarkan macro-F1 validation; threshold juga ditentukan dari validation sebelum evaluasi test.

Konfigurasi lengkap tersedia pada `experiment_config.json`.

## Hasil evaluasi

Hasil berikut berasal dari 4.050 citra test pada pembagian yang disimpan dalam repository.

| Model | Accuracy test | Macro-F1 test |
|---|---:|---:|
| CNN dasar | 88,07% | 0,8807 |
| EfficientNet-B0 | 94,40% | 0,9431 |

Threshold confidence sebesar 0,992040 dipilih dengan coverage validation minimum 70%. Pada test, 2.825 prediksi diterima dan 1.225 ditolak. Selective accuracy pada prediksi yang diterima adalah 99,61%, dengan coverage 69,75%. Nilai selective accuracy menggunakan denominator yang berbeda dari accuracy seluruh test.

Hasil per kelas, confusion matrix, learning curves, probabilitas, serta checkpoint tersedia pada `artifacts/`. Hash checkpoint diperiksa saat inferensi agar model dan threshold tetap sesuai.

## Struktur repository

| Lokasi | Isi |
|---|---|
| `app.py` | Antarmuka Gradio |
| `src/` | Preprocessing, model, training, evaluasi, Grad-CAM, threshold, dan inferensi |
| `scripts/` | Eksekusi eksperimen, pembuatan diagram, pengujian aplikasi, dan ekspor paket hosting |
| `notebooks/` | Notebook eksperimen beserta output |
| `artifacts/` | Checkpoint, manifest, hasil numerik, dan grafik |
| `examples/` | Tiga contoh citra dari data test |
| `tests/` | Pengujian preprocessing, integritas data, model, dan threshold |

## Menjalankan aplikasi

Gunakan Python 3.12. Contoh berikut menggunakan PowerShell pada root proyek.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-deploy.txt -r requirements-dev.txt
.\.venv\Scripts\python.exe app.py
```

Buka `http://127.0.0.1:7860` dan pilih gambar pada bagian contoh atau unggah citra RGB. Checkpoint yang diperlukan aplikasi sudah disertakan. Versi lingkungan eksperimen dicatat pada `requirements-validated.txt`.

| Contoh | Label asli | Prediksi | Keputusan |
|---|---|---|---|
| `correct_Residential_1033.jpg` | Residential | Residential | Diterima |
| `wrong_HerbaceousVegetation_2664.jpg` | HerbaceousVegetation | Highway | Diterima |
| `rejected_Pasture_1451.jpg` | Pasture | Pasture | Ditolak |

Contoh salah yang tetap diterima memperlihatkan bahwa confidence tinggi belum menjamin prediksi benar. Persentase pada diagram dibulatkan; nilai lebih rinci ditampilkan pada status prediksi.

## Menjalankan eksperimen

Dataset diunduh dari sumber resminya dan disimpan sebagai `data/EuroSAT_RGB.zip`. Folder data tidak disertakan dalam repository.

Buka `notebooks/PROYEK_UTS_DEEP_LEARNING_KELOMPOK_8.ipynb` melalui Jupyter atau Google Colab dan jalankan section 1–10 secara berurutan. Notebook memuat modul pendukung agar dapat dijalankan di Colab dengan ZIP dataset. Pada salinan repository lokal, notebook memakai modul yang tersedia di proyek.

Eksperimen juga dapat dijalankan melalui terminal:

```powershell
.\.venv\Scripts\python.exe scripts/run_experiment.py
```

Hasil training yang sudah tersimpan digunakan kembali setelah konfigurasi dan hash checkpoint diperiksa. Pelatihan baru membutuhkan koneksi internet untuk mengambil bobot ImageNet. Untuk eksperimen dengan konfigurasi berbeda, gunakan salinan kerja dan direktori hasil terpisah; test tidak digunakan untuk memilih ulang model atau threshold.

## Pengujian dan ekspor aplikasi

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/check_app_integration.py
.\.venv\Scripts\python.exe scripts/package_for_space.py
```

Paket aplikasi dihasilkan pada `deploy/space_package/`, berisi aplikasi, dependensi, modul inferensi, checkpoint, threshold, dan contoh citra. Folder hasil ekspor ini dikecualikan dari Git. Untuk Gradio Space, isi folder tersebut ditempatkan pada root Space dengan konfigurasi dari README hasil ekspor. Ketersediaan hosting mengikuti ketentuan layanan dan akun yang digunakan.

## Keterbatasan

Eksperimen memakai satu seed dan pembagian acak berdasarkan berkas; hasilnya belum membuktikan generalisasi geografis. Audit hash mendeteksi duplikat identik, tetapi tidak meniadakan kemiripan antarpatch. Grad-CAM bersifat kualitatif, dengan feature map akhir 2×2 pada input 64×64. Pembesaran visual tidak menambah ketepatan lokalisasi. Confidence threshold menukar cakupan dengan ketepatan pada prediksi yang diterima dan belum divalidasi sebagai pendeteksi gambar di luar domain EuroSAT.

## Referensi

1. P. Helber, B. Bischke, A. Dengel, dan D. Borth, “EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification,” 2019. https://doi.org/10.1109/JSTARS.2019.2918242
2. P. Helber dkk., EuroSAT dataset versi 2. https://zenodo.org/records/7711810
3. M. Tan dan Q. V. Le, “EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,” ICML, 2019. https://proceedings.mlr.press/v97/tan19a.html
4. R. R. Selvaraju dkk., “Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization,” ICCV, 2017. https://doi.org/10.1109/ICCV.2017.74
