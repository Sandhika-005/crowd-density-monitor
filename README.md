# CrowdScope — Crowd Counting & Density Monitor

> Proyek computer vision berbiaya rendah untuk menghitung orang dari webcam atau video, memperkirakan tingkat kepadatan area, dan mencatat perubahan keramaian dari waktu ke waktu.

**Nama repository GitHub yang disarankan:** `crowd-density-monitor`  
**Nama aplikasi:** `CrowdScope — Crowd Counting & Density Monitor`

CrowdScope dirancang untuk berbagai lokasi, bukan hanya satu tempat. Contoh penerapan: area kampus, kantin, perpustakaan, mal, halte, ruang tunggu, tempat acara, area antrean, pintu masuk gedung, atau ruang publik lain yang penggunaannya diizinkan.

## 1. Masalah yang ingin diselesaikan

Penghitungan orang secara manual memerlukan waktu dan bisa menjadi tidak konsisten ketika area ramai. CrowdScope menggunakan kamera untuk memperkirakan jumlah orang yang tampak pada frame, mengklasifikasikan tingkat kepadatan berdasarkan kapasitas yang dikonfigurasi, dan menyimpan log untuk analisis.

Contoh alur: kamera di area pilihan → deteksi orang → hitung orang dalam area pantauan → tampilkan status kepadatan → simpan data tren → bantu pengelola memahami waktu atau area yang ramai.

## 2. Fitur versi awal

- Input dari webcam laptop/USB atau file video.
- Deteksi kelas `person` menggunakan model YOLO pretrained.
- Overlay jumlah orang, persentase kapasitas, status kepadatan, FPS, dan peringatan kapasitas.
- Area penghitungan opsional (*region of interest*/ROI) yang dapat ditentukan di `config.yaml`.
- Nama lokasi/area dapat diatur pada konfigurasi atau lewat argumen `--location`.
- Pencatatan CSV yang menyimpan waktu, nama lokasi, jumlah orang, kapasitas, status kepadatan, dan FPS.
- Dashboard Streamlit untuk melihat tren jumlah orang, distribusi kepadatan, dan jam yang lebih sepi.
- Ambang kepadatan, kapasitas, confidence, dan perangkat inferensi bisa dikonfigurasi.

## 3. Batasan penting

Versi awal ini memakai **deteksi orang satu per satu**, bukan *density estimation*. Metode ini cocok sebagai baseline untuk kerumunan yang orang-orangnya masih cukup terlihat. Ketika banyak orang saling menutupi, kepala sangat kecil di gambar, atau kamera terlalu jauh/tinggi, model dapat melewatkan beberapa orang.

Angka yang tampil adalah **jumlah orang yang terdeteksi pada frame saat itu**, bukan jumlah orang unik sepanjang hari. Untuk menghitung pengunjung unik atau arah masuk/keluar, diperlukan object tracking dan perancangan yang sesuai. Untuk kerumunan sangat padat, pertimbangkan model density-estimation yang dilatih dan dievaluasi pada dataset crowd counting.

Status “mendekati kapasitas” atau “melebihi kapasitas” ditentukan dari angka kapasitas yang kamu tetapkan. Nilai tersebut bukan standar keselamatan resmi dan tidak boleh menjadi satu-satunya dasar keputusan keselamatan.

## 4. Struktur proyek

```text
crowd-density-monitor/
├── README.md
├── requirements.txt
├── config.yaml
├── crowd_counter.py
├── crowd_utils.py
├── dashboard.py
├── .gitignore
├── data/                  # taruh video uji yang diizinkan di sini
├── outputs/               # log CSV/video anotasi
├── docs/DATASET_GUIDE.md
└── tests/test_density.py
```

## 5. Persyaratan

- Python 3.10–3.13, bergantung pada ketersediaan wheel paket untuk sistemmu.
- Webcam atau file video.
- Koneksi internet pada instalasi/pemakaian pertama untuk mengunduh paket dan bobot YOLO.
- Laptop tanpa GPU dapat menggunakan `device: cpu`; FPS bisa lebih rendah.

## 6. Instalasi di Windows (PowerShell)

Ekstrak ZIP, buka PowerShell di folder proyek, lalu jalankan:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Jika PowerShell menolak aktivasi environment, coba memakai executable virtual environment secara langsung: ` .\.venv\Scripts\python.exe `.

## 7. Cara menjalankan

### Webcam

```powershell
python crowd_counter.py --source 0 --location "Area Kampus"
```

`--location` adalah nama area yang ingin dicatat. Ganti dengan nama seperti `Perpustakaan`, `Pintu Masuk Gedung A`, `Ruang Tunggu`, atau `Area Antrean`. Kalau tidak diberikan, program memakai `location_name` dari `config.yaml`.

Tekan `q` pada jendela video untuk berhenti. Log secara default ditambahkan ke `outputs/crowd_log.csv`.

### File video

Taruh video yang diizinkan di folder `data/`, lalu jalankan:

```powershell
python crowd_counter.py --source "data/example.mp4" --location "Area Uji" --save-video
```

Video hasil anotasi disimpan ke `outputs/annotated_crowd.mp4` secara default.

### Ubah konfigurasi lokasi/kapasitas

Edit `config.yaml`:

```yaml
location_name: "Area Pantauan"
capacity: 60
```

Kapasitas sebaiknya disesuaikan dengan informasi yang relevan untuk lokasi dan tujuan pengamatan. Jika ingin memantau area lain, ubah `location_name` dan `capacity` atau gunakan `--location` / `--capacity` saat menjalankan.

### Dashboard tren

Setelah program menghasilkan log, jalankan:

```powershell
streamlit run dashboard.py
```

Dashboard membaca `outputs/crowd_log.csv` secara default. Jika CSV memuat beberapa lokasi, pilih lokasi lewat filter sidebar. Kamu juga dapat mengunggah CSV yang sesuai. Agar tren waktu ramai/sepi lebih bermakna, kumpulkan data pada beberapa rentang waktu dan beberapa hari. Jika menggunakan video lama, timestamp pada log adalah waktu saat program dijalankan, bukan waktu asli pengambilan video.

## 8. Area penghitungan (ROI)

Secara default seluruh frame dihitung. Untuk membatasi penghitungan pada area tertentu, edit `roi_normalized` di `config.yaml` sebagai titik poligon dengan koordinat ternormalisasi dari `0.0` sampai `1.0`:

```yaml
roi_normalized:
  - [0.10, 0.20]
  - [0.90, 0.20]
  - [0.95, 0.95]
  - [0.05, 0.95]
```

Tentukan titik mengikuti keliling area dalam tampilan kamera. ROI perlu disesuaikan dengan sudut dan resolusi kamera aktual.

## 9. Pengujian

```powershell
python -m unittest discover -s tests -v
```

Unit test tidak memerlukan kamera atau model YOLO. Untuk mengukur kinerja model secara nyata:

1. Siapkan klip video yang penggunaan/rekamannya diizinkan.
2. Tentukan jumlah orang secara manual pada timestamp tertentu sebagai hitungan referensi.
3. Bandingkan hitungan model dan referensi pada timestamp yang sama.
4. Hitung MAE (*mean absolute error*), catat FPS, pencahayaan, sudut kamera, dan jumlah kasus orang tertutup.
5. Laporkan metrik aktual; jangan mengisi hasil pengujian dengan angka perkiraan.

## 10. Dataset untuk pengembangan lanjutan

Baca [`docs/DATASET_GUIDE.md`](docs/DATASET_GUIDE.md). Dataset crowd counting sering menggunakan anotasi titik kepala atau density map, sedangkan YOLO person detection membutuhkan bounding box. Format tersebut tidak dapat dipertukarkan langsung tanpa metode atau konversi anotasi yang sesuai.

## 11. Privasi dan etika

- Gunakan video/kamera dengan izin yang sesuai dan perhatikan aturan lokasi.
- Beri pemberitahuan apabila dibutuhkan oleh institusi atau pengelola area.
- Proyek ini menghitung orang; tidak perlu mengenali identitas wajah.
- Simpan rekaman mentah sesedikit mungkin. Log CSV hanya menyimpan angka dan metadata operasional.
- Jangan commit video, frame, dataset, atau data berisi orang ke repository publik tanpa izin yang sesuai.
- Jangan gunakan status kepadatan sederhana sebagai satu-satunya dasar keputusan darurat.

## 12. Referensi teknis

- Ultralytics Quickstart: <https://docs.ultralytics.com/quickstart/>
- PyTorch — instalasi lokal: <https://pytorch.org/get-started/locally/>
- UCSD Pedestrian dataset/project: <https://www.svcl.ucsd.edu/projects/peoplecnt/>
- Mall Dataset: <https://staff.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html>
- UCF-CC-50: <https://www.crcv.ucf.edu/data/ucf-cc-50/>

Periksa lisensi paket, model, dan dataset sebelum distribusi atau pemakaian komersial. Ultralytics memiliki ketentuan lisensi tersendiri; baca <https://www.ultralytics.com/license>.
