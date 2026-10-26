# Crowd counting

Proyek crowd counting berbasis computer vision dan deep learning untuk memperkirakan jumlah orang dalam gambar atau video menggunakan density map estimation. Model mengadaptasi arsitektur CSRNet dengan frontend VGG16 dan backend dilated convolution. Hasil prediksi dapat dilihat melalui aplikasi web Streamlit atau inferensi terminal, berupa estimasi jumlah orang dan visualisasi density heatmap. Pada video, tersedia pula grafik dan CSV estimasi per frame.

Kode dipisahkan dari notebook `crowdcounting.ipynb`. Semua file utama Python dan konfigurasi harus berada dalam satu folder proyek. Gunakan Python 3.10 atau lebih baru.

## File

- `config.json`: path dataset, pembagian validasi, dan pengaturan training.
- `process_dataset.py`: membaca label MATLAB ShanghaiTech, membuat density map, menyimpan manifest train/valid/test.
- `dataset.py`: dataset dan normalisasi ImageNet.
- `model.py`: arsitektur notebook dengan enam convolution backend 512 channel.
- `train.py`: training, pemilihan checkpoint berdasarkan MAE real count validasi, evaluasi test sekali setelah training, dan grafik.
- `inference_utils.py`: prediksi dan visualisasi bersama.
- `inference_photo.py`: inference foto dan heatmap.
- `inference_video.py`: inference setiap frame, video beranotasi, dan CSV hitungan per frame.
- `app.py`: antarmuka web Streamlit untuk upload gambar/video, menjalankan prediksi, menampilkan heatmap, grafik, serta mengunduh hasil.
- `requirements.txt`: daftar library Python yang diperlukan untuk menjalankan project.

## Kebutuhan

- Python 3.10 atau lebih baru dan `pip`.
- Library sesuai `requirements.txt`, termasuk PyTorch, torchvision, OpenCV, Streamlit, NumPy, SciPy, Pillow, pandas, dan matplotlib.
- CPU dapat digunakan; GPU NVIDIA dengan CUDA bersifat opsional untuk mempercepat pemrosesan, jika instalasi PyTorch mendukungnya.
- Dataset ShanghaiTech Part B untuk preprocessing dan training, atau checkpoint `best_csrnet.pth` yang kompatibel jika hanya menjalankan inferensi.
- Ruang penyimpanan untuk dataset, checkpoint, dan hasil pemrosesan. Gambar beresolusi tinggi dan video panjang membutuhkan memori serta waktu pemrosesan lebih besar.

## Jalankan

Buka terminal di folder proyek. Jika mengambil source code dari GitHub:

```bash
git clone https://github.com/Sandhika-005/crowd-density-monitor.git
cd crowd-density-monitor
```

Pasang PyTorch dan torchvision yang sesuai perangkat; requirements dapat dipakai pada environment baru:

```bash
pip install -r requirements.txt
```

Edit `train_path` dan `test_path` pada `config.json`. Setiap path berisi folder `images` dan `ground-truth`. Path relatif dalam config dihitung dari lokasi config. Dataset yang digunakan adalah ShanghaiTech Part B; sumber dataset: https://www.kaggle.com/datasets/tthien/shanghaitech.

Jalankan preprocessing, training, atau inferensi sesuai kebutuhan:

```bash
python process_dataset.py --config config.json
python train.py --config config.json
python inference_photo.py --image foto.jpg --checkpoint outputs/best_csrnet.pth
python inference_video.py --video video.mp4 --checkpoint outputs/best_csrnet.pth
```

Jalankan preprocessing dan training terlebih dahulu jika checkpoint belum tersedia. Jika sudah memiliki checkpoint kompatibel, inferensi bisa langsung dijalankan tanpa mengulang training. Pastikan `foto.jpg` atau `video.mp4` sesuai dengan lokasi file input.

Jika `output_dir` diubah, sesuaikan `--checkpoint` juga. Inferensi tidak mengunduh bobot VGG. Training pertama dengan `pretrained=true` membutuhkan unduhan bobot ImageNet atau cache torchvision yang tersedia. File checkpoint bobot model tidak termasuk dalam paket source code; checkpoint harus dibuat melalui training atau disediakan terpisah dengan arsitektur yang sesuai.

## Hasil

`outputs/` berisi `best_csrnet.pth`, `last_csrnet.pth`, `training_history.csv`, `training_curves.png`, `valid_predictions.csv`, `test_predictions.csv`, `metrics.json`, dan salinan config setelah training selesai. Training menggunakan batch size 1 dan MSE mean terhadap density map. MAE/RMSE dihitung dari jumlah prediksi dibanding jumlah titik anotasi; count tidak dibulatkan atau dipotong saat evaluasi.

Inferensi gambar melalui `inference_photo.py` menghasilkan gambar dengan overlay heatmap, file `.npy` berisi density map, dan file `.json` berisi estimasi jumlah orang. Sementara itu, `inference_video.py` menghasilkan video beranotasi dan file `.csv` berisi estimasi jumlah orang pada setiap frame beserta waktunya.

Dataset training dibagi deterministik 80:20 sesuai config; dari 400 gambar train awal ShanghaiTech Part B diperoleh 320 train dan 80 validasi, sedangkan 316 gambar test tetap dipisahkan. Pembagian ini disimpan dalam manifest sehingga tidak berubah saat training ulang. Kepadatan dihitung pada resolusi 1/8, setiap Gaussian dinormalisasi menjadi 1, dan titik tepi dijepit ke batas map agar tidak hilang. Manifest menggunakan path absolut; jalankan preprocessing lagi setelah memindahkan dataset.

Arsitektur dipertahankan dari notebook, bukan diganti ke CSRNet paper. Checkpoint lama dengan key `model_state` dan arsitektur notebook dapat dipakai untuk inference. Checkpoint untuk backend CSRNet asli tidak kompatibel.

Heatmap ditampilkan setelah nilai negatif dipotong hanya untuk visualisasi. Hitungan tetap menggunakan jumlah output mentah model. Video menghitung orang pada masing-masing frame, bukan total pengunjung unik. Audio tidak disalin ke video hasil. Inference menggunakan resolusi asli tanpa resize; resolusi tinggi membutuhkan memori lebih besar. Model Part B dapat kurang akurat pada sudut kamera, pencahayaan, atau lokasi baru.

Kode telah diperiksa sintaksnya dan generator density diuji dengan titik kosong, bertumpuk, serta di tepi. Laporan progres juga mendokumentasikan pengujian awal aplikasi dengan enam gambar dan dua video. Beberapa hasil menunjukkan overestimation; pengukuran akurasi secara menyeluruh masih memerlukan perbandingan dengan jumlah orang sebenarnya pada data uji yang telah diverifikasi.

## Demo Streamlit

```bash
pip install -r requirements.txt
streamlit run app.py
```

Isi lokasi checkpoint di sidebar, pilih CPU/GPU, lalu upload gambar atau video dan klik Proses. Checkpoint default: `outputs/best_csrnet.pth`. File checkpoint disediakan secara lokal; gambar/video dapat diupload lewat antarmuka.

### Cara Menggunakan Aplikasi

1. Jalankan aplikasi menggunakan `streamlit run app.py`.
2. Masukkan lokasi checkpoint model pada sidebar.
3. Pilih perangkat CPU atau GPU yang tersedia.
4. Pilih jenis input berupa gambar atau video.
5. Upload file yang ingin dianalisis.
6. Klik tombol Proses untuk menjalankan prediksi.
7. Lihat estimasi jumlah orang dan visualisasi density heatmap.
8. Unduh hasil pemrosesan jika diperlukan.

### Format Input yang Didukung

- Gambar: JPG, JPEG, PNG, dan WEBP.
- Video: MP4, MOV, AVI, dan MKV.

Aplikasi memproses gambar sebagai satu citra, sedangkan video dianalisis frame demi frame menggunakan model yang sama. Berkas anotasi `.mat` tidak diperlukan untuk inferensi karena digunakan pada tahap persiapan dataset training.

Gambar menghasilkan estimasi count, overlay heatmap, gambar unduhan, dan density NPY. Video menghasilkan preview selama proses, MP4 H.264 untuk browser, grafik count terhadap waktu, serta CSV setiap frame. File sementara dibersihkan setelah pemrosesan. Video hasil tidak memiliki audio dan tidak menghitung pengunjung unik. Pemrosesan semua frame bisa lama untuk video panjang. Batas upload default Streamlit adalah 200 MB; untuk mengubahnya:

```bash
streamlit run app.py --server.maxUploadSize 500
```

Download hasil sebelum mengganti input; tombol download tidak menjalankan ulang inference. Checkpoint bobot model tidak termasuk dalam paket. Model dipertahankan dari notebook agar checkpoint lama kompatibel. Pengujian awal Streamlit telah didokumentasikan dalam laporan progres, tetapi hasil prediksi belum tentu akurat pada semua jenis kerumunan.

## Repository

Source code dan dokumentasi proyek: https://github.com/Sandhika-005/crowd-density-monitor

Pastikan repository dapat diakses oleh dosen, seluruh source code dan `requirements.txt` tersedia, serta checkpoint yang diperlukan untuk demo dapat diperoleh atau dibuat dengan mengikuti instruksi training di atas. Jika checkpoint disimpan di luar repository, cantumkan tautan unduh dan petunjuk penempatannya di sini sebelum pengumpulan.
