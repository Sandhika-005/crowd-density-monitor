# Crowd counting

Kode dipisahkan dari notebook crowdcounting.ipynb. Semua file harus berada dalam satu folder. Python 3.10 atau lebih baru.

## File

- `config.json`: path dataset, pembagian validasi, dan pengaturan training.
- `process_dataset.py`: membaca label MATLAB ShanghaiTech, membuat density map, menyimpan manifest train/valid/test.
- `dataset.py`: dataset dan normalisasi ImageNet.
- `model.py`: arsitektur notebook dengan enam convolution backend 512 channel.
- `train.py`: training, pemilihan checkpoint berdasarkan MAE real count validasi, evaluasi test sekali setelah training, dan grafik.
- `inference_utils.py`: prediksi dan visualisasi bersama.
- `inference_photo.py`: inference foto dan heatmap.
- `inference_video.py`: inference setiap frame, video beranotasi, dan CSV hitungan per frame.

## Jalankan

Buka terminal di folder ini. Pasang PyTorch dan torchvision yang sesuai perangkat; requirements dapat dipakai pada environment baru:

```bash
pip install -r requirements.txt
```

Edit `train_path` dan `test_path` pada `config.json`. Setiap path berisi folder `images` dan `ground-truth`. Path relatif dalam config dihitung dari lokasi config. Jalankan:

```bash
python process_dataset.py --config config.json
python train.py --config config.json
python inference_photo.py --image foto.jpg --checkpoint outputs/best_csrnet.pth
python inference_video.py --video video.mp4 --checkpoint outputs/best_csrnet.pth
```

Jika output_dir diubah, sesuaikan --checkpoint juga. Inferensi tidak mengunduh bobot VGG. Training pertama dengan pretrained=true membutuhkan unduhan bobot ImageNet atau cache torchvision yang tersedia.

## Hasil

`outputs/` berisi best_csrnet.pth, last_csrnet.pth, training_history.csv, training_curves.png, valid_predictions.csv, test_predictions.csv, metrics.json, dan salinan config. Training menggunakan batch size 1 dan MSE mean terhadap density map. MAE/RMSE dihitung dari jumlah prediksi dibanding jumlah titik anotasi; count tidak dibulatkan atau dipotong saat evaluasi.

Dataset training dibagi deterministik 80:20 sesuai config; pembagian ini disimpan dalam manifest sehingga tidak berubah saat training ulang. Kepadatan dihitung pada resolusi 1/8, setiap Gaussian dinormalisasi menjadi 1, dan titik tepi dijepit ke batas map agar tidak hilang. Manifest menggunakan path absolut; jalankan preprocessing lagi setelah memindahkan dataset.

Arsitektur dipertahankan dari notebook, bukan diganti ke CSRNet paper. Checkpoint lama dengan key model_state dan arsitektur notebook dapat dipakai untuk inference. Checkpoint untuk backend CSRNet asli tidak kompatibel.

Heatmap ditampilkan setelah nilai negatif dipotong hanya untuk visualisasi. Hitungan tetap menggunakan jumlah output mentah model. Video menghitung orang pada masing-masing frame, bukan total pengunjung unik. Audio tidak disalin ke video hasil. Inference menggunakan resolusi asli tanpa resize; resolusi tinggi membutuhkan memori lebih besar. Model Part B dapat kurang akurat pada sudut kamera atau lokasi baru.

Kode sudah diperiksa sintaksnya dan generator density diuji dengan titik kosong, bertumpuk, serta di tepi. Training GPU dan inference end-to-end belum dijalankan di lingkungan pembuatan file karena PyTorch/OpenCV tidak tersedia.

## Demo Streamlit

```bash
pip install -r requirements.txt
streamlit run app.py
```

Isi lokasi checkpoint di sidebar, pilih CPU/GPU, lalu upload gambar atau video dan klik Proses. Checkpoint default: outputs/best_csrnet.pth. File checkpoint disediakan lokal; gambar/video dapat diupload lewat antarmuka.

Gambar menghasilkan estimasi count, overlay heatmap, gambar unduhan, dan density NPY. Video menghasilkan preview selama proses, MP4 H.264 untuk browser, grafik count terhadap waktu, serta CSV setiap frame. File sementara dibersihkan setelah pemrosesan. Video hasil tidak memiliki audio dan tidak menghitung pengunjung unik. Pemrosesan semua frame bisa lama untuk video panjang. Batas upload default Streamlit adalah 200 MB; untuk mengubahnya:

```bash
streamlit run app.py --server.maxUploadSize 500
```

Download hasil sebelum mengganti input; tombol download tidak menjalankan ulang inference. Checkpoint bobot model tidak termasuk dalam paket. Model dipertahankan dari notebook agar checkpoint lama kompatibel. Kode UI diperiksa sintaksnya; demo end-to-end belum dijalankan karena PyTorch, OpenCV, dan Streamlit tidak tersedia pada lingkungan pembuatan file.
