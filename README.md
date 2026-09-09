# YOLOX-S — Workshop Tools

Implementasi ini melakukan fine-tuning YOLOX-S untuk tiga kelas alat pada dataset
lokal: `Wrenches`, `pliers`, dan `screwdriver`. Dataset memakai anotasi COCO, tetapi
folder gambarnya bernama `train`, `val`, dan `test`.

Struktur proyek:

```text
yolox-project/
|-- README.md
|-- requirements-workshop.txt
|-- YOLOX/                         # clone repositori YOLOX resmi
|   |-- .venv/                     # environment Python lokal
|   |-- tools/                     # train.py dan eval.py
|   |-- YOLOX_outputs/workshop_yolox_s/  # checkpoint dan log training
|-- exps/
|   |-- workshop_yolox_s.py         # konfigurasi model, dataset, training
|-- datasets/
|   |-- workshop_tools/
|       |-- annotations/
|       |   |-- instances_train.json
|       |   |-- instances_train.yolox.json
|       |   |-- instances_val.json
|       |   |-- instances_test.json
|       |-- train/                 # 360 gambar
|       |-- val/                   # 20 gambar
|       |-- test/                  # 10 gambar
|       |-- labels.json
|       |-- class_mapping.json
|       |-- preparation_report.json
|       |-- README.dataset.txt
|       |-- README.roboflow.txt
|-- models/                        # simpan bobot yolox_s.pth di sini
|-- scripts/
    |-- prepare_workshop.py
    |-- validate_dataset.py
    |-- build_training_annotations.py
    |-- predict.py
```

YOLOX menggunakan experiment Python di `exps/workshop_yolox_s.py` sebagai
konfigurasi, sehingga tidak memerlukan `coco.yaml`. Bobot YOLOX menggunakan `.pth`.
Nama split tetap `train`, `val`, dan `test`; tidak perlu menambahkan tahun COCO.
Path dataset default dihitung dari lokasi script, bukan direktori terminal.
`WORKSHOP_DATA_DIR` dapat digunakan untuk menunjuk dataset di lokasi lain.

Gunakan proyek ini untuk training dan editing. Pastikan checkpoint pretrained
`models/yolox_s.pth` tersedia sebelum training. Training dan evaluasi menggunakan
`tools/train.py` dan `tools/eval.py` dari repositori YOLOX resmi (lihat langkah di bawah).

Experiment memakai `instances_train.yolox.json`. Anotasi asli tidak ditimpa. Sebanyak
12 bbox train yang hanya berukuran 0,5–2 px pada salah satu sisi dihapus dari salinan
training; semuanya merupakan kotak tambahan yang nyaris berupa titik/garis pada gambar
yang masih memiliki bbox objek normal. Validation dan test tidak dibersihkan atau
diubah.

## 1. Validasi data

Pada penggunaan pertama, selesaikan bagian 2 untuk menyiapkan environment terlebih
dahulu. Kemudian jalankan sebelum training dari root proyek `yolox-project`:

```powershell
cd D:\repositories\yolox-project
.\YOLOX\.venv\Scripts\Activate.ps1
python -m pip install -r .\requirements-workshop.txt
python scripts/validate_dataset.py
python scripts/validate_dataset.py --train-ann instances_train.yolox.json
```

Perintah pertama mengaudit anotasi asli; perintah kedua memastikan anotasi yang benar-
benar dipakai training. Pemeriksaan mencakup JSON, gambar hilang/rusak, dimensi, bbox,
category mapping, file duplikat byte-identik, dan nama sumber lintas split.

Untuk membuat ulang anotasi training bersih:

```powershell
python scripts/build_training_annotations.py
```

## 2. Siapkan environment YOLOX resmi

YOLOX dapat dijalankan dengan Python 3.11.9. Disarankan memakai environment Python
3.11 terpisah. Pilih build PyTorch dan torchvision yang menyediakan wheel untuk Python
3.11 serta cocok dengan GPU/driver dari https://pytorch.org/get-started/locally/, lalu
clone dan install YOLOX resmi secara editable:

```powershell
cd D:\repositories\yolox-project
# Jalankan clone hanya jika folder YOLOX belum tersedia.
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
cd YOLOX
# Pada clone baru, terapkan perbaikan dependency dan loader checkpoint lokal.
git apply ../patches/yolox-local.patch
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel

# Pasang torch dan torchvision untuk Python 3.11 sesuai selector PyTorch terlebih dahulu.
# Training YOLOX ini membutuhkan GPU NVIDIA dan PyTorch dengan dukungan CUDA.
# Build CPU dapat digunakan untuk script inference lokal.
# Sebelum instalasi, ganti onnx-simplifier==0.4.10 di requirements.txt
# repositori YOLOX menjadi onnxsim==0.4.36 (lihat penjelasan di bawah).
python -m pip install --only-binary=onnxsim -r requirements.txt
python -m pip install --no-build-isolation -v -e .
python -m pip install -r "../requirements-workshop.txt"
```

Untuk Python 3.11 pada Windows x64, gunakan `onnxsim==0.4.36` sebagai pengganti
`onnx-simplifier==0.4.10` di `YOLOX/requirements.txt`. Salinan YOLOX lokal sudah
disesuaikan; lakukan penggantian ini juga jika clone ulang. Versi lama dapat gagal
saat build dengan pesan CMake tidak ditemukan atau `Invalid version: 'unknown'`.
Paket pengganti menyediakan wheel sehingga tidak perlu membangun ONNX Simplifier
dari source. `setup.py` YOLOX membaca requirements yang sama saat instalasi editable.
Lihat [wheel ONNX Simplifier 0.4.36](https://pypi.org/project/onnxsim/0.4.36/#files).

Folder `YOLOX/` tidak disertakan dalam Git proyek ini. Perubahan lokal disimpan di
`patches/yolox-local.patch`, dibuat terhadap commit upstream
`6ddff4824372906469a7fae2dc3206c7aa4bbaee`. Patch mencakup dependency ONNX dan
loader evaluasi untuk metadata NumPy pada checkpoint. Terapkan hanya pada clone
baru; salinan lokal saat ini sudah memuat perbaikan tersebut.

Pada Windows, pastikan interpreter yang aktif memang Python 3.11.9:

```powershell
python --version
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

Sebelum training, hasil `torch.cuda.is_available()` harus `True`. Versi dengan
akhiran `+cpu` dan `torch.version.cuda` bernilai `None` berarti build CPU terpasang.

YOLOX menggunakan dependency native seperti PyTorch, torchvision, OpenCV, dan
pycocotools. Karena itu, jangan memaksakan versi torch/torchvision dari contoh lama;
gunakan pasangan versi yang ditawarkan selector PyTorch untuk Python 3.11 dan platform
Anda. Jika instalasi YOLOX gagal saat membangun extension, periksa juga Visual Studio
Build Tools dan Ninja yang dibutuhkan oleh environment tersebut.

Contoh menggunakan root proyek `D:/repositories/yolox-project`, dengan repositori
YOLOX di dalamnya: `D:/repositories/yolox-project/YOLOX`. Jika proyek dipindahkan,
sesuaikan perintah `cd`. Path relatif pada contoh mengikuti susunan ini.

Jika muncul error `where cl`, instal workload **Desktop development with C++**
melalui Visual Studio Installer, termasuk MSVC x64/x86 dan Windows SDK. Buka
**x64 Native Tools Command Prompt for VS**, lalu periksa `where cl`.
Untuk menggunakan terminal VS Code, tutup semua jendela VS Code dan jalankan
perintah berikut dari terminal developer tersebut:

```bat
cd /d D:\repositories\yolox-project
code .
```

Di VS Code yang terbuka, gunakan terminal PowerShell baru dan aktifkan `.venv`
sesuai contoh. Periksa `where.exe cl` sebelum training. Semua blok `powershell`
di README menggunakan backtick sebagai penyambung baris; jangan beri spasi setelah
backtick atau menyalin tanda prompt `>>`.

Catat versi yang berhasil agar eksperimen dapat direproduksi:

```powershell
git rev-parse HEAD
python -m pip freeze > environment-lock.txt
```

Download checkpoint pretrained `yolox_s.pth` dari model zoo repositori resmi. Fine-
tuning dari bobot COCO direkomendasikan oleh dokumentasi YOLOX untuk custom dataset.

## 3. Training baseline

Jalankan dari folder `yolox-project/YOLOX`. Contoh satu GPU, total batch 8:

```powershell
cd D:\repositories\yolox-project\YOLOX
.\.venv\Scripts\Activate.ps1
$env:WORKSHOP_EVAL_SPLIT = "val"
python tools/train.py `
  -f "../exps/workshop_yolox_s.py" `
  -d 1 -b 8 --fp16 `
  -c "../models/yolox_s.pth"
```

Konfigurasi mempertahankan baseline resmi: 300 epoch, EMA, multi-scale, Mosaic, MixUp,
warmup 5 epoch, dan 15 epoch terakhir tanpa augmentasi kuat. Ini sengaja dijadikan
baseline terlebih dahulu. Dataset train asal Roboflow memang sudah memiliki augmentasi
offline; bila kurva validation menunjukkan overfit atau augmentasi gabungan terlalu
agresif, ubah satu faktor per eksperimen dan bandingkan AP validation.

Pedoman praktis:

- Batch rekomendasi awal adalah 8 per GPU; turunkan ke 4 atau 2 jika CUDA OOM. YOLOX
  menghitung learning rate dari total batch.
- Gunakan `--fp16` hanya pada CUDA. Jangan gunakan FP16 untuk CPU.
- Gunakan `--cache ram` hanya bila RAM cukup; caching bukan syarat training.
- Jangan memilih epoch atau threshold dari test set.
- Untuk smoke test pipeline, tambahkan override `max_epoch 1 data_num_workers 0`.

Dengan direktori kerja di atas, checkpoint terbaik tersimpan di:

```text
D:/repositories/yolox-project/YOLOX/YOLOX_outputs/workshop_yolox_s/best_ckpt.pth
```

## 4. Evaluasi

Pilih checkpoint berdasarkan validation:

```powershell
cd D:\repositories\yolox-project\YOLOX
.\.venv\Scripts\Activate.ps1
$env:WORKSHOP_EVAL_SPLIT = "val"
python tools/eval.py `
  -f "../exps/workshop_yolox_s.py" `
  -c "YOLOX_outputs/workshop_yolox_s/best_ckpt.pth" `
  -d 1 -b 8 --conf 0.001 --fp16 --fuse
```

Setelah model dan semua keputusan final terkunci, evaluasi test satu kali:

```powershell
$env:WORKSHOP_EVAL_SPLIT = "test"
python tools/eval.py `
  -f "../exps/workshop_yolox_s.py" `
  -c "YOLOX_outputs/workshop_yolox_s/best_ckpt.pth" `
  -d 1 -b 8 --conf 0.001 --fp16 --fuse
$env:WORKSHOP_EVAL_SPLIT = "val"
```

Jangan tambahkan `--test` atau `--testdev`; file test lokal memiliki ground truth dan
harus melalui evaluator biasa.

## 5. Inference gambar

Demo resmi YOLOX memakai nama kelas COCO secara default. Gunakan script lokal agar
hasil menampilkan tiga label workshop yang benar. Jalankan dari root proyek.
Contoh berikut memakai gambar test sebagai demonstrasi, bukan untuk memilih threshold:

```powershell
cd D:\repositories\yolox-project
.\YOLOX\.venv\Scripts\Activate.ps1
python scripts/predict.py `
  "datasets/workshop_tools/test" `
  --checkpoint "YOLOX/YOLOX_outputs/workshop_yolox_s/best_ckpt.pth" `
  --device cuda --fp16 --fuse `
  --conf 0.25 --nms 0.45 `
  --output "predictions"
```

Ganti `datasets/workshop_tools/test` dengan file atau folder gambar Anda.
Hasil contoh disimpan di `D:/repositories/yolox-project/predictions`.
Path checkpoint pada inference diawali `YOLOX/` karena direktori kerja berbeda
dari perintah training dan evaluasi.

Output berisi gambar beranotasi serta `predictions.json` dengan label, confidence, bbox
`xyxy`, dan waktu inference. Untuk CPU gunakan `--device cpu` tanpa `--fp16`.

## Kondisi dan batasan data

- Split sumber dipertahankan: train 360, val 20, test 10; tidak ada resplit.
- Semua gambar berukuran 640×640 hasil resize stretch dari Roboflow.
- Tidak ditemukan gambar byte-identik atau nama sumber yang bocor antar-split.
- Distribusi bbox train relatif seimbang: 135 Wrenches, 162 pliers, 174 screwdriver
  setelah 12 artefak Wrenches dibuang.
- Tujuh bbox train mencakup setidaknya 99% gambar dan perlu inspeksi manual lanjutan.
- Tidak ada negative/background-only image. Tambahkan contoh latar deployment tanpa
  ketiga alat untuk mengukur dan menekan false positive.
- Test hanya 10 gambar, jadi AP test memiliki variance tinggi dan belum layak dianggap
  estimasi performa produksi.
- Dataset berlisensi CC BY 4.0; pertahankan atribusi dari `README.dataset.txt`.

## Referensi

- Implementasi resmi: https://github.com/Megvii-BaseDetection/YOLOX
- Custom training: https://github.com/Megvii-BaseDetection/YOLOX/blob/main/docs/train_custom_data.md
- Paper YOLOX: https://arxiv.org/abs/2107.08430
- Dataset: https://universe.roboflow.com/yj-ebtzo/workshop-tools-3/dataset/2
