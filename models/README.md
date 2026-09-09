Simpan checkpoint pretrained YOLOX-S (`yolox_s.pth`) di folder ini.
Jika belum tersedia, unduh bobot sesuai petunjuk di [README utama](../README.md).

Checkpoint hasil training disimpan terpisah di
`../YOLOX/YOLOX_outputs/workshop_yolox_s/best_ckpt.pth`.
Gunakan checkpoint hasil training tersebut untuk evaluasi dan prediksi kelas workshop.

`best_ckpt.pth` di folder ini adalah salinan identik checkpoint hasil training
tersebut, disertakan di GitHub agar dapat diunduh untuk latihan integrasi kamera.
File ini menyimpan bobot model, state optimizer, dan metadata training.
Gunakan `scripts/predict.py` untuk memuatnya dengan dukungan metadata NumPy.

```powershell
python scripts/predict.py datasets/workshop_tools/test --checkpoint models/best_ckpt.pth --device cuda --output predictions
```

`workshop_yolox_s.pth` adalah salinan bobot model dari checkpoint terbaik untuk
disertakan di GitHub dan digunakan aplikasi. File ini hanya memuat state model,
tanpa optimizer atau metadata training; bukan checkpoint untuk melanjutkan training.
File ini dikecualikan dari aturan ignore bobot di `.gitignore`.

Untuk memakai bobot aplikasi dari root proyek:

```powershell
python scripts/predict.py datasets/workshop_tools/test --checkpoint models/workshop_yolox_s.pth --device cuda --output predictions
```

Aplikasi Python juga memerlukan `exps/workshop_yolox_s.py`,
`datasets/workshop_tools/labels.json`, kode YOLOX, dan dependensinya.
