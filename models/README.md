Simpan checkpoint pretrained YOLOX-S (`yolox_s.pth`) di folder ini.
Jika belum tersedia, unduh bobot sesuai petunjuk di [README utama](../README.md).

Checkpoint hasil training disimpan terpisah di
`../YOLOX/YOLOX_outputs/workshop_yolox_s/best_ckpt.pth`.
Gunakan checkpoint hasil training tersebut untuk evaluasi dan prediksi kelas workshop.

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
