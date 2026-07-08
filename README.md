# Vision 2026: Automatic Number Plate Recognition (ANPR) Pipeline

Bu proje, YOLOv8 mimarisi kullanılarak geliştirilmiş, bölgesel (Avrupa, ABD,
Brezilya) plaka tespiti ve Akıllı Ulaşım Sistemleri (ITS) analitiği boru hattıdır.
Model, OpenALPR benchmark veri seti üzerinde fine-tune edilmek üzere tasarlanmıştır.

## Kurulum (Setup)

Projeyi lokal ortamınıza klonlayın ve sanal ortamı başlatın:

```bash
git clone https://github.com/enesavci16/vision-2026.git
cd vision-2026
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
# source .venv/bin/activate
pip install -r requirements.txt   # ya da: uv sync
```

## Veri Hazırlığı (Data Preparation)

Veri seti repoya dahil **değildir**; boyutu ve üretilebilir olması nedeniyle
ayrıca indirilmelidir.

1. **İndir:** OpenALPR benchmark veri setini
   [openalpr/benchmarks](https://github.com/openalpr/benchmarks) reposundan alın.

2. **Bilinen veri anomalisi — `PAG5219.jpg` silinmeli:**
   `endtoend/br/` klasöründe `PAG5219` stem'i altında iki farklı görüntü bulunur
   (`.jpg` ve `.png`). Orijinal OpenALPR etiketi `.png` dosyasını işaret eder;
   `.jpg` etiketsiz/hatalı bir kopyadır. Silinmezse split script'i 445 örnek
   üretir ve `br/train` bölümünde görüntü-etiket paritesi bozulur.

```bash
   rm datasets/openalpr_benchmarks/endtoend/br/PAG5219.jpg
```

3. **Split script'ini çalıştır** (stratified, bölge-bazlı 70/10/20):

```bash
   python src/anpr/prepare_dataset.py \
       --source ./datasets/openalpr_benchmarks/endtoend \
       --dest ./datasets/openalpr \
       --seed 42
```

4. **Beklenen çıktı** (doğrulama):
   - Toplam: **444** görüntü (eu 108 + us 222 + br 114)
   - Dağılım: train **309** / val **43** / test **92**

> **Not (Ultralytics yol çözümlemesi):** `data.yaml` içindeki `path: openalpr`
> alanı, dosyanın konumuna göre değil, Ultralytics'in global `settings.yaml`
> içindeki `DATASETS_DIR` (varsayılan `~/datasets/`) değerine göre çözümlenir.
> Bu nedenle bölünmüş veri seti `~/datasets/openalpr/` altında bulunmalıdır.