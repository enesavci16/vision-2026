# Q1 Konsolidasyon — Araç Hız Ölçüm Pipeline'ı

Kameradan km/h'ye: YOLO + BEV Dönüşümü + SpeedMeasurer ile uçtan uca trafik hız ölçümü.

## Pipeline Mimarisi

```
Video Karesi
    │
    ▼
YOLO (YOLOv8n)          → Araç tespiti (bbox + class)
    │
    ▼
YOLO Built-in Tracker   → Track ID ataması (persist=True)
    │
    ▼
PerspectiveTransformer  → Bbox merkezi → BEV koordinatı (H matrisi)
    │
    ▼
SpeedMeasurer           → Öklid mesafesi + dt → km/h
    │
    ▼
Overlay + VideoWriter   → annotated.mp4
    │
    ▼
JSON Rapor              → speed_report.json
```

## Veri Kaynağı

- **Video:** Açık kaynak trafik CCTV görüntüsü (Sysvideo 4K IP Camera Demo)
- **Model:** YOLOv8n (Ultralytics — pretrained COCO)
- **Kalibrasyon:** Manuel — `SRC_POINTS` pick_points.py ile seçildi

## Kurulum

```bash
# Venv aktive et
source .venv/Scripts/activate  # Windows Git Bash

# Bağımlılıklar
pip install ultralytics opencv-python numpy
```

## Çalıştırma

```bash
# Repo kökünden
PYTHONPATH=src python consolidation/q1_speed_pipeline/pipeline.py
```

Çıktılar:
- `consolidation/q1_speed_pipeline/outputs/annotated.mp4`
- `consolidation/q1_speed_pipeline/outputs/speed_report.json`

## Koordinat Kalibrasyonu

```bash
PYTHONPATH=src python consolidation/q1_speed_pipeline/pick_points.py \
    --video consolidation/q1_speed_pipeline/video/traffic.mp4
```

Video karesinde 4 noktaya tıkla (sol üst → sağ üst → sağ alt → sol alt).
Çıkan koordinatları `config.py`'daki `SRC_POINTS`'e yapıştır.

## Konfigürasyon

`consolidation/q1_speed_pipeline/config.py`:

| Parametre | Açıklama |
|---|---|
| `VIDEO_PATH` | Girdi video yolu |
| `MODEL_PATH` | YOLO model dosyası |
| `CONFIDENCE_THR` | Tespit eşiği (0.0–1.0) |
| `PIXEL_PER_METER` | BEV canvas ölçeği |
| `SRC_POINTS` | Video karesindeki 4 kalibrasyon noktası |
| `DST_POINTS` | BEV canvas'taki karşılık noktaları |

## Örnek Çıktı

```json
{
    "1": {
        "average_speed": 47.2,
        "max_speed": 156.7,
        "current_speed": 54.6,
        "measurement_count": 180
    }
}
```

## Limitasyonlar

- **ByteTrack entegre değil** — YOLO built-in tracker kullanıldı. Araç görüş
  alanından çıkıp girince yeni ID atanıyor. P2 entegrasyonu P4'te planlanıyor.
- **pixel_per_meter manuel kalibre edildi** — Gerçek mesafe ölçümü yapılmadı,
  şerit genişliği tahminine dayalı. Üretim sisteminde ground truth gerekir.
- **Hız tüm geçmişin ortalaması** — Sliding window uygulanmadı. Duran araç
  geçmiş hızını ortalamaya taşıyor. Sliding window P9 kapsamında.
- **get_report() her karede çağrılıyor** — O(n) maliyet. Araç sayısı arttıkça
  yavaşlar. Cache veya incremental güncelleme P9 kapsamında.
- **max_speed değerleri yüksek** — Track ID tutarsızlığından kaynaklanan BEV
  sıçramaları anlık yüksek hız üretiyor. avg_speed daha güvenilir.

## Modül Bağımlılıkları

| Modül | Proje | Konum |
|---|---|---|
| `TrafficDetector` | P1 | `src/p1_detector/models/yolo.py` |
| `PerspectiveTransformer` | P3 | `src/perspective_transformer.py` |
| `SpeedMeasurer` | P3 | `src/speed_measurer.py` |