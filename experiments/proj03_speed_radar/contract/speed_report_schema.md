# Speed Report — JSON Schema

## Açıklama
`SpeedMeasurer.get_report()` metodunun döndürdüğü JSON yapısı.

## Üst seviye yapı
Her anahtar bir `track_id`'dir (örn: `"car_1"`).

## Alan Açıklamaları

| Alan | Tip | Açıklama |
|------|-----|----------|
| average_speed | float | Average speed of the tracked object across all measurements (km/h) |
| max_speed | float | Maximum speed recorded for the tracked object (km/h) |
| current_speed | float | Most recent speed measurement (km/h)  |
| measurement_count | int |Total number of position updates received for this track |

## Kenar Durumlar
- Tek update alan araç: İki konum noktası olmadığı için hız 
  hesaplanamaz. average_speed, max_speed ve current_speed 
  0.0 olarak raporlanır.
- Aynı timestamp ile iki update: ValueError fırlatılır, 
  raporda bu araç son geçerli haliyle görünür.
- Bilinmeyen track_id ile get_speed() çağrısı: KeyError fırlatılır.