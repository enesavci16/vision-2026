import pytest
from speed_measurer import SpeedMeasurer

@pytest.fixture
def measurer():
    """Her test için temiz bir SpeedMeasurer instance'ı döndürür."""
    return SpeedMeasurer(pixel_per_meter=10.0)

# ---------------------------------------------------------
# Senaryo 1: İlk update() çağrısı (yeni track_id)
# ---------------------------------------------------------
def test_update_first_call_adds_to_history_without_speed(measurer):
    # 1. Arrange
    track_id = "car_1"
    initial_coord = (0.0, 0.0)
    initial_time = 1.0
    
    # 2. Act
    measurer.update(track_id, initial_coord, initial_time)
    
    # 3. Assert - Public API üzerinden doğrulama (Private _history erişimi kaldırıldı)
    report = measurer.get_report()
    assert track_id in report
    assert report[track_id]["measurement_count"] == 1
    # Tek ölçüm olduğu için hız hesaplanamaz, kontrat gereği 0.0 dönmeli
    assert measurer.get_speed(track_id) == 0.0

# ---------------------------------------------------------
# Senaryo 2: İkinci update() çağrısı ve hız hesabı
# ---------------------------------------------------------
def test_update_second_call_calculates_and_saves_speed(measurer):
    # 1. Arrange
    track_id = "car_1"
    first_coord = (0.0, 0.0)
    first_time = 0.0
    second_coord = (10.0, 0.0)
    second_time = 1.0 
    
    # 2. Act
    measurer.update(track_id, first_coord, first_time)
    measurer.update(track_id, second_coord, second_time)
    
    # 3. Assert - Public API üzerinden doğrulama
    report = measurer.get_report()
    assert report[track_id]["measurement_count"] == 2
    # Hız hesaplanmış olmalı (1 saniyede 1m hareket = 3.6 km/h)
    assert measurer.get_speed(track_id) == pytest.approx(3.6)

# ---------------------------------------------------------
# Senaryo 3: compute_speed (@staticmethod) — İzole birim testi
# ---------------------------------------------------------
def test_compute_speed_deterministic_input_returns_accurate_value():
    # 1. Arrange
    px_per_m = 1.0 
    coord_1 = (0.0, 0.0)
    time_1 = 0.0
    coord_2 = (10.0, 0.0) 
    time_2 = 1.0 
    
    # 2. Act
    calculated_speed = SpeedMeasurer.compute_speed(px_per_m, coord_1, time_1, coord_2, time_2)
    
    # 3. Assert
    assert calculated_speed == pytest.approx(36.0, abs=0.001)

# ---------------------------------------------------------
# Senaryo 4: get_speed(unknown_track_id) KeyError fırlatır
# ---------------------------------------------------------
def test_get_speed_unknown_track_id_raises_keyerror(measurer):
    # 1. Arrange
    unknown_id = "ghost_car"
    
    # 2 & 3. Act & Assert
    with pytest.raises(KeyError):
        measurer.get_speed(unknown_id)

# ---------------------------------------------------------
# Senaryo 5: get_report() — Çoklu track, çoklu update
# ---------------------------------------------------------
def test_get_report_multiple_tracks_calculates_correct_stats(measurer):
    # 1. Arrange
    measurer.update("car_1", (0.0, 0.0), 0.0)
    measurer.update("car_1", (10.0, 0.0), 1.0) 
    
    measurer.update("car_2", (5.0, 5.0), 0.0)
    
    # 2. Act
    report = measurer.get_report()
    
    # 3. Assert
    assert "car_1" in report
    assert "car_2" in report
    
    assert report["car_1"]["measurement_count"] == 2
    assert report["car_1"]["average_speed"] == pytest.approx(3.6)
    
    # Kontrat doğrulaması: Tek update alan araç için tüm hızlar 0.0 olmalı
    assert report["car_2"]["measurement_count"] == 1
    assert report["car_2"]["current_speed"] == 0.0
    assert report["car_2"]["average_speed"] == 0.0
    assert report["car_2"]["max_speed"] == 0.0

# ---------------------------------------------------------
# Senaryo 6: Edge case - dt=0 Koruması (Sıfıra bölünme)
# ---------------------------------------------------------
def test_update_same_timestamp_raises_valueerror(measurer):
    # 1. Arrange
    track_id = "car_1"
    coord_1 = (0.0, 0.0)
    time_1 = 1.5
    coord_2 = (10.0, 0.0)
    time_2 = 1.5 
    
    measurer.update(track_id, coord_1, time_1)
    
    # 2 & 3. Act & Assert
    with pytest.raises(ValueError):
        measurer.update(track_id, coord_2, time_2)  