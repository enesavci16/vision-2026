"""
Homografi kalibrasyon aracı.

Kullanım:
    python pick_points.py --video VIDEO_PATH

Tıklama sırası (4 nokta):
    1. Sol üst
    2. Sağ üst
    3. Sağ alt
    4. Sol alt

Çıktı: config.py'a yapıştırılacak SRC_POINTS değerleri.
"""

import argparse
import cv2
import numpy as np

points: list[tuple[int, int]] = []
frame_display = None


def mouse_callback(event, x, y, flags, param):
    global points, frame_display
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            print(f"  Nokta {len(points)}: ({x}, {y})")
            cv2.circle(frame_display, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(
                frame_display,
                str(len(points)),
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            if len(points) > 1:
                for i in range(len(points) - 1):
                    cv2.line(frame_display, points[i], points[i + 1], (0, 255, 255), 2)
            if len(points) == 4:
                cv2.line(frame_display, points[3], points[0], (0, 255, 255), 2)
                print("\n✅ 4 nokta seçildi!")
                print("\nConfig'e yapıştır:")
                print(f"SRC_POINTS = np.float32({[list(p) for p in points]})")
            cv2.imshow("Nokta Seç", frame_display)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Video dosyası yolu")
    parser.add_argument("--frame", type=int, default=30, help="Hangi kare (default: 30)")
    args = parser.parse_args()

    global frame_display

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Video okunamadı.")
        return

    frame_display = frame.copy()

    print("=" * 50)
    print("NOKTA SEÇİCİ — 4 noktaya tıkla")
    print("Sıra: Sol üst → Sağ üst → Sağ alt → Sol alt")
    print("Çıkmak için: Q")
    print("=" * 50)

    cv2.imshow("Nokta Seç", frame_display)
    cv2.setMouseCallback("Nokta Seç", mouse_callback)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()

    if len(points) == 4:
        print("\n" + "=" * 50)
        print("SRC_POINTS hazır:")
        print(f"SRC_POINTS = np.float32({[list(p) for p in points]})")
        print("=" * 50)
    else:
        print(f"\n⚠️ Sadece {len(points)} nokta seçildi, 4 gerekli.")


if __name__ == "__main__":
    main()
