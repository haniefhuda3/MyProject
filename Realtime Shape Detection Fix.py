import cv2
import numpy as np


def deteksi_bentuk(frame):
    # Konversi gambar ke grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Mengaburkan gambar untuk mengurangi noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Deteksi tepi menggunakan Canny edge detector
    edged = cv2.Canny(blurred, 50, 150)

    # Menemukan kontur dalam gambar yang telah di-edge
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Memfilter kontur kecil berdasarkan area
        if cv2.contourArea(contour) > 500:
            # Meng-approximate kontur untuk menemukan bentuk
            approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)

            # Mengidentifikasi bentuk berdasarkan jumlah sisi
            if len(approx) == 3:
                nama_bentuk = "Segitiga"
            elif len(approx) == 4:
                (x, y, w, h) = cv2.boundingRect(approx)
                rasio_aspek = w / float(h)
                if 0.95 <= rasio_aspek <= 1.05:
                    nama_bentuk = "Persegi"
                else:
                    nama_bentuk = "Persegi Panjang"
            elif len(approx) == 5:
                nama_bentuk = "Pentagon"
            else:
                nama_bentuk = "Lingkaran"

            # Menggambar kontur dan nama bentuk pada frame
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(frame, nama_bentuk, (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


def main():
    # Membuka webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Tidak dapat membuka webcam.")
        return

    while True:
        # Membaca frame dari webcam
        ret, frame = cap.read()
        if not ret:
            print("Gagal mengambil frame.")
            break

        # Memproses frame untuk mendeteksi bentuk
        deteksi_bentuk(frame)

        # Menampilkan frame dengan deteksi bentuk
        cv2.imshow("Deteksi Bentuk", frame)

        # Keluar jika tombol 'q' ditekan
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Melepaskan webcam dan menutup semua jendela
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()