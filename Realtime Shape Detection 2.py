import math
import numpy as np
import cv2

# Dictionary dari semua kontur
contours = {}
# Array dari sisi poligon
approx = []
# Skala teks
scale = 2
# Kamera
cap = cv2.VideoCapture(0)
print("Tekan 'q' untuk keluar")

# Definisikan codec dan buat objek VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

# Fungsi untuk menghitung sudut
def angle(pt1, pt2, pt0):
    dx1 = pt1[0][0] - pt0[0][0]
    dy1 = pt1[0][1] - pt0[0][1]
    dx2 = pt2[0][0] - pt0[0][0]
    dy2 = pt2[0][1] - pt0[0][1]
    return float((dx1 * dx2 + dy1 * dy2)) / math.sqrt(float((dx1 * dx1 + dy1 * dy1)) * (dx2 * dx2 + dy2 * dy2) + 1e-10)

while cap.isOpened():
    # Menangkap frame per frame
    ret, frame = cap.read()
    if ret:
        # Ubah ke grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Canny
        canny = cv2.Canny(frame, 80, 240, 3)

        # Kontur
        contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
            # Mendekati kontur dengan akurasi proporsional terhadap perimeter kontur
            approx = cv2.approxPolyDP(contours[i], cv2.arcLength(contours[i], True) * 0.02, True)

            # Lewati objek kecil atau non-konveks
            if abs(cv2.contourArea(contours[i])) < 100 or not cv2.isContourConvex(approx):
                continue

            # Segitiga
            if len(approx) == 3:
                x, y, w, h = cv2.boundingRect(contours[i])
                cv2.putText(frame, 'TRI', (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
            elif 4 <= len(approx) <= 6:
                # Jumlah vertex kurva poligonal
                vtc = len(approx)
                # Dapatkan cos dari semua sudut
                cos = []
                for j in range(2, vtc + 1):
                    cos.append(angle(approx[j % vtc], approx[j - 2], approx[j - 1]))
                # Urutkan cos secara ascending
                cos.sort()
                # Dapatkan cos terkecil dan terbesar
                mincos = cos[0]
                maxcos = cos[-1]

                # Gunakan derajat yang diperoleh di atas dan jumlah vertex untuk menentukan bentuk kontur
                x, y, w, h = cv2.boundingRect(contours[i])
                if vtc == 4:
                    cv2.putText(frame, 'Rect', (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
                elif vtc == 5:
                    cv2.putText(frame, 'Penta', (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
                elif vtc == 6:
                    cv2.putText(frame, 'Hexa', (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                # Deteksi dan label lingkaran
                area = cv2.contourArea(contours[i])
                x, y, w, h = cv2.boundingRect(contours[i])
                radius = w / 2
                if abs(1 - (float(w) / h)) <= 2 and abs(1 - (area / (math.pi * radius * radius))) <= 0.2:
                    cv2.putText(frame, 'Bulat', (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)

        # Tampilkan frame hasil
        out.write(frame)
        cv2.imshow('frame', frame)
        cv2.imshow('canny', canny)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Jika 'q' ditekan
            break

# Ketika semuanya selesai, release capture
cap.release()
out.release()
cv2.destroyAllWindows()