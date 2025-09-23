import cv2
import pytesseract
import csv
import re

# --- Cargar imagen ---
img = cv2.imread("tabla.png")

# --- Convertir a gris y binarizar ---
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# --- OCR ---
custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(thresh, config=custom_config)

# --- Procesar texto ---
rows = text.split("\n")  # dividir por líneas
clean_rows = []

for row in rows:
    row = row.strip()
    if row:  # evitar vacíos
        # Reemplazar múltiples espacios por 1
        row = re.sub(r"\s{2,}", ";", row)  
        clean_rows.append(row)

# --- Guardar en CSV ---
with open("tabla.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    for row in clean_rows:
        writer.writerow(row.split(";"))
