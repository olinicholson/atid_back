import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
import sys

# --- CONFIG ---
OUTPUT_CSV = "historical_trends.csv"

# --- SETUP DRIVER ---
def setup_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
    return driver

# --- SCRAPER ---
def scrape_historical_trends(driver, date_str, hour_str):
    url = f"https://archive.twitter-trending.com/argentina/{date_str}"
    print(f"Abriendo página: {url}")
    driver.get(url)
    time.sleep(3)

    # Buscar todos los bloques de hora
    bloques = driver.find_elements(By.CLASS_NAME, "tek_tablo")
    target_block = None
    for b in bloques:
        try:
            hora_bloque = b.find_element(By.CLASS_NAME, "trend_baslik611").text.strip()
            if hora_bloque == hour_str:
                target_block = b
                break
        except:
            continue

    if not target_block:
        print(f"No se encontró la hora {hour_str} en la página.")
        return []

    # Extraer tendencias dentro del bloque
    rows = target_block.find_elements(By.CSS_SELECTOR, "table.tablo611 tr.tr_table")
    trend_list = []
    for r in rows:
        try:
            posicion = r.find_element(By.CLASS_NAME, "sira611").text.strip()
        except:
            posicion = ""
        try:
            nombre = r.find_element(By.CLASS_NAME, "trend611").text.strip()
        except:
            nombre = ""
        # El volumen a veces está en la fila siguiente
        try:
            volumen_elem = r.find_element(By.XPATH, 'following-sibling::tr[1]//span[@class="volume61"]')
            volumen = volumen_elem.text.strip()
        except:
            volumen = ""
        trend_list.append({
            "posicion": posicion,
            "nombre": nombre,
            "volumen": volumen
        })

    print(f"Tendencias encontradas para {date_str} {hour_str}: {len(trend_list)}")
    return trend_list

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python trend-hist.py <fecha> (formato DD-MM-YYYY)")
        sys.exit(1)

    date_input = sys.argv[1].strip()
    hour_input = input("Insertar hora (HH:MM): ").strip()

    driver = setup_driver()
    try:
        trends = scrape_historical_trends(driver, date_input, hour_input)
    except Exception as e:
        print(f"Error durante scraping: {e}")
        trends = []
    driver.quit()

    if trends:
        df = pd.DataFrame(trends)
        df["date"] = date_input
        df["hour"] = hour_input

        # Numerar tendencias
        df["numero"] = range(1, len(df)+1)
        cols_order = ["numero", "posicion", "nombre", "volumen", "date", "hour"]
        df = df[cols_order]

        if pd.io.common.file_exists(OUTPUT_CSV):
            df_old = pd.read_csv(OUTPUT_CSV)
            df_all = pd.concat([df_old, df], ignore_index=True)
        else:
            df_all = df

        df_all.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
        print(f"Guardadas {len(trends)} tendencias en {OUTPUT_CSV}")
    else:
        print("No se guardaron tendencias.")
