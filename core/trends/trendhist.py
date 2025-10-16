import time
import pandas as pd
import re
from datetime import datetime
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
def scrape_historical_trends(driver, date_str, hour_str, max_delta_minutes=None):
    # Backwards-compatible: use helper to get blocks map and pick the block matching hour_str
    blocks_map = scrape_trends_for_date(driver, date_str)
    if not blocks_map:
        return []

    # Try exact match first
    target_block = None
    if hour_str in blocks_map:
        target_block = blocks_map[hour_str]

    if not target_block:
        # If exact hour not found, and tolerance provided, pick the closest hour key from blocks_map
        if max_delta_minutes is not None and blocks_map:
            # convert blocks_map keys (hour strings) to minutes
            candidates = []
            for raw_hour in blocks_map.keys():
                m = re.search(r"(\d{1,2}:\d{2})", raw_hour)
                if not m:
                    continue
                try:
                    t = datetime.strptime(m.group(1), "%H:%M")
                    minutes = t.hour * 60 + t.minute
                    candidates.append((minutes, raw_hour))
                except Exception:
                    continue

            m2 = re.search(r"(\d{1,2}:\d{2})", hour_str)
            if m2 and candidates:
                try:
                    target_t = datetime.strptime(m2.group(1), "%H:%M")
                    target_minutes = target_t.hour * 60 + target_t.minute
                    best = min(candidates, key=lambda x: abs(x[0] - target_minutes))
                    diff = abs(best[0] - target_minutes)
                    if diff <= max_delta_minutes:
                        chosen_raw = best[1]
                        target_block = blocks_map.get(chosen_raw)
                        print(f"No se encontró {hour_str}; usando el bloque más cercano {chosen_raw} (diferencia {diff} minutos)")
                    else:
                        print(f"No se encontró la hora {hour_str} en la página y ningún bloque está dentro de {max_delta_minutes} minutos.")
                        return []
                except Exception:
                    print(f"No se pudo parsear la hora objetivo {hour_str} para búsqueda por rango.")
                    return []
            else:
                print(f"No se encontró la hora {hour_str} en la página.")
                return []
        else:
            print(f"No se encontró la hora {hour_str} en la página.")
            return []

    # target_block is expected to be a list of dicts already (from helper)
    trend_list = []
    if target_block is None:
        return []
    # If helper returned the trend dicts directly, just return them
    trend_list = target_block
    print(f"Tendencias encontradas para {date_str} {hour_str}: {len(trend_list)}")
    return trend_list


def scrape_trends_for_date(driver, date_str):
    """
    Scrape the trends page for a whole date and return a mapping hour_str -> list of trend dicts.
    This loads the page once and extracts every hour block.
    """
    url = f"https://archive.twitter-trending.com/argentina/{date_str}"
    print(f"Abriendo página (date): {url}")
    driver.get(url)
    time.sleep(3)

    bloques = driver.find_elements(By.CLASS_NAME, "tek_tablo")
    result = {}
    for b in bloques:
        try:
            hora_bloque = b.find_element(By.CLASS_NAME, "trend_baslik611").text.strip()
        except Exception:
            continue

        # Extract rows inside this block
        rows = b.find_elements(By.CSS_SELECTOR, "table.tablo611 tr.tr_table")
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

        # store list under normalized hour string
        result[hora_bloque] = trend_list

    print(f"Bloques horarios extraídos para {date_str}: {len(result)}")
    return result

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
