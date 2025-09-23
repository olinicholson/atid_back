import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
import random
import os

# --- CONFIG ---
LOGIN_USERNAME = "negribaci"
LOGIN_PASSWORD = "holaManola1"
PHONE_NUMBER = "3878689688"

OUTPUT_TRENDS = "trends.csv"

def setup_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
    return driver

def login(driver):
    driver.get("https://x.com/login")
    time.sleep(3)
    driver.find_element(By.NAME, "text").send_keys(LOGIN_USERNAME, Keys.RETURN)
    time.sleep(3)
    try:
        phone_input = driver.find_element(By.NAME, "text")
        if phone_input:
            phone_input.send_keys(PHONE_NUMBER, Keys.RETURN)
            time.sleep(3)
    except:
        pass
    driver.find_element(By.NAME, "password").send_keys(LOGIN_PASSWORD, Keys.RETURN)
    time.sleep(5)

def get_trending_topics(driver, max_scrolls=10, limit=None):
    """Extrae los nombres de tendencias desde la sección de tendencias"""
    driver.get("https://x.com/explore/tabs/trending")
    time.sleep(5)

    last_height = driver.execute_script("return document.body.scrollHeight")
    scrolls = 0
    seen = set()
    trends = []

    while scrolls < max_scrolls:
        trend_elems = driver.find_elements(By.XPATH, '//div[@data-testid="trend"]')
        print(f"[trends] encontrados {len(trend_elems)} elementos (scroll {scrolls})")

        for te in trend_elems:
            try:
                # Buscar específicamente el texto principal de la tendencia (color blanco)
                name_elem = te.find_element(
                    By.XPATH,
                    './/div[@dir="ltr" and contains(@style, "rgb(231, 233, 234)")]//span'
                )
                trend_name = name_elem.text.strip()

                if trend_name and trend_name not in seen:
                    seen.add(trend_name)
                    trends.append(trend_name)
                    print(f"[trends] #{len(trends)} agregado: {trend_name}")

                    if limit and len(trends) >= limit:
                        print(f"[trends] alcanzado límite {limit}")
                        return trends
            except:
                continue

        # Scroll para cargar más tendencias
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(random.uniform(2, 4))
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            print("[trends] no hay más contenido para cargar.")
            break
        last_height = new_height
        scrolls += 1

    print(f"[trends] retornando {len(trends)} tendencias")
    return trends

if __name__ == "__main__":
    driver = setup_driver()
    login(driver)

    try:
        trends = get_trending_topics(driver, max_scrolls=10, limit=50)  # podés ajustar el límite
    except Exception as e:
        print(f"Error durante scraping de tendencias: {e}")
        trends = []
    driver.quit()

    if trends:
        df = pd.DataFrame({"id": list(range(1, len(trends)+1)), "trend": trends})
        df.to_csv(OUTPUT_TRENDS, index=False, encoding="utf-8")
        print(f"Guardadas {len(trends)} tendencias en {OUTPUT_TRENDS}")
    else:
        print("No se guardaron tendencias.")
