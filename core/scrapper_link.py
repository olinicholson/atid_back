import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
import os
import random

# --- CONFIG ---
LOGIN_USERNAME = "guillerminabacigalupo@hotmail.com"
LOGIN_PASSWORD = "holaManola1"

OUTPUT_CSV = "linkedin_posts.csv"
SCROLL_PAUSE = 2.0
MAX_SCROLLS = 100
# -------------

def setup_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
    return driver

def login(driver):
    driver.get("https://www.linkedin.com/login")
    time.sleep(3)
    driver.find_element(By.ID, "username").send_keys(LOGIN_USERNAME)
    driver.find_element(By.ID, "password").send_keys(LOGIN_PASSWORD, Keys.RETURN)
    time.sleep(5)

    # --- ESPERA MANUAL PARA 2FA ---
    print("⚠️ Si aparece verificación en dos pasos, ingrésala manualmente.")
    print("⏳ Esperando 30 segundos antes de continuar...")
    time.sleep(30)   # te da tiempo a poner el código

def scrape_posts(driver, profile_url):
    driver.get(profile_url)
    time.sleep(10)

    # --- CAPTURAR NOMBRE DEL PERFIL ---
    try:
        name_elem = driver.find_element(By.TAG_NAME, "h1")
        profile_id = name_elem.text.strip()
    except:
        print("⚠️ No se pudo obtener el nombre del perfil, se usará la URL.")
        profile_id = profile_url

    last_height = driver.execute_script("return document.body.scrollHeight")
    scrolls = 0
    posts = []
    seen = set()

    while scrolls < MAX_SCROLLS:
        post_divs = driver.find_elements(By.XPATH, '//div[contains(@class, "feed-shared-update-v2")]')
        print(f"Scroll {scrolls+1}: {len(post_divs)} posts encontrados")

        for div in post_divs:
            try:
                text_elem = div.find_element(By.XPATH, './/div[contains(@class,"update-components-text")]')
                post_text = text_elem.text.strip()
            except:
                post_text = ""

            try:
                date_elem = div.find_element(By.XPATH, './/span[contains(@class,"visually-hidden")]')
                post_date = date_elem.text.strip()
            except:
                post_date = ""

            try:
                reactions_elem = div.find_element(By.XPATH, './/span[contains(@class,"social-details-social-counts__reactions-count")]')
                reactions = reactions_elem.text.strip()
            except:
                reactions = "0"

            try:
                comments_elem = div.find_element(By.XPATH, './/span[contains(text(),"comentario")]')
                comments = comments_elem.text.strip().split()[0]
            except:
                comments = "0"

            try:
                shares_elem = div.find_element(By.XPATH, './/span[contains(text(),"compartido")]')
                shares = shares_elem.text.strip().split()[0]
            except:
                shares = "0"

            post_id = post_text + post_date
            if post_id in seen or not post_text:
                continue
            seen.add(post_id)

            posts.append({
                "profile": profile_id,   # 👈 ahora guarda el nombre del perfil
                "text": post_text,
                "reactions": reactions,
                "comments": comments,
                "shares": shares,
                "created_at": post_date,
                "profile_url": profile_url  # 👈 opcional: guardo la URL también
            })

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(random.uniform(2, 5))
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            print("No hay más contenido para cargar.")
            break
        last_height = new_height
        scrolls += 1

    print(f"Total posts extraídos: {len(posts)}")
    return posts

if __name__ == "__main__":
    profile = input("Ingrese la URL del perfil de LinkedIn: ").strip()

    driver = setup_driver()
    login(driver)
    try:
        posts = scrape_posts(driver, profile)
    except Exception as e:
        print(f"Error con {profile}: {e}")
        posts = []
    driver.quit()

    if posts:
        df_new = pd.DataFrame(posts)
        cols = ["profile", "text", "reactions", "comments", "shares", "created_at"]
        df_new = df_new[cols]

        if os.path.exists(OUTPUT_CSV):
            df_old = pd.read_csv(OUTPUT_CSV)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_all = df_new

        df_all.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
        print(f"Guardados {len(df_new)} posts nuevos en {OUTPUT_CSV} (total: {len(df_all)})")
    else:
        print("No se guardaron posts.")
