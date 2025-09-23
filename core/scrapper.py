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
LOGIN_USERNAME = "negribaci"
LOGIN_PASSWORD = "holaManola1"
PHONE_NUMBER = "3878689688"

OUTPUT_CSV = "posts_galicia.csv"   # archivo
SCROLL_PAUSE = 2.0
MAX_SCROLLS = 500  # ajusta según lo que quieras scrapear
# -------------

# cuentas : brubankarg, uala_arg, BalanzCapital, cocoscap, Supervielle_ARG, BancoGalicia, VisaArgentina, BcoHipotecario, 
# Banco_Patagonia, mercadopago, Banza_AR, lemonapp_ar

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
    # Si pide teléfono/email
    try:
        phone_input = driver.find_element(By.NAME, "text")
        if phone_input:
            phone_input.send_keys(PHONE_NUMBER, Keys.RETURN)
            time.sleep(3)
    except:
        pass
    driver.find_element(By.NAME, "password").send_keys(LOGIN_PASSWORD, Keys.RETURN)
    time.sleep(5)

def scrape_posts(driver, username):
    driver.get(f"https://x.com/{username}")
    time.sleep(10)  # Espera mayor para carga inicial
    last_height = driver.execute_script("return document.body.scrollHeight")
    scrolls = 0
    posts = []
    seen = set()
    while scrolls < MAX_SCROLLS:
        tweet_articles = driver.find_elements(By.XPATH, '//article[@data-testid="tweet"]')
        print(f"Scroll {scrolls+1}: {len(tweet_articles)} artículos encontrados")
        for art in tweet_articles:
            try:
                text_elem = art.find_element(By.XPATH, './/div[@data-testid="tweetText"]')
                tweet_text = text_elem.text
            except:
                tweet_text = ""
            try:
                date_elem = art.find_element(By.XPATH, './/time')
                tweet_date = date_elem.get_attribute("datetime")
            except:
                tweet_date = ""
            # Extraer métricas (likes, retweets, replies, views)
            likes = safe_get_count(art, "like")
            retweets = safe_get_count(art, "retweet")
            replies = safe_get_count(art, "reply")
            views = safe_get_views(art)
            tweet_id = tweet_text + tweet_date
            if tweet_id in seen or not tweet_text:
                continue
            seen.add(tweet_id)
            posts.append({
                "username": username,
                "text": tweet_text,
                "likes": likes,
                "retweets": retweets,
                "replies": replies,
                "views": views,
                "created_at": tweet_date
            })
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        # Espera aleatoria entre 2 y 5 segundos
        time.sleep(random.uniform(2, 5))
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            print("No hay más contenido para cargar.")
            break
        last_height = new_height
        scrolls += 1
    print(f"Total posts extraídos para {username}: {len(posts)}")
    return posts

def safe_get_count(art, testid):
    """Devuelve el número de métricas (like, retweet, reply) o 0 si no existe"""
    try:
        elem = art.find_element(
            By.XPATH,
            f'.//button[@data-testid="{testid}"]//span[@data-testid="app-text-transition-container"]//span'
        )
        text = elem.text.strip()
        if not text:
            return 0
        return int(text.replace(",", "").replace(".", ""))
    except:
        return 0

def safe_get_views(art):
    """Devuelve el número de vistas de un tweet"""
    try:
        elem = art.find_element(
            By.XPATH,
            './/a[contains(@href, "/analytics")]//span[@data-testid="app-text-transition-container"]//span'
        )
        text = elem.text.strip()
        if not text:
            return 0
        # soporta 'k' y 'M'
        text = text.replace("\u202f", "").replace("\xa0", "").replace(",", "").replace(".", "")
        if "k" in text.lower():
            return int(float(text.lower().replace("k", "")) * 1000)
        if "m" in text.lower():
            return int(float(text.lower().replace("m", "")) * 1000000)
        return int(text)
    except:
        return 0

if __name__ == "__main__":
    # --- pedimos usuario a mano ---
    user = input("Ingrese el usuario de X (sin @): ").strip()

    driver = setup_driver()
    login(driver)
    try:
        posts = scrape_posts(driver, user)
    except Exception as e:
        print(f"Error con {user}: {e}")
        posts = []
    driver.quit()

    if posts:
        df_new = pd.DataFrame(posts)
        cols = ["username", "text", "likes", "retweets", "replies", "views", "created_at"]
        df_new = df_new[cols]

        # Si ya existe el archivo, append sin sobrescribir
        if os.path.exists(OUTPUT_CSV):
            df_old = pd.read_csv(OUTPUT_CSV)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_all = df_new

        df_all.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
        print(f"Guardados {len(df_new)} posts nuevos en {OUTPUT_CSV} (total: {len(df_all)})")
    else:
        print("No se guardaron posts.")
