import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
import os
import random
from urllib.parse import quote_plus

# --- CONFIG ---
LOGIN_USERNAME = "negribaci"
LOGIN_PASSWORD = "holaManola1"
PHONE_NUMBER = "3878689688"

OUTPUT_CSV = "posts_trends.csv"
SCROLL_PAUSE = 2.0
MAX_SCROLLS = 500
# ---------------------------------------------------------------------------

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
                # Buscar específicamente el texto principal de la tendencia (blanco)
                name_elem = te.find_element(
                    By.XPATH,
                    './/div[@dir="ltr" and contains(@style, "rgb(231, 233, 234)")]//span'
                )
                trend_name = name_elem.text.strip()

                if trend_name and trend_name not in seen:
                    seen.add(trend_name)
                    trends.append(trend_name)
                    print(f"[trends] agregado: {trend_name}")

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

def parse_count_text(text):
    """Parsea counts tipo '1.2k', '3M', '7 163', etc -> int"""
    if not text:
        return 0
    text = text.strip().lower().replace("\u202f", "").replace("\xa0", "").replace(",", "")
    try:
        if text.endswith("k"):
            return int(float(text[:-1]) * 1000)
        if text.endswith("m"):
            return int(float(text[:-1]) * 1000000)
        return int(text.replace(".", ""))
    except:
        return 0

def safe_get_count(art, testid):
    """Devuelve el número de métricas (like, retweet, reply) o 0 si no existe"""
    try:
        elem = art.find_element(
            By.XPATH,
            f'.//button[@data-testid="{testid}"]//span[@data-testid="app-text-transition-container"]//span'
        )
        text = elem.text.strip()
        return parse_count_text(text)
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
        return parse_count_text(text)
    except:
        return 0

def scrape_trending(driver, trend_limit=10, per_trend_scrolls=8):
    """
    Para cada tendencia encontrada, busca en X los posts y devuelve lista de posts con campo 'trend'.
    """
    topics = get_trending_topics(driver, max_scrolls=trend_limit)
    all_posts = []

    for topic in topics:
        print(f"\n[search] Scrapeando tendencia: {topic}")
        q = quote_plus(topic)
        search_url = f"https://x.com/search?q={q}&src=trend_click"
        driver.get(search_url)
        time.sleep(4)  # esperar a que cargue resultados

        last_height = driver.execute_script("return document.body.scrollHeight")
        scrolls = 0
        seen = set()

        while scrolls < per_trend_scrolls:
            tweet_articles = driver.find_elements(By.XPATH, '//article[@data-testid="tweet"]')
            print(f"  Scroll {scrolls+1}: {len(tweet_articles)} artículos encontrados")
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

                likes = safe_get_count(art, "like")
                retweets = safe_get_count(art, "retweet")
                replies = safe_get_count(art, "reply")
                views = safe_get_views(art)

                tweet_id = (tweet_text + tweet_date)[:200]  # pequeño id
                if tweet_id in seen or not tweet_text:
                    continue
                seen.add(tweet_id)

                all_posts.append({
                    "trend": topic,
                    "text": tweet_text,
                    "likes": likes,
                    "retweets": retweets,
                    "replies": replies,
                    "views": views,
                    "created_at": tweet_date
                })

            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(random.uniform(2, 4))
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                print("  No hay más contenido en esta búsqueda.")
                break
            last_height = new_height
            scrolls += 1

        print(f"  -> posts recolectados para '{topic}': {len([p for p in all_posts if p['trend']==topic])}")

    print(f"Total posts extraídos de todas las tendencias: {len(all_posts)}")
    return all_posts

if __name__ == "__main__":
    driver = setup_driver()
    login(driver)
    try:
        posts = scrape_trending(driver, trend_limit=10, per_trend_scrolls=6)
    except Exception as e:
        print(f"Error durante scraping: {e}")
        posts = []
    driver.quit()

    if posts:
        df_new = pd.DataFrame(posts)
        cols = ["trend", "text", "likes", "retweets", "replies", "views", "created_at"]
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
