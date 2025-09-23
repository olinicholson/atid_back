# Este codigo trae a los seguidos, osea la que gente que siguen, mis seguidores
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager

# --- CONFIG ---
LOGIN_USERNAME = "negribaci"
LOGIN_PASSWORD = "holaManola1"
PHONE_NUMBER = "3878689688"
FOLLOWERS_FILE = "followers_infobae.csv"   # archivo ya generado
OUTPUT_CSV = "following_of_followers.csv"
SCROLL_PAUSE = 2.0
MAX_SCROLLS = 50  # ajustalo, puede ser pesado
# -------------

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

def scrape_following(driver, username):
    """Devuelve la lista de seguidos de un usuario dado"""
    driver.get(f"https://x.com/{username}/following")
    time.sleep(5)

    last_height = driver.execute_script("return document.body.scrollHeight")
    scrolls = 0
    seen = set()
    data = []

    while scrolls < MAX_SCROLLS:
        user_cells = driver.find_elements(By.XPATH, '//button[@data-testid="UserCell"]')
        for uc in user_cells:
            try:
                username_span = uc.find_element(By.XPATH, './/span[starts-with(text(),"@")]')
                following_handle = username_span.text
            except:
                continue
            if following_handle in seen:
                continue
            seen.add(following_handle)

            try:
                name_span = uc.find_element(By.XPATH, './/span[not(starts-with(text(),"@"))]')
                following_name = name_span.text
            except:
                following_name = None

            data.append({
                "follower_username": username,
                "following_username": following_handle,
                "following_name": following_name
            })

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
        scrolls += 1

    return data

if __name__ == "__main__":
    followers_df = pd.read_csv(FOLLOWERS_FILE)
    follower_usernames = followers_df["username"].tolist()

    driver = setup_driver()
    login(driver)

    all_data = []
    for i, follower in enumerate(follower_usernames, 1):
        print(f"[{i}/{len(follower_usernames)}] Scrapeando seguidos de {follower}...")
        try:
            following_data = scrape_following(driver, follower.strip("@"))
            all_data.extend(following_data)
        except Exception as e:
            print(f"Error con {follower}: {e}")
            continue

    driver.quit()

    df = pd.DataFrame(all_data)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Guardados {len(df)} registros en {OUTPUT_CSV}")
