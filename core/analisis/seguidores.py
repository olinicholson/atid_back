# Este codigo me trae los usuarios y nombres de las personas que siguen a cierta cuenta
import time
import os
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
USERNAME_TO_SCRAPE = "infobae"
OUTPUT_CSV = f"followers_{USERNAME_TO_SCRAPE}.csv"
SCROLL_PAUSE = 2.0
MAX_SCROLLS = 500
# -------------

def setup_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
    return driver

def login(driver):
    driver.get("https://x.com/login")
    time.sleep(3)

    # Usuario
    driver.find_element(By.NAME, "text").send_keys(LOGIN_USERNAME, Keys.RETURN)
    time.sleep(3)

    # Si pide teléfono o email
    try:
        phone_input = driver.find_element(By.NAME, "text")
        if phone_input:
            phone_input.send_keys(PHONE_NUMBER, Keys.RETURN)
            time.sleep(3)
    except:
        pass

    # Contraseña
    driver.find_element(By.NAME, "password").send_keys(LOGIN_PASSWORD, Keys.RETURN)
    time.sleep(5)

def scrape_followers(username):
    driver = setup_driver()
    login(driver)
    driver.get(f"https://x.com/{username}/followers")
    time.sleep(5)

    last_height = driver.execute_script("return document.body.scrollHeight")
    scrolls = 0
    seen = set()
    data = []

    while scrolls < MAX_SCROLLS:
        user_cells = driver.find_elements(By.XPATH, '//button[@data-testid="UserCell"]')
        print(f"Scroll {scrolls} - encontrados: {len(user_cells)}")

        for uc in user_cells:
            try:
                username_span = uc.find_element(By.XPATH, './/span[starts-with(text(),"@")]')
                user_handle = username_span.text
            except:
                continue
            if user_handle in seen:
                continue
            seen.add(user_handle)

            try:
                name_span = uc.find_element(By.XPATH, './/span[not(starts-with(text(),"@"))]')
                full_name = name_span.text
            except:
                full_name = None

            data.append({
                "username": user_handle,
                "name": full_name
            })

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
        scrolls += 1

    driver.quit()
    return data

if __name__ == "__main__":
    followers = scrape_followers(USERNAME_TO_SCRAPE)
    df_new = pd.DataFrame(followers)

    # Si ya existe un archivo previo, combinar
    if os.path.exists(OUTPUT_CSV):
        df_old = pd.read_csv(OUTPUT_CSV)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
        df_combined.drop_duplicates(subset=["username"], inplace=True)
    else:
        df_combined = df_new

    df_combined.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Guardados {len(df_combined)} seguidores únicos en {OUTPUT_CSV}")
