import os, re, time, random, urllib.parse
import pandas as pd
from pathlib import Path
from zoneinfo import ZoneInfo

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# ============ CONFIG ============
LOGIN_USERNAME = os.getenv("LI_USER") or "guillerminabacigalupo@hotmail.com"
LOGIN_PASSWORD = os.getenv("LI_PASS") or "holaManola1"

HEADLESS     = False
WAIT         = 20
TZ           = ZoneInfo("America/Argentina/Buenos_Aires")

NAV_PAUSE    = (1.2, 2.0)
SCROLL_PAUSE = (0.6, 1.2)
MAX_REACTORS = None     # None = todos; o número para limitar
# ===============================


def setup_driver():
    opts = webdriver.ChromeOptions()
    opts.add_argument("--start-maximized")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--lang=es-ES")
    if HEADLESS:
        opts.add_argument("--headless=new")
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=opts)
    driver.set_window_size(1366, 900)
    return driver


def login(driver):
    driver.get("https://www.linkedin.com/login")
    WebDriverWait(driver, WAIT).until(EC.presence_of_element_located((By.ID, "username"))).send_keys(LOGIN_USERNAME)
    driver.find_element(By.ID, "password").send_keys(LOGIN_PASSWORD, Keys.RETURN)
    try:
        WebDriverWait(driver, WAIT).until(
            EC.any_of(
                EC.presence_of_element_located((By.ID, "global-nav")),
                EC.presence_of_element_located((By.TAG_NAME, "main"))
            )
        )
    except:
        pass
    time.sleep(25)  # margen por si hay 2FA


def post_slug(post_url: str) -> str:
    parsed = urllib.parse.urlparse(post_url)
    base = re.sub(r"[^a-zA-Z0-9]+", "_", (parsed.path + parsed.query))[:80]
    return base.strip("_") or "post"


def open_reactions_modal(driver, post_url: str):
    """Abre la publicación y clickea el contador de reacciones para abrir el modal."""
    driver.get(post_url)
    time.sleep(random.uniform(*NAV_PAUSE))

    selectors = [
        "//button[@aria-label and (contains(@aria-label,'reacci') or contains(@aria-label,'reactions'))]",
        "//span[contains(@class,'social-details-social-counts__reactions-count')]/ancestor::button",
        "//button[contains(., 'reaccion') or contains(., 'reactions')]",
    ]
    for xp in selectors:
        try:
            btn = WebDriverWait(driver, 6).until(EC.element_to_be_clickable((By.XPATH, xp)))
            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btn)
            time.sleep(0.6)
            btn.click()
            WebDriverWait(driver, WAIT).until(
                EC.presence_of_element_located((By.XPATH, "//div[@role='dialog' or contains(@class,'artdeco-modal')]"))
            )
            return True
        except:
            continue
    raise RuntimeError("No pude abrir el modal de reacciones; verificá el link del post y el idioma de la UI.")


def scroll_modal_collect_reactors(driver):
    """Scrollea dentro del modal y junta los perfiles /in/ de la lista de reacciones."""
    modal = WebDriverWait(driver, WAIT).until(
        EC.presence_of_element_located((By.XPATH, "//div[@role='dialog' or contains(@class,'artdeco-modal')]"))
    )
    candidates = [
        ".//div[contains(@class,'artdeco-modal__content')]",
        ".//div[contains(@class,'scaffold-finite-scroll__content')]",
        ".//div[contains(@class,'reactions-list') or contains(@class,'reactors')]",
        ".//div[contains(@class,'artdeco-entity-lockup')]/ancestor::div[contains(@class,'scaffold')]"
    ]
    scrollable = None
    for xp in candidates:
        try:
            scrollable = modal.find_element(By.XPATH, xp)
            break
        except:
            continue
    if scrollable is None:
        scrollable = modal

    seen_urls = set()
    stable_count_streak = 0
    last_count = 0

    while True:
        links = modal.find_elements(By.XPATH, ".//a[contains(@href,'/in/') and @href]")
        for a in links:
            href = a.get_attribute("href") or ""
            m = re.search(r"https?://www\.linkedin\.com/in/[^/?#]+/?", href)
            if m:
                seen_urls.add(m.group(0))

        if len(seen_urls) == last_count:
            stable_count_streak += 1
        else:
            stable_count_streak = 0
        last_count = len(seen_urls)

        if MAX_REACTORS and len(seen_urls) >= MAX_REACTORS:
            break
        if stable_count_streak >= 5:
            break

        try:
            driver.execute_script(
                "arguments[0].scrollTop = arguments[0].scrollTop + arguments[0].clientHeight;", scrollable
            )
        except:
            driver.execute_script("window.scrollBy(0, Math.floor(window.innerHeight*0.7));")
        time.sleep(random.uniform(*SCROLL_PAUSE))

    return list(seen_urls)


def extract_user_name(driver) -> str:
    """Devuelve el <h1> del perfil (nombre del usuario)."""
    try:
        return WebDriverWait(driver, 6).until(EC.presence_of_element_located((By.XPATH, "//h1"))).text.strip()
    except:
        return ""


def extract_followed_companies(driver, profile_url: str):
    """
    Va a Intereses > Empresas del perfil y extrae únicamente los nombres principales (en negrita).
    """
    candidates = [
        profile_url.rstrip("/") + "/details/interests/companies/",
        profile_url.rstrip("/") + "/details/interests/?detailScreenTabIndex=1",
        profile_url.rstrip("/") + "/details/interests/",
    ]

    def clean_name(name: str) -> str:
        if not isinstance(name, str):
            return ""
        # Quitar espacios extra y números de seguidores
        name = re.sub(r"\s+", " ", name.strip())
        name = re.sub(r"\b\d[\d\.\,]*\s*seguidores?\b", "", name, flags=re.IGNORECASE)
        return name.strip()

    for url in candidates:
        try:
            driver.get(url)
            time.sleep(random.uniform(1.2, 2.0))

            # Click en la pestaña "Empresas" si existe
            try:
                tab = WebDriverWait(driver, 3).until(
                    EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Empresas') or contains(., 'Companies')]"))
                )
                driver.execute_script("arguments[0].scrollIntoView({block:'center'});", tab)
                time.sleep(0.3)
                tab.click()
                time.sleep(random.uniform(0.8, 1.2))
            except:
                pass

            companies = set()

            # 1️⃣ Extrae solo el texto en negrita o título principal de cada tarjeta
            name_xpaths = [
                "//div[contains(@class,'entity-result__content')]//span[contains(@class,'entity-result__title-text')]/a/span[1]",
                "//a[contains(@href,'/company/')]/span[1]",
                "//div[contains(@class,'artdeco-entity-lockup__title')]//a/span[1]",
                "//a[contains(@href,'/company/')]/strong",
            ]

            for xp in name_xpaths:
                elements = driver.find_elements(By.XPATH, xp)
                for el in elements:
                    txt = clean_name(el.text)
                    if txt:
                        companies.add(txt)

            if companies:
                return sorted(companies)

        except Exception as e:
            print(f"[WARN] error al intentar {url}: {e}")
            continue

    return []


def main():
    post_url = input("Pegá la URL de la publicación de LinkedIn: ").strip()
    out_name = f"empresas_{post_slug(post_url)}.csv"
    out_path = Path(out_name)

    driver = setup_driver()
    login(driver)

    rows = []
    try:
        if not open_reactions_modal(driver, post_url):
            raise RuntimeError("No se pudo abrir el modal de reacciones.")

        reactors = scroll_modal_collect_reactors(driver)
        print(f"Total de perfiles detectados en reacciones: {len(reactors)}")

        for i, prof in enumerate(reactors, 1):
            try:
                driver.get(prof)
                time.sleep(random.uniform(*NAV_PAUSE))

                usuario = extract_user_name(driver)
                empresas = extract_followed_companies(driver, prof)

                if empresas:
                    for emp in empresas:
                        rows.append({"Usuario": usuario, "Empresa": emp})
                else:
                    # Si no muestra empresas, no registramos fila (opcional)
                    pass

                if i % 10 == 0:
                    print(f"Procesados {i}/{len(reactors)} perfiles...")

                time.sleep(random.uniform(0.6, 1.2))

            except Exception as e:
                print(f"[WARN] Perfil {i} {prof}: {e}")
                continue

    finally:
        driver.quit()

    if rows:
        df = pd.DataFrame(rows, columns=["Usuario", "Empresa"])
        # Dedup dentro de esta corrida para evitar repeticiones obvias
        df.drop_duplicates(subset=["Usuario", "Empresa"], inplace=True)

        if out_path.exists():
            old = pd.read_csv(out_path)
            all_df = pd.concat([old, df], ignore_index=True)
        else:
            all_df = df

        all_df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"✅ Guardado: {out_path} (nuevas filas: {len(df)} | total: {len(all_df)})")
    else:
        print("No se recolectó información (0 filas).")


if __name__ == "__main__":
    main()
