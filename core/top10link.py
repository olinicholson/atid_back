# -*- coding: utf-8 -*-
import os, re, time, random, urllib.parse
import pandas as pd
from pathlib import Path
from zoneinfo import ZoneInfo

from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
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
WAIT         = 18
TZ           = ZoneInfo("America/Argentina/Buenos_Aires")

NAV_PAUSE    = (1.0, 1.8)
SCROLL_PAUSE = (0.6, 1.2)
MAX_REACTORS = None       # None = todos

# Anti cuelgues / watchdogs
PAGELOAD_TIMEOUT       = 20   # seg, por cada .get()
SCRIPT_TIMEOUT         = 20   # seg, scripts JS
SAFEGET_RETRIES        = 2    # reintentos para .get()
PROFILE_HARD_TIMEOUT   = 45   # seg máximos por perfil (todo el ciclo)
INTERESTS_DEADLINE     = 20   # seg máximos solo para abrir/leer Intereses
INTERESTS_RETRIES      = 1    # reintentos para abrir Intereses
COOLDOWN_EVERY_N       = 10   # cooldown cada N perfiles
COOLDOWN_RANGE         = (5.0, 9.0)
# ===============================


def setup_driver():
    opts = webdriver.ChromeOptions()
    opts.add_argument("--start-maximized")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--lang=es-ES")
    opts.add_argument("--disable-features=NetworkService")
    opts.add_argument("--disable-dev-shm-usage")
    opts.page_load_strategy = "eager"   # no espera imágenes/ads
    if HEADLESS:
        opts.add_argument("--headless=new")

    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=opts)
    driver.set_window_size(1366, 900)
    driver.set_page_load_timeout(PAGELOAD_TIMEOUT)
    driver.set_script_timeout(SCRIPT_TIMEOUT)
    return driver


def safe_get(driver, url, *, retries=SAFEGET_RETRIES, timeout=PAGELOAD_TIMEOUT):
    """Navega con timeout; si cuelga, aborta la carga y reintenta. Devuelve True/False."""
    for attempt in range(retries + 1):
        try:
            driver.set_page_load_timeout(timeout)
            driver.get(url)
            return True
        except TimeoutException:
            try:
                driver.execute_script("window.stop();")
            except WebDriverException:
                pass
            if attempt >= retries:
                print(f"[SAFE_GET] Timeout en {url}. Saltando.")
                return False
            time.sleep(1.0 + attempt)
        except WebDriverException as e:
            if attempt >= retries:
                print(f"[SAFE_GET] Error '{e}' en {url}. Saltando.")
                return False
            time.sleep(1.0 + attempt)
    return False


def login(driver):
    if not safe_get(driver, "https://www.linkedin.com/login"):
        raise RuntimeError("No se pudo abrir la página de login.")
    WebDriverWait(driver, WAIT).until(EC.presence_of_element_located((By.ID, "username"))).send_keys(LOGIN_USERNAME)
    driver.find_element(By.ID, "password").send_keys(LOGIN_PASSWORD, Keys.RETURN)
    try:
        WebDriverWait(driver, WAIT).until(
            EC.any_of(
                EC.presence_of_element_located((By.ID, "global-nav")),
                EC.presence_of_element_located((By.TAG_NAME, "main"))
            )
        )
    except TimeoutException:
        pass
    time.sleep(20)  # margen 2FA manual


def post_slug(post_url: str) -> str:
    parsed = urllib.parse.urlparse(post_url)
    base = re.sub(r"[^a-zA-Z0-9]+", "_", (parsed.path + parsed.query))[:80]
    return base.strip("_") or "post"


def open_reactions_modal(driver, post_url: str):
    if not safe_get(driver, post_url):
        raise RuntimeError("No pude abrir el post.")
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
            time.sleep(0.5)
            btn.click()
            WebDriverWait(driver, WAIT).until(
                EC.presence_of_element_located((By.XPATH, "//div[@role='dialog' or contains(@class,'artdeco-modal')]"))
            )
            return True
        except TimeoutException:
            continue
    raise RuntimeError("No pude abrir el modal de reacciones.")


def scroll_modal_collect_reactors(driver):
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

    seen_urls, last_count, stable = set(), 0, 0
    while True:
        links = modal.find_elements(By.XPATH, ".//a[contains(@href,'/in/') and @href]")
        for a in links:
            href = a.get_attribute("href") or ""
            m = re.search(r"https?://www\.linkedin\.com/in/[^/?#]+/?", href)
            if m: seen_urls.add(m.group(0))

        stable = stable + 1 if len(seen_urls) == last_count else 0
        last_count = len(seen_urls)

        if MAX_REACTORS and len(seen_urls) >= MAX_REACTORS: break
        if stable >= 5: break

        try:
            driver.execute_script(
                "arguments[0].scrollTop = arguments[0].scrollTop + arguments[0].clientHeight;", scrollable
            )
        except:
            driver.execute_script("window.scrollBy(0, Math.floor(window.innerHeight*0.7));")
        time.sleep(random.uniform(*SCROLL_PAUSE))

    return list(seen_urls)


def extract_user_name(driver) -> str:
    try:
        return WebDriverWait(driver, 6).until(EC.presence_of_element_located((By.XPATH, "//h1"))).text.strip()
    except TimeoutException:
        return ""


def extract_followed_companies(driver, profile_url: str, deadline_ts: float = None):
    """
    Intereses > Empresas: devuelve SOLO el nombre en negrita/título (span[aria-hidden='true'] bajo div.t-bold).
    Acepta deadline_ts opcional para cortar temprano si se excede.
    """
    candidates = [
        profile_url.rstrip("/") + "/details/interests/companies/",
        profile_url.rstrip("/") + "/details/interests/?detailScreenTabIndex=1",
        profile_url.rstrip("/") + "/details/interests/",
    ]
    name_xpaths_primary = [
        ("//div[@data-view-name='profile-component-entity']"
         "//a[@data-field='active_tab_companies_interests' and contains(@href,'/company/')]"
         "//div[contains(@class,'t-bold')]//span[@aria-hidden='true'][1]")
    ]
    name_xpaths_fallbacks = [
        ("//div[@data-view-name='profile-component-entity']"
         "//a[@data-field='active_tab_companies_interests' and contains(@href,'/company/')]//span[@aria-hidden='true'][1]"),
        "//div[contains(@class,'artdeco-entity-lockup__title')]//a[contains(@href,'/company/')]/span[@aria-hidden='true'][1]",
        ("//div[contains(@class,'entity-result__content')]"
         "//span[contains(@class,'entity-result__title-text')]/a/span[1]"),
    ]

    def time_left(default=20) -> float:
        if deadline_ts is None:
            return default
        return max(0.0, min(default, deadline_ts - time.time()))

    def clean_name(s: str) -> str:
        if not isinstance(s, str): return ""
        s = re.sub(r"\s+", " ", s.strip())
        s = re.sub(r"\b\d[\d\.\,]*\s*seguidores?\b", "", s, flags=re.IGNORECASE).strip()
        return s

    companies = set()
    hard_end = time.time() + time_left(20)

    for url in candidates:
        # navegación con timeout “suave”
        if time.time() > hard_end: break
        if not safe_get(driver, url, timeout=max(5, int(time_left()))):
            continue
        time.sleep(random.uniform(1.0, 1.8))

        # tab Empresas (si existe)
        try:
            tab = WebDriverWait(driver, max(2, int(min(4, time_left())))).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Empresas') or contains(., 'Companies')]"))
            )
            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", tab)
            time.sleep(0.3); tab.click(); time.sleep(0.8)
        except Exception:
            pass

        found_any = False

        # Primario
        for xp in name_xpaths_primary:
            elems = driver.find_elements(By.XPATH, xp)
            if elems: found_any = True
            for el in elems:
                txt = clean_name(el.text)
                if txt: companies.add(txt)

        # Fallbacks si no hay nada
        if not companies:
            for xp in name_xpaths_fallbacks:
                elems = driver.find_elements(By.XPATH, xp)
                if elems: found_any = True
                for el in elems:
                    txt = clean_name(el.text)
                    if txt: companies.add(txt)

        if companies or found_any:
            break

    return sorted(companies)

def main():
    post_url = input("Pegá la URL de la publicación de LinkedIn: ").strip()
    out_path = Path("intereses_empresas.csv")   # nombre fijo pedido

    driver = setup_driver()
    login(driver)

    rows = []
    try:
        open_reactions_modal(driver, post_url)
        reactors = scroll_modal_collect_reactors(driver)
        print(f"Total de perfiles detectados en reacciones: {len(reactors)}")

        for i, prof in enumerate(reactors, 1):
            profile_deadline = time.time() + PROFILE_HARD_TIMEOUT
            try:
                if not safe_get(driver, prof, timeout=min(PAGELOAD_TIMEOUT, int(PROFILE_HARD_TIMEOUT/2))):
                    print(f"[Perfil] no abre: {prof} -> salto")
                    continue

                usuario = extract_user_name(driver)
                empresas = extract_followed_companies(driver, prof, deadline_ts=profile_deadline)

                if empresas:
                    for emp in empresas:
                        rows.append({"Usuario": usuario, "Empresa": emp})

                if i % COOLDOWN_EVERY_N == 0:
                    cool = random.uniform(*COOLDOWN_RANGE)
                    print(f"[Cooldown] Pausa {cool:.1f}s para evitar rate-limit…")
                    time.sleep(cool)

                if i % 10 == 0:
                    print(f"Procesados {i}/{len(reactors)} perfiles...")

            except Exception as e:
                print(f"[WARN] Perfil {i} {prof}: {e}")
            finally:
                # si el perfil consumió demasiado tiempo, informar (ya se forzó por deadline interno)
                if time.time() > profile_deadline:
                    print(f"[WATCHDOG] Perfil {i} excedió {PROFILE_HARD_TIMEOUT}s. Siguiente.")

    finally:
        driver.quit()

    if rows:
        df = pd.DataFrame(rows, columns=["Usuario", "Empresa"])
        df.drop_duplicates(subset=["Usuario", "Empresa"], inplace=True)

        # Si el archivo ya existe, lo leemos y concatenamos
        if out_path.exists():
            old = pd.read_csv(out_path)
            combined = pd.concat([old, df], ignore_index=True)
            # Eliminar duplicados globales (por Usuario y Empresa)
            combined.drop_duplicates(subset=["Usuario", "Empresa"], inplace=True)
        else:
            combined = df

        # Guardar el archivo actualizado sin pisar lo previo
        combined.to_csv(out_path, index=False, encoding="utf-8")

        print(f"✅ Datos agregados correctamente.")
        print(f"   Nuevas filas: {len(df)} | Total acumulado: {len(combined)}")
    else:
        print("No se recolectó información (0 filas).")

if __name__ == "__main__":
    main()
