import os, re, time, random, urllib.parse
import pandas as pd
from datetime import datetime, timedelta
# Si usás Python < 3.9:
# from backports.zoneinfo import ZoneInfo
from zoneinfo import ZoneInfo

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# ===================== CONFIG =====================
LOGIN_USERNAME = os.getenv("LI_USER") or "guillerminabacigalupo@hotmail.com"
LOGIN_PASSWORD = os.getenv("LI_PASS") or "holaManola1"

WAIT               = 20              # WebDriverWait (s)
SCROLL_PAUSE       = (1.2, 2.6)      # pausa aleatoria entre scrolls
HEADLESS           = False
TZ                 = ZoneInfo("America/Argentina/Salta")

# “Raspar más” y ser pacientes con el lazy-load
MAX_SCROLLS          = 600           # profundidad total de scroll
STALL_TOLERANCE      = 14            # fallback general (anti-estancamiento ya existente)
TRY_LOAD_MORE_EACH   = 3             # cada N scrolls intenta clickear “Mostrar más”
MICRO_SCROLL_STEPS   = 6             # micro-scrolls por ciclo
MICRO_SCROLL_PAUSE   = (0.35, 0.6)   # pausa por micro-scroll
WAIT_NEW_POSTS_SECS  = 12            # esperar hasta Xs a que aparezcan posts nuevos tras llegar al fondo
RENDER_RETRIES       = 3             # reintentos para forzar el render de un post
PER_POST_SETTLE      = (0.35, 0.7)   # pausa tras centrar cada post

# NUEVOS UMBRALES de corte por no-crecimiento
DOM_STALL_PATIENCE   = 6             # si el DOM no crece en 6 scrolls seguidos -> cortar
ADDED_STALL_PATIENCE = 4             # si no se agregan posts en 4 scrolls seguidos -> cortar
# ==================================================

# =============== UTIL: números ====================
ONLY_NUM_RE = re.compile(r"[\d]+(?:[.,\u202F\u00A0]\d+)*")

def normalize_count_number(s: str) -> int:
    if not s:
        return 0
    m = ONLY_NUM_RE.search(s)
    if not m:
        return 0
    num_str = m.group(0)
    num_str = num_str.replace("\u00A0", "").replace("\u202F", "").replace(" ", "")
    num_str = re.sub(r"[^\d]", "", num_str)
    return int(num_str) if num_str.isdigit() else 0

def exact_followers(s: str) -> int:
    digits = re.findall(r"\d", s or "")
    return int("".join(digits)) if digits else 0
# ==================================================

# =============== UTIL: fechas =====================
TIME_RE = re.compile(
    r"(?P<num>\d+)\s*(?P<Unit>min|minutos|hora|horas|h|día|días|semana|semanas|mes|meses|año|años)",
    re.IGNORECASE
)

def relative_to_absolute_date(text: str, now_dt: datetime) -> datetime:
    if not text:
        return now_dt
    t = text.lower()
    m = TIME_RE.search(t)
    if not m:
        return now_dt
    n = int(m.group("num"))
    unit = m.group("Unit")
    delta = timedelta()
    if unit.startswith("min"):
        delta = timedelta(minutes=n)
    elif unit in ("h", "hora", "horas"):
        delta = timedelta(hours=n)
    elif unit in ("día", "días"):
        delta = timedelta(days=n)
    elif unit in ("semana", "semanas"):
        delta = timedelta(weeks=n)
    elif unit in ("mes", "meses"):
        delta = timedelta(days=30*n)
    elif unit in ("año", "años"):
        delta = timedelta(days=365*n)
    return now_dt - delta

def format_ddmmyyyy(dt: datetime) -> str:
    return dt.astimezone(TZ).strftime("%d/%m/%Y")
# ==================================================

# ============== SELENIUM: setup/login =============
def setup_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--lang=es-ES")
    if HEADLESS:
        options.add_argument("--headless=new")
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
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
    print("⚠️ Si aparece 2FA, ingresalo manualmente (30s).")
    time.sleep(30)
# ==================================================

# ======= Navegar a la sección de Publicaciones =====
def open_posts_section(driver, base_url: str):
    def has_posts():
        try:
            driver.find_element(By.XPATH, "//div[@data-urn and contains(@data-urn,'urn:li:activity:')]")
            return True
        except:
            return False

    driver.get(base_url)
    time.sleep(2)

    tab_xpaths = [
        "//a[contains(., 'Publicaciones')]",
        "//a[contains(., 'Posts')]",
        "//a[contains(., 'Actividad')]",
        "//a[contains(@href, 'recent-activity')]",
        "//a[contains(@href, '/posts')]",
    ]
    for xp in tab_xpaths:
        try:
            el = WebDriverWait(driver, 4).until(EC.element_to_be_clickable((By.XPATH, xp)))
            href = el.get_attribute("href")
            if href:
                driver.get(href)
            else:
                el.click()
            time.sleep(2)
            if has_posts():
                return
        except:
            pass

    candidates = [
        base_url.rstrip("/") + "/recent-activity/all/",
        base_url.rstrip("/") + "/posts/",
        base_url.rstrip("/") + "/?feedView=all",
    ]
    for url in candidates:
        driver.get(url)
        try:
            WebDriverWait(driver, 6).until(
                EC.presence_of_element_located((By.XPATH, "//div[@data-urn and contains(@data-urn,'urn:li:activity:')]"))
            )
            return
        except:
            continue

    WebDriverWait(driver, WAIT).until(
        EC.presence_of_element_located((By.XPATH, "//div[@data-urn and contains(@data-urn,'urn:li:activity:')]"))
    )
# ==================================================

# ====== “Raspar más”: botón Mostrar/Ver/Load more =====
def try_click_load_more(driver):
    labels = [
        "Mostrar más", "Ver más", "Cargar más",
        "Show more", "Load more", "See more"
    ]
    for label in labels:
        try:
            btn = WebDriverWait(driver, 2).until(
                EC.element_to_be_clickable((By.XPATH, f"//button[contains(., '{label}')]"))
            )
            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btn)
            time.sleep(0.5)
            btn.click()
            time.sleep(random.uniform(1.0, 1.5))
            return True
        except:
            continue
    return False
# ==================================================

# =============== SCROLL SUAVE Y ESPERAS ===============
def smooth_scroll_chunk(driver):
    for _ in range(MICRO_SCROLL_STEPS):
        driver.execute_script("window.scrollBy(0, Math.floor(window.innerHeight*0.5));")
        time.sleep(random.uniform(*MICRO_SCROLL_PAUSE))

def wait_for_new_posts(driver, previous_count, timeout=WAIT_NEW_POSTS_SECS):
    end = time.time() + timeout
    while time.time() < end:
        count = len(driver.find_elements(By.XPATH, "//div[@data-urn and contains(@data-urn,'urn:li:activity:')]"))
        if count > previous_count:
            return True
        time.sleep(0.8)
    return False

def ensure_post_rendered(driver, div):
    for _ in range(RENDER_RETRIES):
        try:
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", div)
            time.sleep(random.uniform(*PER_POST_SETTLE))
            div.find_element(By.XPATH, ".//li[contains(@class,'social-details-social-counts__item')]")
            return True
        except:
            driver.execute_script("window.scrollBy(0, -Math.floor(window.innerHeight*0.2));")
            time.sleep(0.3)
            driver.execute_script("window.scrollBy(0, Math.floor(window.innerHeight*0.25));")
            time.sleep(0.4)
    return False
# =======================================================

# ================== EXTRACCIONES ==================
def get_profile_name(driver, fallback_url: str) -> str:
    try:
        return driver.find_element(By.TAG_NAME, "h1").text.strip()
    except:
        return fallback_url

def get_followers_count(driver) -> int:
    count = 0
    try:
        spans = driver.find_elements(By.XPATH, "//span[@aria-hidden='true' and contains(., 'seguidores')]")
        for sp in spans:
            count = max(count, exact_followers(sp.text))
    except:
        pass
    return count

def extract_reactions(div) -> int:
    try:
        span = div.find_element(By.XPATH, ".//span[contains(@class,'social-details-social-counts__reactions-count')]")
        return normalize_count_number(span.text)
    except:
        try:
            btn = div.find_element(By.XPATH, ".//li[contains(@class,'social-details-social-counts__reactions')]//button[@aria-label]")
            return normalize_count_number(btn.get_attribute("aria-label") or "")
        except:
            return 0

def extract_comments(div) -> int:
    try:
        span = div.find_element(By.XPATH, ".//li[contains(@class,'social-details-social-counts__comments')]//span[@aria-hidden='true' and contains(normalize-space(.), 'comentario')]")
        return normalize_count_number(span.text)
    except:
        try:
            btn = div.find_element(By.XPATH, ".//li[contains(@class,'social-details-social-counts__comments')]//button[@aria-label]")
            return normalize_count_number(btn.get_attribute("aria-label") or "")
        except:
            return 0

def extract_shares(div) -> int:
    try:
        span = div.find_element(By.XPATH, ".//li[.//span[@aria-hidden='true' and contains(., 'compartid')]]//span[@aria-hidden='true']")
        return normalize_count_number(span.text)
    except:
        try:
            btn = div.find_element(By.XPATH, ".//li[.//button[@aria-label and contains(., 'compartid')]]//button[@aria-label]")
            return normalize_count_number(btn.get_attribute("aria-label") or "")
        except:
            return 0

def extract_created_at(div) -> str:
    candidates = []
    try:
        candidates = div.find_elements(By.XPATH, ".//span[@aria-hidden='true' and contains(., '•')]")
    except:
        candidates = []
    if not candidates:
        try:
            candidates = div.find_elements(
                By.XPATH,
                ".//span[@aria-hidden='true' and (contains(., 'min') or contains(., 'hora') or contains(., 'día') or contains(., 'semana') or contains(., 'mes') or contains(., 'año') or contains(., ' h '))]"
            )
        except:
            pass

    now_dt = datetime.now(TZ)
    for sp in candidates:
        raw = sp.text.strip()
        if not raw:
            continue
        abs_dt = relative_to_absolute_date(raw, now_dt)
        if abs_dt:
            return format_ddmmyyyy(abs_dt)
    return format_ddmmyyyy(now_dt)
# ==================================================

# ========== NOMBRE DE ARCHIVO POR CUENTA ==========
def output_csv_for_profile(profile_url: str) -> str:
    try:
        path = urllib.parse.urlparse(profile_url).path.strip("/")
        parts = [p for p in path.split("/") if p]
        slug = parts[1] if len(parts) >= 2 else parts[0]
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "", slug).lower()
        if not slug:
            slug = "perfil"
        return f"linkedin_{slug}.csv"
    except:
        return "linkedin_perfil.csv"
# ==================================================

# ================== SCRAPE POSTS ==================
def scrape_posts(driver, profile_url):
    open_posts_section(driver, profile_url)

    profile_id = get_profile_name(driver, profile_url)
    followers  = get_followers_count(driver)

    posts, seen = [], set()
    scrolls, last_seen, stall = 0, 0, 0

    # NUEVAS variables para corte por estancamiento
    prev_dom_count = 0
    dom_no_growth_streak = 0

    prev_seen_count = 0
    added_none_streak = 0

    while scrolls < MAX_SCROLLS:
        post_divs = driver.find_elements(By.XPATH, "//div[@data-urn and contains(@data-urn,'urn:li:activity:')]")
        current_dom_count = len(post_divs)
        print(f"Scroll {scrolls+1}: {current_dom_count} posts en DOM")

        for div in post_divs:
            try:
                urn = div.get_attribute("data-urn") or ""
                if not urn or urn in seen:
                    continue
                seen.add(urn)

                ensure_post_rendered(driver, div)

                try:
                    text_elem = div.find_element(By.XPATH, ".//div[contains(@class,'update-components-text')]")
                    post_text = text_elem.text.strip()
                except:
                    post_text = ""

                reactions  = extract_reactions(div)
                comments   = extract_comments(div)
                shares     = extract_shares(div)
                created_at = extract_created_at(div)

                posts.append({
                    "profile": profile_id,
                    "text": post_text,
                    "reactions": reactions,
                    "comments": comments,
                    "shares": shares,
                    "created_at": created_at,
                    "followers": followers  # al final del CSV
                })

            except Exception as e:
                print(f"Post error: {e}")

        # ===== CONTROL: No crecimiento del DOM =====
        if current_dom_count <= prev_dom_count:
            dom_no_growth_streak += 1
        else:
            dom_no_growth_streak = 0
        prev_dom_count = current_dom_count

        # ===== CONTROL: No se agregan posts =====
        added = len(seen) - prev_seen_count
        if added <= 0:
            added_none_streak += 1
        else:
            added_none_streak = 0
        prev_seen_count = len(seen)

        # Cortes por estancamiento "duro"
        if dom_no_growth_streak >= DOM_STALL_PATIENCE:
            print(f"No crece el DOM en {DOM_STALL_PATIENCE} scrolls seguidos. Corto.")
            break
        if added_none_streak >= ADDED_STALL_PATIENCE:
            print(f"No se agregan posts en {ADDED_STALL_PATIENCE} scrolls seguidos. Corto.")
            break

        # Scroll suave + pausas
        smooth_scroll_chunk(driver)
        time.sleep(random.uniform(*SCROLL_PAUSE))
        scrolls += 1

        # cada N scrolls intenta botón "Mostrar más / Ver más"
        if scrolls % TRY_LOAD_MORE_EACH == 0:
            try_click_load_more(driver)

        # Esperar a que aparezcan posts nuevos tras llegar al fondo
        if not wait_for_new_posts(driver, current_dom_count, timeout=WAIT_NEW_POSTS_SECS):
            # si no aparecieron, aplicar estrategia anti-estancamiento general
            stall += 1
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(random.uniform(1.2, 1.8))
            driver.execute_script("window.scrollBy(0, -Math.floor(window.innerHeight*0.5));")
            time.sleep(random.uniform(0.6, 1.0))
            try_click_load_more(driver)
            if stall >= STALL_TOLERANCE:
                print("No hay más contenido para cargar (límite de estancamiento).")
                break
        else:
            stall = 0

        last_seen = len(seen)

    print(f"Total posts extraídos: {len(posts)}")
    return posts
# ==================================================

# ======================= MAIN =====================
if __name__ == "__main__":
    profile = input("Ingrese la URL del perfil de LinkedIn: ").strip()

    driver = setup_driver()
    login(driver)
    try:
        posts = scrape_posts(driver, profile)
    except Exception as e:
        print(f"Error con {profile}: {e}")
        posts = []
    finally:
        driver.quit()

    if posts:
        df_new = pd.DataFrame(posts)
        cols = ["profile","text","reactions","comments","shares","created_at","followers"]
        df_new = df_new[cols]

        OUTPUT_CSV = output_csv_for_profile(profile)

        if os.path.exists(OUTPUT_CSV):
            df_old = pd.read_csv(OUTPUT_CSV)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_all = df_new

        df_all.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
        print(f"Guardados {len(df_new)} posts nuevos en {OUTPUT_CSV} (total ahora: {len(df_all)})")
    else:
        print("No se guardaron posts.")
