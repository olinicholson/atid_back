import os
import glob
import json
import time
import re
from datetime import datetime

import pandas as pd
try:
    from rapidfuzz import fuzz
    _HAS_RAPIDFUZZ = True
except Exception:
    fuzz = None
    _HAS_RAPIDFUZZ = False
    import difflib

try:
    from bs4 import BeautifulSoup
    _HAS_BS4 = True
except Exception:
    _HAS_BS4 = False


# Ensure repo root is on sys.path so `core` package imports work when the script is run directly
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.trends.trendhist import setup_driver, scrape_historical_trends, scrape_trends_for_date
from core.trends.similarity import match_post_trend

CACHE_PATH = os.path.join(os.path.dirname(__file__), "trend_cache.json")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")  # ../data

# Stats about scraping per date
SCRAPE_STATS = {}

def record_scrape(date_str, duration_secs, from_cache=False, html_path=None):
    """Record that a given date was obtained either from cache or scraped, with duration.
    Optionally record path to saved HTML."""
    if date_str is None:
        return
    if date_str in SCRAPE_STATS:
        return
    SCRAPE_STATS[date_str] = {
        'from_cache': bool(from_cache),
        'duration_secs': float(duration_secs),
        'html_saved': bool(html_path),
        'html_path': html_path
    }

def save_stats():
    stats_path = os.path.join(os.path.dirname(__file__), 'trend_scrape_stats.json')
    try:
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(SCRAPE_STATS, f, indent=2, ensure_ascii=False)
        print(f"Stats guardadas en: {stats_path}")
    except Exception as e:
        print(f"No se pudo guardar stats: {e}")


def load_cache():
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_cache(cache):
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_best_trend_and_score(post_text, trend_names):
    """
    Compute best matching trend and score. Uses rapidfuzz if available, otherwise falls back to difflib.
    Returns (best_trend, score) with score in range 0-100.
    """
    if not trend_names:
        return "", 0.0
    post_norm = normalize_text(post_text)
    trends_norm = [normalize_text(t) for t in trend_names]

    scores = []
    if _HAS_RAPIDFUZZ and fuzz is not None:
        try:
            scores = [fuzz.partial_ratio(post_norm, t) for t in trends_norm]
        except Exception:
            scores = []

    if not scores:
        # fallback to difflib SequenceMatcher (returns 0..1), scale to 0..100
        for t in trends_norm:
            try:
                r = difflib.SequenceMatcher(None, post_norm, t).ratio()
                scores.append(r * 100.0)
            except Exception:
                scores.append(0.0)

    max_score = max(scores)
    best_idx = scores.index(max_score)
    best_trend = trend_names[best_idx]
    return best_trend, float(max_score)


def date_hour_from_iso(iso_str):
    # Expecting ISO like 2025-09-19T20:08:28.000Z
    try:
        dt = pd.to_datetime(iso_str, utc=True)
        # pd.to_datetime can return NaT for invalid/missing values - handle that
        if pd.isna(dt):
            return None, None
    except Exception:
        # fallback
        try:
            dt = datetime.fromisoformat(iso_str)
        except Exception:
            return None, None
    # Format date to DD-MM-YYYY as trendhist expects
    try:
        date_str = dt.strftime("%d-%m-%Y")
        hour_str = dt.strftime("%H:%M")
    except Exception:
        return None, None
    return date_str, hour_str


def process_file(file_path, driver, cache, overwrite=False, max_delta_minutes=60, batch_size=200, save_html=False, page_delay=1.0):
    print(f"Procesando: {file_path}")
    df = pd.read_csv(file_path)

    if 'text' not in df.columns or 'created_at' not in df.columns:
        print(f"  -> Archivo {file_path} no tiene columnas 'text' o 'created_at'. Se salta.")
        return

    # Prepare output DataFrame and columns, and check output existence early
    out_path = file_path.replace('.csv', '_with_trends.csv')
    if os.path.exists(out_path) and not overwrite:
        print(f"  -> Archivo de salida {out_path} ya existe. No se sobreescribe (use overwrite=True).")
        return

    output_df = df.copy()
    output_df['best_trend'] = ""
    output_df['trend_similarity'] = 0.0
    output_df['best_trend_position'] = None

    # Build list of unique dates and post hour-times
    dh_list = []
    for idx, ca in enumerate(df['created_at']):
        date_str, hour_str = date_hour_from_iso(ca)
        if date_str is None:
            dh_list.append((None, None))
        else:
            dh_list.append((date_str, hour_str))

    # Target hours per date (fixed 10:00, 14:00, 20:00) - can be customized via main.target_hours
    target_hours = getattr(main, 'target_hours', ["10:00", "14:00", "20:00"])[:]

    # For each needed date, ensure we have the three target hour lists cached
    unique_dh = {}
    needed_dates = sorted({d for d, _ in dh_list if d})
    for date_str in needed_dates:
        # For each target hour, check cache key
        missing_hours = [th for th in target_hours if f"{date_str}|{th}" not in cache]
        if not missing_hours:
            # load from cache into unique_dh
            for th in target_hours:
                key = f"{date_str}|{th}"
                unique_dh[(date_str, th)] = cache.get(key, [])
            # record cache usage
            record_scrape(date_str, 0.0, from_cache=True, html_path=cache.get(f"{date_str}|__html"))
            continue

        # If driver is None we cannot scrape missing hours -> leave empty lists
        if driver is None:
            print(f"Driver no disponible y faltan horas para {date_str}: {missing_hours}. Se asignarán listas vacías.")
            for th in target_hours:
                key = f"{date_str}|{th}"
                unique_dh[(date_str, th)] = cache.get(key, [])
            continue

        # For this date try (1) parse saved HTML in cache, (2) scrape via Selenium if parsing not possible
        try:
            blocks_map = None
            parsed_from_html = False

            # helper: parse trends from saved HTML (returns dict hour->list of trends)
            def parse_trends_from_html(html_text):
                result = {}
                try:
                    if _HAS_BS4:
                        soup = BeautifulSoup(html_text, 'html.parser')
                        bloques = soup.find_all(class_=re.compile(r'.*tek_tablo.*'))
                        for b in bloques:
                            hora_elem = b.find(class_=re.compile(r'.*trend_baslik.*'))
                            if not hora_elem:
                                continue
                            hora = hora_elem.get_text(strip=True)
                            rows = b.select('table tr')
                            trend_list = []
                            for r in rows:
                                nombre_el = r.find(class_=re.compile(r'.*trend611.*'))
                                pos_el = r.find(class_=re.compile(r'.*sira611.*'))
                                vol_el = r.find(class_=re.compile(r'.*volume.*'))
                                if nombre_el:
                                    nombre = nombre_el.get_text(strip=True)
                                    posicion = pos_el.get_text(strip=True) if pos_el else ''
                                    volumen = vol_el.get_text(strip=True) if vol_el else ''
                                    trend_list.append({'posicion': posicion, 'nombre': nombre, 'volumen': volumen})
                            if trend_list:
                                result[hora] = trend_list
                    else:
                        # very simple fallback: find hours and trend names via regex
                        hours = re.findall(r'>(\d{1,2}:\d{2})<', html_text)
                        trend_names = re.findall(r'class=["\']?trend611["\']?[^>]*>([^<]+)<', html_text, flags=re.I)
                        if hours:
                            h = hours[0]
                            result[h] = [{'posicion': '', 'nombre': re.sub(r'<[^>]+>', '', t).strip(), 'volumen': ''} for t in trend_names]
                except Exception:
                    return {}
                return result

            # 1) try parse saved HTML if present in cache
            html_cache_key = f"{date_str}|__html"
            html_path = cache.get(html_cache_key)
            if html_path and os.path.exists(html_path):
                try:
                    with open(html_path, 'r', encoding='utf-8') as hf:
                        html_text = hf.read()
                    pm = parse_trends_from_html(html_text)
                    if pm:
                        blocks_map = pm
                        parsed_from_html = True
                        duration = 0.0
                        record_scrape(date_str, duration, from_cache=True, html_path=html_path)
                except Exception as e:
                    print(f"  Warning: error parsing saved HTML for {date_str}: {e}")

            # 2) if not parsed from HTML, scrape the date page once
            if not blocks_map:
                t0 = time.time()
                blocks_map = scrape_trends_for_date(driver, date_str)
                duration = time.time() - t0

                # Optionally save HTML of scraped page
                if save_html:
                    try:
                        html = driver.page_source
                        html_fname = f"trend_{date_str}_{int(time.time())}.html"
                        html_path = os.path.join(os.path.dirname(__file__), html_fname)
                        with open(html_path, 'w', encoding='utf-8') as hf:
                            hf.write(html)
                        cache[f"{date_str}|__html"] = html_path
                    except Exception as e:
                        print(f"  Warning: no se pudo guardar HTML para {date_str}: {e}")

            # For each target hour, pick nearest block in blocks_map and cache only those lists
            for th in target_hours:
                selected = []
                if blocks_map and th in blocks_map:
                    selected = blocks_map[th]
                else:
                    # find nearest hour key within tolerance
                    candidates = []
                    if blocks_map:
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

                    m2 = re.search(r"(\d{1,2}:\d{2})", th)
                    if m2 and candidates:
                        try:
                            target_t = datetime.strptime(m2.group(1), "%H:%M")
                            target_minutes = target_t.hour * 60 + target_t.minute
                            best = min(candidates, key=lambda x: abs(x[0] - target_minutes))
                            diff = abs(best[0] - target_minutes)
                            if diff <= max_delta_minutes:
                                chosen_raw = best[1]
                                selected = blocks_map.get(chosen_raw, [])
                            else:
                                selected = []
                        except Exception:
                            selected = []
                    else:
                        selected = []

                key = f"{date_str}|{th}"
                cache[key] = selected
                unique_dh[(date_str, th)] = selected

            save_cache(cache)
            if not parsed_from_html:
                record_scrape(date_str, duration, from_cache=False, html_path=cache.get(f"{date_str}|__html"))
            time.sleep(page_delay)
        except Exception as e:
            print(f"  Error scrapping/parsing date {date_str}: {e}")
            for th in target_hours:
                unique_dh[(date_str, th)] = cache.get(f"{date_str}|{th}", [])

    # Now compute best trend for each post and save incrementally
    total = len(df)
    for idx, (date_str, hour_str) in enumerate(dh_list):
        text = df.at[idx, 'text']
        if date_str is None:
            output_df.at[idx, 'best_trend'] = ""
            output_df.at[idx, 'trend_similarity'] = 0.0
            output_df.at[idx, 'best_trend_position'] = None
        else:
            trends_data = unique_dh.get((date_str, hour_str), [])
            trend_names = [t.get('nombre', '') for t in trends_data]

            # Binary matching helper: returns (best_trend_or_empty, 1/0)
            def tokens_set(s):
                s = str(s).lower()
                s = re.sub(r"[^a-z0-9#\s]", " ", s)
                return set(re.findall(r"#?[a-z0-9]{2,}", s))

            # Use permissive matching via match_post_trend when available.
            matched = False
            matched_trend = ""
            match_mode = getattr(main, 'match_mode', 'substring')
            fuzzy_threshold = getattr(main, 'fuzzy_threshold', 60)
            min_token_len = getattr(main, 'min_token_len', 3)

            for t in trend_names:
                if not t:
                    continue
                try:
                    is_match = match_post_trend(text, t, mode=match_mode, fuzzy_threshold=fuzzy_threshold, min_token_len=min_token_len)
                except Exception:
                    # fallback to simple token intersection
                    is_match = bool(tokens_set(text) & tokens_set(t))
                if is_match:
                    matched = True
                    matched_trend = t
                    break

            output_df.at[idx, 'best_trend'] = matched_trend
            output_df.at[idx, 'trend_similarity'] = 1.0 if matched else 0.0
            # get position if available
            pos = None
            if matched_trend:
                for t in trends_data:
                    if t.get('nombre') == matched_trend:
                        pos = t.get('posicion')
                        break
            output_df.at[idx, 'best_trend_position'] = pos

        # Save progress every batch_size rows
        if (idx + 1) % batch_size == 0 or (idx + 1) == total:
            try:
                output_df.to_csv(out_path, index=False, encoding='utf-8')
                print(f"  Guardado intermedio: {out_path} ({idx+1}/{total})")
            except Exception as e:
                print(f"  Error guardando progreso en {out_path}: {e}")

    # Final save (already saved in loop but ensure written)
    try:
        output_df.to_csv(out_path, index=False, encoding='utf-8')
        print(f"  -> Guardado final: {out_path}")
    except Exception as e:
        print(f"  Error guardando archivo final {out_path}: {e}")


def main(overwrite=False, limit_files=None):
    # Load cache
    cache = load_cache()

    # Find CSV files in core/data
    data_glob = os.path.join(os.path.dirname(__file__), '..', 'data', 'posts_*.csv')
    csv_files = glob.glob(data_glob)
    csv_files = sorted(csv_files)
    if limit_files:
        csv_files = csv_files[:limit_files]

    if not csv_files:
        print(f"No se encontraron archivos en {data_glob}")
        return
    # Determine which dates we need from the CSV files (use created_at column)
    needed_dates = set()
    for file_path in csv_files:
        try:
            tmp = pd.read_csv(file_path, usecols=['created_at'])
            for ca in tmp['created_at'].dropna().unique():
                d, _ = date_hour_from_iso(ca)
                if d:
                    needed_dates.add(d)
        except Exception:
            # if file can't be read quickly, skip - process_file will handle it later
            continue

    # Check cache for all dates
    target_hours = getattr(main, 'target_hours', ["10:00", "14:00", "20:00"])

    # Check cache for all dates: ensure the three target-hour keys exist
    missing_dates = [d for d in needed_dates if any(f"{d}|{th}" not in cache for th in target_hours)]
    
    if not missing_dates:
        print(f"Todas las fechas necesarias están en cache ({len(needed_dates)} fechas). No se iniciará el navegador.")
        if main.save_html:
            print("Nota: --save-html fue solicitado pero no se realizará porque se usarán datos desde cache.")
        driver = None
        # process files without starting Selenium (process_file reads from cache)
        for file_path in csv_files:
            process_file(file_path, driver, cache, overwrite=overwrite, max_delta_minutes=main.max_delta_minutes, batch_size=main.max_batch_size, save_html=False, page_delay=main.page_delay)
    else:
        print(f"Faltan {len(missing_dates)} fechas en cache. Se iniciará el navegador para obtener: {missing_dates[:5]}{'...' if len(missing_dates)>5 else ''}")
        driver = setup_driver()
        try:
            for file_path in csv_files:
                process_file(file_path, driver, cache, overwrite=overwrite, max_delta_minutes=main.max_delta_minutes, batch_size=main.max_batch_size, save_html=main.save_html, page_delay=main.page_delay)
        finally:
            try:
                driver.quit()
            except Exception:
                pass
    # save scrape stats
    save_stats()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Agregar similitud con tendencias a CSVs de posts')
    parser.add_argument('--overwrite', action='store_true', help='Sobrescribir archivos de salida si existen', default=True)
    parser.add_argument('--limit', type=int, default=None, help='Procesar solo N archivos para pruebas')
    parser.add_argument('--tolerance', '--max-delta', dest='tolerance', type=int, default=3600,
                        help='Tolerancia en minutos para buscar el bloque de hora más cercano (default 60)')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=200,
                        help='Cantidad de filas entre guardados intermedios (default 200)')
    parser.add_argument('--save-html', dest='save_html', action='store_true', default=True,
                        help='Guardar el HTML de la página por fecha en la carpeta de trends (útil para debugging)')
    parser.add_argument('--page-delay', dest='page_delay', type=float, default=1.0,
                        help='Segundos a esperar después de cargar cada página (default 1.0)')
    parser.add_argument('--match-mode', dest='match_mode', type=str, default='substring',
                        choices=['token','substring','fuzzy'],
                        help='Modo de matching para la similitud binaria: token, substring o fuzzy (default substring)')
    parser.add_argument('--fuzzy-threshold', dest='fuzzy_threshold', type=int, default=60,
                        help='Umbral para matching fuzzy (0-100)')
    parser.add_argument('--min-token-len', dest='min_token_len', type=int, default=3,
                        help='Longitud mínima de token para matching (default 3)')
    args = parser.parse_args()

    # Attach tolerance to main function for easier passing into inner loop
    main.max_delta_minutes = args.tolerance
    main.max_batch_size = args.batch_size
    main.save_html = args.save_html
    main.page_delay = args.page_delay
    main.match_mode = args.match_mode
    main.fuzzy_threshold = args.fuzzy_threshold
    main.min_token_len = args.min_token_len
    main(overwrite=args.overwrite, limit_files=args.limit)
