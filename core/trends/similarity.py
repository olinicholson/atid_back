from rapidfuzz import fuzz
import re
from core.trends.trendhist import setup_driver, scrape_historical_trends


def evaluate_post_relation_fuzzy(post_text, date_str, hour_str, driver):
    """Legacy fuzzy scorer (kept for compatibility). Returns (best_trend, score).
    """
    trends_data = scrape_historical_trends(driver, date_str, hour_str)
    trend_names = [t["nombre"] for t in trends_data]

    if not trend_names:
        print("No se encontraron tendencias para esa fecha/hora.")
        return None, 0.0

    def normalize_text(text):
        text = text.lower()
        text = re.sub(r"#", "", text)
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    post_norm = normalize_text(post_text)
    trends_norm = [normalize_text(t) for t in trend_names]

    # Calcular scores usando partial_ratio
    scores = [fuzz.partial_ratio(post_norm, t) for t in trends_norm]

    # Imprimir todas las tendencias y su score
    print(f"Tendencias para {date_str} {hour_str} y sus scores vs post:")
    for i, (t, s) in enumerate(zip(trend_names, scores), 1):
        print(f"{i}. {t} -> {s:.1f}")

    max_score = max(scores)
    best_trend = trend_names[scores.index(max_score)]

    print(f"\nPost: '{post_text[:30]}...' | Tendencia más relacionada: {best_trend} (similitud {max_score:.1f})")
    return best_trend, max_score


def evaluate_post_relation_binary(post_text, date_str, hour_str, driver):
    """Binary similarity: returns (matching_trend_or_None, 1) if any token from the post
    appears in any trend name for the given date/hour, otherwise (None, 0).

    Matching is performed on normalized alpha-numeric tokens (lowercased) and ignores
    trivial 1-2 character tokens.
    """
    trends_data = scrape_historical_trends(driver, date_str, hour_str)
    trend_names = [t.get("nombre", "") for t in trends_data]

    if not trend_names:
        return None, 0

    # Default simple token-based matching (kept for compatibility). If you need
    # more permissive matching (substring or fuzzy), use match_post_trend below.
    def tokens(text, min_len=2):
        text = str(text).lower()
        text = re.sub(r"[^a-z0-9#\s]", " ", text)
        toks = re.findall(r"#?[a-z0-9]{%d,}" % (min_len,), text)
        return set(toks)

    post_tokens = tokens(post_text, min_len=2)
    if not post_tokens:
        return None, 0

    for tname in trend_names:
        trend_tokens = tokens(tname, min_len=2)
        if post_tokens & trend_tokens:
            return tname, 1

    return None, 0


def match_post_trend(post_text, trend_text, mode='token', fuzzy_threshold=60, min_token_len=3):
    """Generic permissive matcher. Modes:
      - 'token': token overlap (default)
      - 'substring': checks substring containment of tokens
      - 'fuzzy': uses rapidfuzz.partial_ratio (or difflib) with threshold
    Returns True if considered a match.
    """
    def normalize(s):
        s = str(s).lower()
        s = re.sub(r"[^a-z0-9#\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    post = normalize(post_text)
    trend = normalize(trend_text)

    # tokens (skip very short tokens)
    post_toks = set(re.findall(r"#?[a-z0-9]{%d,}" % (min_token_len,), post))
    trend_toks = set(re.findall(r"#?[a-z0-9]{%d,}" % (min_token_len,), trend))

    if mode == 'token':
        return bool(post_toks & trend_toks)

    if mode == 'substring':
        # check tokens containment in either direction
        for t in trend_toks:
            if t in post:
                return True
        for p in post_toks:
            if p in trend:
                return True
        # fallback to whole-string containment
        if trend and (trend in post or post in trend):
            return True
        return False

    if mode == 'fuzzy':
        # use rapidfuzz if available
        try:
            from rapidfuzz import fuzz as _rf_fuzz
            score = _rf_fuzz.partial_ratio(post, trend)
            return score >= fuzzy_threshold
        except Exception:
            # fallback to difflib
            import difflib
            r = difflib.SequenceMatcher(None, post, trend).ratio()
            return (r * 100.0) >= fuzzy_threshold

    # unknown mode -> no match
    return False

if __name__ == "__main__":
    driver = setup_driver()
    post_text = "Me encantó el reality de Telefe esta noche"
    date_posted = "06-08-2024"
    hour_posted = "00:30"

    best_trend, score = evaluate_post_relation_fuzzy(post_text, date_posted, hour_posted, driver)
    driver.quit()
    print("Resultado fuzzy:", best_trend, score)

"""from rapidfuzz import fuzz
import re
from trendhist import setup_driver, scrape_historical_trends

def evaluate_post_relation_fuzzy(post_text, date_str, hour_str, driver):
    # 1️⃣ Obtener tendencias de esa fecha y hora
    trends_data = scrape_historical_trends(driver, date_str, hour_str)
    trend_names = [t["nombre"] for t in trends_data]

    if not trend_names:
        print("No se encontraron tendencias para esa fecha/hora.")
        return None, 0.0

    # 2️⃣ Normalizar texto
    def normalize_text(text):
        text = text.lower()
        text = re.sub(r"#", "", text)
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    post_norm = normalize_text(post_text)
    trends_norm = [normalize_text(t) for t in trend_names]

    # 3️⃣ Comparar post vs cada tendencia
    scores = [fuzz.token_set_ratio(post_norm, t) for t in trends_norm]

    max_score = max(scores)
    best_trend = trend_names[scores.index(max_score)]

    print(f"Tendencias para {date_str} {hour_str}:")
    for i, t in enumerate(trend_names, 1):
        print(f"{i}. {t}")

    print(f"\nPost: '{post_text[:30]}...' | Tendencia más relacionada: {best_trend} (similitud {max_score:.1f})")
    return best_trend, max_score

if __name__ == "__main__":
    driver = setup_driver()
    post_text = "Me encantó el reality de Telefe esta noche"
    date_posted = "06-08-2024"
    hour_posted = "00:30"

    best_trend, score = evaluate_post_relation_fuzzy(post_text, date_posted, hour_posted, driver)
    driver.quit()
    print("Resultado fuzzy:", best_trend, score)
"""