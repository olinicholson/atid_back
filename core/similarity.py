from rapidfuzz import fuzz
import re
from trendhist import setup_driver, scrape_historical_trends

def evaluate_post_relation_fuzzy(post_text, date_str, hour_str, driver):
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