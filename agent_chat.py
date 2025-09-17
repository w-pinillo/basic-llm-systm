import re
import requests
import json
import time

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen:4b"

# --- Herramienta clima ---
def get_weather(city):
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=es&format=json"
    geo_data = requests.get(geo_url).json()

    if "results" not in geo_data:
        return f"No encontré la ciudad {city}."

    lat = geo_data["results"][0]["latitude"]
    lon = geo_data["results"][0]["longitude"]

    weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&timezone=auto"
    weather_data = requests.get(weather_url).json()

    if "current_weather" not in weather_data:
        return "No pude obtener el clima en este momento."

    current = weather_data["current_weather"]
    temp = current["temperature"]
    wind = current["windspeed"]

    return f"El clima en {city} es {temp}°C con viento de {wind} km/h."


# --- Herramienta búsqueda web (simple) ---
def web_search(query):
    url = "https://duckduckgo.com/?q=" + query.replace(" ", "+")
    return f"Puedes ver resultados en: {url}"


# --- Llamada a Qwen ---
def ask_qwen(prompt, model=MODEL, timeout=30):
    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("response", "")
    except requests.exceptions.RequestException as e:
        return f"[Error conectando con Ollama: {e}]"


# --- Prompt con instrucciones JSON ---
TOOL_PROMPT = """
Eres un asistente que SOLO puede responder en JSON válido.

Herramientas disponibles:
1. get_weather(city) → da el clima actual de una ciudad.
2. web_search(query) → busca información en internet.
3. answer(text) → responde directamente al usuario.

Reglas:
- NO expliques nada fuera del JSON.
- NO uses texto adicional.
- Tu salida debe ser SOLO un JSON válido.

Ejemplos válidos:
{{"action": "get_weather", "args": {{"city": "Madrid"}}}}
{{"action": "web_search", "args": {{"query": "últimas noticias de IA"}}}}
{{"action": "answer", "args": {{"text": "Hola, ¿cómo estás?"}}}}

Usuario: {user_input}
Agente:
"""

def safe_json_parse(output: str):
    """
    Intenta extraer y parsear JSON de la salida del modelo.
    Si falla, lo trata como una respuesta normal (action=answer).
    """
    output = output.strip()

    # Caso 1: la salida completa es JSON válido
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        pass

    # Caso 2: buscar el primer bloque {...}
    import re
    match = re.search(r"\{.*\}", output, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Caso 3: fallback → tratamos todo como respuesta normal
    return {"action": "answer", "args": {"text": output}}

# --- Ejecutar acción según JSON ---
def execute_action(model_output):
    action_obj = safe_json_parse(model_output)

    action = action_obj.get("action")
    args = action_obj.get("args", {})

    if action == "get_weather":
        return get_weather(args.get("city", ""))
    elif action == "web_search":
        return web_search(args.get("query", ""))
    elif action == "answer":
        return args.get("text", "")
    else:
        return f"[Error: acción desconocida {action}]"

# --- Bucle principal ---
def main():
    print("Agente con herramientas (Qwen + JSON). Escribe 'salir' para terminar.")
    while True:
        try:
            user_input = input("Tú: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSaliendo...")
            break

        if not user_input:
            continue
        if user_input.lower() in {"salir", "exit", "quit"}:
            print("Adiós 👋")
            break

        # Construir prompt con la instrucción de JSON
        prompt = TOOL_PROMPT.format(user_input=user_input)

        # Pedir al modelo
        model_output = ask_qwen(prompt)

        # Ejecutar acción
        respuesta = execute_action(model_output)

        print("Agente:", respuesta)


if __name__ == "__main__":
    main()

