# agent_chat.py
import requests
import json
import time

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen:4b"

def ask_qwen(prompt, model=MODEL, stream=False, timeout=30):
    """
    Llama a Ollama. Si stream=True imprime tokens según llegan.
    Devuelve la respuesta completa como string (o un mensaje de error).
    """
    payload = {"model": model, "prompt": prompt, "stream": stream}
    try:
        if stream:
            with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=timeout) as resp:
                resp.raise_for_status()
                collected = ""
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line.decode("utf-8"))
                        token = chunk.get("response", "")
                        collected += token
                        print(token, end="", flush=True)
                    except Exception:
                        pass
                print()
                return collected
        else:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")
    except requests.exceptions.RequestException as e:
        return f"[Error conectando con Ollama: {e}]"


def build_prompt(history, system_prompt=None):
    """
    Convierte history = list of (role, text) a un prompt legible.
    role: "user" o "agent"
    Añade 'Agente:' al final para que el modelo genere la respuesta.
    """
    parts = []
    if system_prompt:
        parts.append(system_prompt.strip())
        parts.append("")
    for role, text in history:
        if role == "user":
            parts.append(f"Usuario: {text}")
        else:
            parts.append(f"Agente: {text}")
    parts.append("Agente:")
    return "\n".join(parts)


def main():
    system_prompt = (
        "Eres un asistente conversacional en español. Responde de forma clara, breve y útil."
    )
    history = []  # [(role, text), ...]
    print("Chat con Qwen (qwen:4b). Escribe 'salir' para terminar.")
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

        # añadir entrada del usuario al historial
        history.append(("user", user_input))

        # limitar historial a N turnos para no exceder contexto
        max_turn_pairs = 6  # número de pares usuario/agente a conservar
        # cada turno es usuario+agente ≈ 2 entradas; tomamos last 2*max_turn_pairs
        short_history = history[-(max_turn_pairs * 2):]

        # construir prompt
        prompt = build_prompt(short_history, system_prompt)

        # pedir respuesta al modelo (stream=False por defecto)
        respuesta = ask_qwen(prompt, stream=False)

        # guardar respuesta en el historial y mostrar al usuario
        history.append(("agent", respuesta))
        print("Agente:", respuesta)
        # pequeño retardo opcional para evitar spam al servidor
        time.sleep(0.05)


if __name__ == "__main__":
    main()
