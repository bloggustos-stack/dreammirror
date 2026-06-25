"""
DreamMirror - Analizator de vise în stil jungian
Parte din universul ONYR.WORLD / Archetype Academy
"""
import os
import json
import re
import requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

MODELS_TO_TRY = ["gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-flash-latest"]

ANALYSIS_SCHEMA_HINT = """
Răspunde STRICT cu un obiect JSON valid, fără text adițional înainte sau după, cu exact această structură:
{
  "titlu": "string - titlu simbolic și poetic pentru acest vis, în aceeași limbă ca visul",
  "rezumat": "string - 1-2 propoziții, rezumat al firelor narative cheie ale visului",
  "arhetipuri": [
    {"nume": "string - numele arhetipului", "descriere": "string - semnificația arhetipală generală", "rolInVis": "string - cum se manifestă în acțiunea/decorul visului"}
  ],
  "simboluri": [
    {"simbol": "string - elementul simbolic din vis", "semnificatie": "string - sensul literal sau senzația brută", "asocieriJungiene": "string - amplificare simbolică (culturală, mitologică)"}
  ],
  "tensiuniConstientInconstient": "string - tensiunea de bază detectată între atitudinea ego-ului și compensarea inconștientului",
  "intrebariSocratice": ["string", "string", "string"],
  "scenariuImaginatieActiva": "string - un ghid/scenariu de Imaginație Activă bazat pe vis",
  "etapaAlchimica": "string - EXACT una din: Nigredo, Albedo, Citrinitas, Rubedo"
}
"""

SYSTEM_INSTRUCTION_TEMPLATE = """Ești un analist jungian profesionist (psihoterapeut de orientare analitică), expert în psihologia viselor a lui Carl Gustav Jung.
Rolul tău este să analizezi visul oferit în mod structurat, utilizând termenii originali jungieni și oferind o perspectivă alchimică și arhetipală de o profundă valoare introspectivă. Nu folosi clișee facile de "dicționar de vise". Caută compensarea pe care inconștientul o oferă atitudinii conștiente a visătorului.
Starea emoțională raportată de visător la trezire este: "{emotion}".

IMPORTANT: Detectează limba în care este scris visul trimis de utilizator (de exemplu: română, engleză, spaniolă, franceză, germană etc.).
Toate explicațiile, titlurile, descrierile, întrebările și asocierile pe care le generezi în JSON-ul de răspuns TREBUIE să fie în aceeași limbă în care este redactat visul. Dacă limba visului este neclară sau nespecificată, folosește limba engleză ca standard.

Etapa Alchimică (etapaAlchimica) - alege EXACT una dintre acestea patru:
- "Nigredo" (Stadiul umbrei, melancolie, dezintegrare, dezordine inițială necesară).
- "Albedo" (Purificare, reflectare, apariția figurii Anima/Animus, claritatea apei).
- "Citrinitas" (Trezire, lumina solară, înțelepciunea Bătrânului Înțelept).
- "Rubedo" (Reîntregire, unificarea contrariilor, conjuncția sacră, realizarea Sinelui).

{schema_hint}"""

CHAT_SYSTEM_INSTRUCTION_TEMPLATE = """Ești un analist jungian de o blândețe și înțelepciune deosebite. Scopul tău nu este să dai sentințe sau interpretări de neatins, ci să ajuți utilizatorul să asocieze personal elementele visului său.
Porți un dialog terapeutic bazat pe visul următor și pe analiza de bază realizată anterior:

VISUL UTILIZATORULUI:
"{dream_text}"

STADIUL ALCHIMIC DETECTAT:
"{stage}"

TENSIUNEA CONȘTIENT-INCONȘTIENT:
"{tension}"

Instrucțiuni de conversație:
1. Răspunde direct și empatic la asocierile sau întrebările pe care le trimite utilizatorul în ultimul mesaj.
2. Răspunde întotdeauna în aceeași limbă în care se adresează utilizatorul sau în care este redactat visul.
3. Evită răspunsurile extrem de lungi. Limitează-te la 2-3 paragrafe scurte și pune o singură întrebare profundă la final, care ghidează utilizatorul spre auto-descoperire.
4. Fii flexibil: dacă utilizatorul vrea să exploreze o anumită figură, o emoție, sau dorește să facă exercițiul de Imaginație Activă explicat, ghidează-l interactiv pas cu pas."""


def extract_json(text):
    """Extrage JSON valid dintr-un răspuns care poate conține markdown fences."""
    text = (text or "").strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Răspunsul modelului nu este JSON valid: {str(e)}. Conținut primit: {text[:300]}")


def call_gemini(model_name, system_instruction, user_prompt, json_mode=False):
    """Apel HTTP direct către API-ul Gemini, fără SDK greu."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_API_KEY}"

    generation_config = {"temperature": 0.75}
    if json_mode:
        generation_config["responseMimeType"] = "application/json"

    payload = {
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "generationConfig": generation_config,
    }

    response = requests.post(url, json=payload, timeout=60)

    if response.status_code != 200:
        # Răspunsul poate fi JSON cu detalii de eroare sau HTML simplu
        try:
            err_data = response.json()
            err_msg = err_data.get("error", {}).get("message", response.text[:200])
        except Exception:
            err_msg = response.text[:200]
        raise RuntimeError(f"Model {model_name} a returnat {response.status_code}: {err_msg}")

    try:
        data = response.json()
    except Exception:
        raise RuntimeError(f"Model {model_name}: răspuns invalid (nu e JSON).")

    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError("Modelul Gemini nu a returnat niciun rezultat.")

    parts = candidates[0].get("content", {}).get("parts", [])
    if not parts:
        raise RuntimeError("Răspunsul Gemini nu conține text.")

    return parts[0].get("text", "")


def generate_with_fallback(prompt, system_instruction, json_mode=False):
    """Încearcă mai multe modele Gemini în caz de eroare tranzitorie."""
    if not GEMINI_API_KEY:
        raise RuntimeError("Cheia GEMINI_API_KEY nu este configurată pe server.")

    last_error = None
    for model_name in MODELS_TO_TRY:
        try:
            return call_gemini(model_name, system_instruction, prompt, json_mode=json_mode)
        except Exception as e:
            last_error = e
            continue
    raise last_error or RuntimeError("Toate modelele Gemini au eșuat.")


@app.route("/")
def index():
    return render_template("dreammirror.html")


@app.route("/api/analyze-dream", methods=["POST"])
def analyze_dream():
    data = request.get_json(silent=True) or {}
    dream_text = (data.get("dreamText") or "").strip()
    emotion = data.get("emotion") or "Necunoscută"

    if not dream_text:
        return jsonify({"error": "Conținutul visului este obligatoriu."}), 400

    system_instruction = SYSTEM_INSTRUCTION_TEMPLATE.format(
        emotion=emotion, schema_hint=ANALYSIS_SCHEMA_HINT
    )
    prompt = f'Iată visul pe care te rog să îl analizezi conform structurii cerute:\n"{dream_text}"'

    try:
        raw_text = generate_with_fallback(prompt, system_instruction, json_mode=True)
        result = extract_json(raw_text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e) or "A apărut o eroare necunoscută pe server."}), 500


@app.route("/api/chat-dream", methods=["POST"])
def chat_dream():
    data = request.get_json(silent=True) or {}
    dream_text = data.get("dreamText")
    analysis = data.get("analysis") or {}
    chat_history = data.get("chatHistory") or []
    latest_message = data.get("latestMessage")

    if not dream_text or not latest_message:
        return jsonify({"error": "Parametrii dreamText și latestMessage sunt obligatorii."}), 400

    system_instruction = CHAT_SYSTEM_INSTRUCTION_TEMPLATE.format(
        dream_text=dream_text,
        stage=analysis.get("etapaAlchimica", "Nigredo"),
        tension=analysis.get("tensiuniConstientInconstient", ""),
    )

    # Construim istoricul conversației ca prompt secvențial
    conversation_parts = []
    for msg in chat_history:
        role_label = "Utilizator" if msg.get("role") == "user" else "Analist"
        conversation_parts.append(f"{role_label}: {msg.get('text', '')}")
    conversation_parts.append(f"Utilizator: {latest_message}")
    full_prompt = "\n\n".join(conversation_parts)

    try:
        reply_text = generate_with_fallback(full_prompt, system_instruction, json_mode=False)
        return jsonify({"text": reply_text or "Nu am putut genera un răspuns. Încearcă din nou."})
    except Exception as e:
        return jsonify({"error": str(e) or "Eroare la procesarea dialogului."}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
