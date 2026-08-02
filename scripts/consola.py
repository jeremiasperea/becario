"""Consola local: le habla al servicio sin pasar por Telegram.

`BecarioService` no sabe nada de Telegram —recibe (chat_id, user_id, texto)
y devuelve un `Reply`—, así que se lo puede manejar desde la terminal. Sirve
para probar el ruteo y los handlers sin el ida y vuelta del chat.

NO reemplaza la prueba en Telegram: los botones de confirmación (✅ / ❌ / ✏️)
viajan por callbacks de Telegram y acá no existen. Todo lo que dependa de
apretar un botón hay que probarlo en el chat.

Uso:  .venv/bin/python scripts/consola.py [user_id]

Se corre desde la raíz del repo (necesita `main.py` y el `.env`).
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from becario.config import Settings  # noqa: E402

from main import build_bot  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

if len(sys.argv) > 1:
    USER_ID = int(sys.argv[1])
else:
    # Sin argumento se usa el primer usuario del roster: hardcodear un id
    # dejaría el script sirviendo solo para quien lo escribió.
    import json

    roster = json.loads(Path("users.json").read_text(encoding="utf-8"))
    entries = roster["users"] if isinstance(roster, dict) else roster
    USER_ID = int(entries[0]["telegram_user_id"])
CHAT_ID = 1

settings = Settings.from_env()
service = build_bot(settings)._service
print(f"modelo: {settings.ollama_model}  ·  usuario: {USER_ID}")
print("Escribí un pedido (Ctrl-D o 'salir' para terminar).\n")

while True:
    try:
        text = input("vos> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break
    if not text or text in ("salir", "exit", "quit"):
        break
    reply = service.handle_text(chat_id=CHAT_ID, user_id=USER_ID, text=text)
    flags = []
    if reply.awaiting_params:
        flags.append("ESPERANDO UN DATO")
    if getattr(reply, "needs_confirmation", False):
        flags.append("PIDE CONFIRMACIÓN (botón: probalo en Telegram)")
    if not reply.ok:
        flags.append("ok=False")
    print(f"\nbot> {reply.text}")
    if flags:
        print(f"     [{' · '.join(flags)}]")
    print()
