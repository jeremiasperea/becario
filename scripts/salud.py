#!/usr/bin/env python3
"""Chequeo de salud de B.E.C.A.R.I.O.

    .venv/bin/python scripts/salud.py
    .venv/bin/python scripts/salud.py --json        # para un cron o un panel
    .venv/bin/python scripts/salud.py --dias 30     # ventana de las métricas

Existe porque para un servicio que corre desatendido el único síntoma era
alguien diciendo «che, no me contesta» — que es, textualmente, lo que pasó
el 18 de julio y el 2 de agosto de 2026, y que se descubrió semanas después
leyendo la base a mano.

Lo notable es que casi todo lo que hace falta para saber si el bot está
sano YA se venía registrando y no lo miraba nadie: `decisiones_router`
guarda latencia y desenlace de cada ruteo desde hace meses. El p90 de
118.1 s contra un timeout de 120 —el hallazgo que motivó la mitad del
refactor de tolerancia a fallos— salió de una consulta a esa tabla. Este
script es esa consulta, más las tres sondas en vivo, con un umbral que
avisa antes de que el usuario tenga que quejarse.

Código de salida: 0 si está sano, 1 si hay algo que mirar. Pensado para
`cron` o un `systemd` timer:

    */15 * * * * cd /opt/becario && .venv/bin/python scripts/salud.py --json \\
                 >> /var/log/becario-salud.jsonl 2>&1
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import logging
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from becario.config import ConfigError, Settings  # noqa: E402
from becario.infrastructure.ssh_gateway import SSHClusterGatewayFactory  # noqa: E402
from becario.infrastructure.user_registry import JSONUserRegistry  # noqa: E402

# A partir de qué fracción del timeout el p90 deja de ser cómodo. 0.75 no es
# caprichoso: con el timeout en 120 s el p90 real era 118.1 (98%), y para
# cuando eso se nota ya se perdieron pedidos. La idea es avisar mientras
# todavía hay margen para subir el timeout o cambiar de modelo.
_FRACCION_INCOMODA = 0.75


def _callar_el_ruido_de_las_sondas() -> None:
    """Las sondas ya cuentan qué falló y por qué; el detalle de cada
    reintento solo tapa el informe.

    Va DENTRO de `main()` y no al importar el módulo, que fue el primer
    intento: tocar el logging global en el import se lo lleva puesto a
    quien importe esto por otra razón. Lo detectó la propia suite —nueve
    tests que verifican mensajes de log se cayeron de golpe— y es un
    recordatorio barato de que un import no debería cambiarle el estado
    global a nadie.
    """
    logging.getLogger("becario").setLevel(logging.CRITICAL)


def _pct(valores: list[float], p: float) -> float:
    """Percentil por el método del más cercano, que para decenas de
    muestras es tan bueno como interpolar y no inventa valores."""
    if not valores:
        return 0.0
    ordenados = sorted(valores)
    return ordenados[min(len(ordenados) - 1, int(round(p * (len(ordenados) - 1))))]


# ---------------------------------------------------------------------------
# Sondas en vivo
# ---------------------------------------------------------------------------


def _sonda_ollama(s: Settings) -> dict:
    try:
        with urllib.request.urlopen(f"{s.ollama_url}/api/tags", timeout=8) as r:
            nombres = [m["name"] for m in json.load(r)["models"]]
    except Exception as exc:
        return {"ok": False, "detalle": f"no responde en {s.ollama_url}: {exc}"}
    presente = s.ollama_model in nombres or f"{s.ollama_model}:latest" in nombres
    return {
        "ok": presente,
        "detalle": (
            f"{s.ollama_model} presente"
            if presente
            else f"falta {s.ollama_model}; hay {', '.join(nombres[:4]) or 'ninguno'}"
        ),
    }


def _sonda_cluster(s: Settings) -> dict:
    try:
        registry = JSONUserRegistry(s.users_file)
        identidades = list(registry._by_user_id.values())
    except Exception as exc:
        return {"ok": False, "detalle": f"no pude leer el roster: {exc}"}
    if not identidades:
        return {"ok": False, "detalle": "no hay nadie registrado en el roster"}
    gw = SSHClusterGatewayFactory(
        default_host=s.ssh_host, default_port=s.ssh_port
    ).for_identity(identidades[0])
    try:
        # `sinfo` dice de una si Slurm atiende Y si hay nodos: un cluster que
        # responde el SSH pero tiene el controlador caído no sirve para nada.
        r = gw._run("sinfo -h -o '%P %a %D %T' | head -5", reintentable=True)
        if not r.ok:
            return {"ok": False, "detalle": r.message.strip()[:120]}
        lineas = [l for l in r.stdout.strip().splitlines() if l.strip()]
        return {
            "ok": bool(lineas),
            "detalle": " · ".join(lineas[:3]) or "sinfo no devolvió particiones",
        }
    finally:
        gw.close()


# ---------------------------------------------------------------------------
# Lo que la base ya sabía
# ---------------------------------------------------------------------------


def _metricas(db: Path, s: Settings, dias: int) -> dict:
    if not db.exists():
        return {"error": f"no existe {db}"}
    desde = (datetime.now(timezone.utc) - timedelta(days=dias)).isoformat()
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        def tabla(nombre: str) -> bool:
            return conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (nombre,)
            ).fetchone() is not None

        def columna(tabla_: str, nombre: str) -> bool:
            """Las columnas se agregan por migración suave al instanciar su
            repositorio, y este script abre la base en SOLO LECTURA sin
            pasar por ninguno. O sea que puede encontrarse una base más
            vieja que el código — y un chequeo de salud que revienta con
            una base vieja es exactamente lo contrario de lo que promete."""
            return any(
                r[1] == nombre for r in conn.execute(f"PRAGMA table_info({tabla_})")
            )

        m: dict = {"ventana_dias": dias}

        if tabla("decisiones_router"):
            filas = conn.execute(
                "SELECT latency_seconds, outcome FROM decisiones_router "
                "WHERE created_at >= ?", (desde,)
            ).fetchall()
            lat = [f["latency_seconds"] for f in filas if f["latency_seconds"]]
            desenlaces: dict = {}
            for f in filas:
                desenlaces[f["outcome"]] = desenlaces.get(f["outcome"], 0) + 1
            m["router"] = {
                "decisiones": len(filas),
                "p50": round(median(lat), 1) if lat else None,
                "p90": round(_pct(lat, 0.90), 1) if lat else None,
                "max": round(max(lat), 1) if lat else None,
                "timeout": s.ollama_timeout_seconds,
                "desenlaces": desenlaces,
            }

        if tabla("fallos_router"):
            fallos = conn.execute(
                "SELECT reason, COUNT(*) n FROM fallos_router "
                "WHERE created_at >= ? GROUP BY reason", (desde,)
            ).fetchall()
            m["fallos_del_modelo"] = {f["reason"]: f["n"] for f in fallos}

        if tabla("trabajos_monitoreados"):
            trabajos = {
                "en_seguimiento": conn.execute(
                    "SELECT COUNT(*) FROM trabajos_monitoreados WHERE notified = 0"
                ).fetchone()[0],
            }
            if columna("trabajos_monitoreados", "poll_attempts"):
                trabajos["sin_noticias"] = conn.execute(
                    "SELECT COUNT(*) FROM trabajos_monitoreados "
                    "WHERE notified = 0 AND poll_attempts > 0"
                ).fetchone()[0]
            m["trabajos"] = trabajos

        if tabla("chat_messages"):
            ultimo = conn.execute(
                "SELECT MAX(created_at) FROM chat_messages"
            ).fetchone()[0]
            m["ultimo_mensaje"] = ultimo
        return m
    finally:
        conn.close()


# ---------------------------------------------------------------------------


def _diagnostico(datos: dict) -> list[str]:
    """Qué hay para mirar. Lista vacía = está sano."""
    avisos = []
    for nombre in ("ollama", "cluster"):
        if not datos[nombre]["ok"]:
            avisos.append(f"{nombre}: {datos[nombre]['detalle']}")

    router = datos.get("metricas", {}).get("router") or {}
    p90, timeout = router.get("p90"), router.get("timeout")
    if p90 and timeout and p90 > timeout * _FRACCION_INCOMODA:
        avisos.append(
            f"el p90 del router ({p90} s) va por el "
            f"{p90 / timeout:.0%} del timeout ({timeout:g} s): "
            "los pedidos lentos están por empezar a expirar"
        )

    fallos = datos.get("metricas", {}).get("fallos_del_modelo") or {}
    if fallos:
        detalle = ", ".join(f"{n}× {r}" for r, n in fallos.items())
        avisos.append(f"el modelo no contestó {sum(fallos.values())} vez/veces ({detalle})")

    perdidos = (datos.get("metricas", {}).get("trabajos") or {}).get("sin_noticias") or 0
    if perdidos:
        avisos.append(f"{perdidos} trabajo(s) en seguimiento sin poder leer su estado")
    return avisos


def _imprimir(datos: dict, avisos: list[str]) -> None:
    marca = lambda ok: "✅" if ok else "❌"  # noqa: E731
    print(f"B.E.C.A.R.I.O. · salud al {datos['momento'][:19].replace('T', ' ')}\n")
    print("Dependencias")
    print(f"  {marca(datos['ollama']['ok'])} Ollama    {datos['ollama']['detalle']}")
    print(f"  {marca(datos['cluster']['ok'])} Cluster   {datos['cluster']['detalle']}")

    m = datos.get("metricas", {})
    if "error" in m:
        print(f"\n⚠️  Sin métricas: {m['error']}")
    else:
        r = m.get("router")
        if r and r["decisiones"]:
            print(f"\nRouter · {r['decisiones']} decisiones en {m['ventana_dias']} días")
            print(f"  latencia    p50 {r['p50']} s · p90 {r['p90']} s · máx {r['max']} s"
                  f"   (timeout {r['timeout']:g} s)")
            print("  desenlaces  " + " · ".join(f"{n} {o}" for o, n in r["desenlaces"].items()))
        fallos = m.get("fallos_del_modelo")
        if fallos:
            print("  el modelo no contestó  " + ", ".join(f"{n}× {r_}" for r_, n in fallos.items()))
        t = m.get("trabajos")
        if t:
            linea = f"\nTrabajos    {t['en_seguimiento']} en seguimiento"
            if "sin_noticias" in t:
                linea += f" · {t['sin_noticias']} sin noticias"
            print(linea)
        if m.get("ultimo_mensaje"):
            print(f"Último mensaje  {m['ultimo_mensaje'][:19].replace('T', ' ')}")

    print()
    if avisos:
        print("Para mirar:")
        for a in avisos:
            print(f"  ⚠️  {a}")
    else:
        print("✅ Nada para reportar.")


def main() -> int:
    p = argparse.ArgumentParser(description="Chequeo de salud de B.E.C.A.R.I.O.")
    p.add_argument("--json", action="store_true", help="salida JSON, para cron o panel")
    p.add_argument("--dias", type=int, default=7, help="ventana de las métricas (7)")
    args = p.parse_args()
    _callar_el_ruido_de_las_sondas()

    try:
        s = Settings.from_env()
    except ConfigError as exc:
        print(f"❌ Configuración incompleta: {exc}", file=sys.stderr)
        return 1

    datos = {
        "momento": datetime.now(timezone.utc).isoformat(),
        "ollama": _sonda_ollama(s),
        "cluster": _sonda_cluster(s),
        "metricas": _metricas(Path(s.db_path), s, args.dias),
    }
    avisos = _diagnostico(datos)
    datos["sano"] = not avisos
    datos["avisos"] = avisos

    if args.json:
        print(json.dumps(datos, ensure_ascii=False))
    else:
        _imprimir(datos, avisos)
    return 0 if datos["sano"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
