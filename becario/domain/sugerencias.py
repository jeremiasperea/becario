"""Qué le falta a un material, mirando lo que ya se corrió.

La diferencia con `reglas_fisicas`: allá se mira UNA combinación de
parámetros antes de enviarla; acá se mira la SECUENCIA de corridas de un
material y se dice qué paso falta. Un cálculo puede estar perfecto por
dentro y ser el segundo de una serie cuyo primero nunca se hizo.

Tres decisiones que conviene explicar, porque las tres se podrían haber
tomado al revés:

**No hay LLM acá.** El modelo decide que el usuario está pidiendo una
sugerencia; QUÉ sugerir lo decide este módulo, con reglas. Un LLM local de
7B opinando sobre metodología DFT es exactamente la clase de respuesta
creíble y equivocada que el resto del proyecto se ocupa de evitar — y que
además no se puede testear.

**Solo se sugiere lo que el bot sabe hacer.** `CalcKind` tiene cuatro
valores (relajación, estático, convergencia de ENCUT, DOS) y las
sugerencias se limitan a esos. Recomendar un barrido de k-points sería un
consejo correcto y una promesa que el bot no puede cumplir; si algún día
`CalcKind` crece, esto crece con él.

**Se cita solo lo verificable.** El manual no está en el repo (solo el
vocabulario que `build_vasp_tag_vocabulary.py` genera de él), así que la
sección se busca en ese vocabulario y viaja con el tag del que habla la
regla. Las reglas que expresan una práctica y no la definición de un tag
NO llevan §: inventar una cita para que suene más autoritaria sería
justamente lo contrario de lo que la cita promete.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .models import CalcKind
from .vasp_tags import cita

# Estados del historial que significan «esta corrida no sirve como base».
# `enviado` no está: un trabajo en vuelo todavía puede terminar bien.
_MALOS = {"falló", "cancelado", "expiró", "sin rastro"}


@dataclass(frozen=True)
class CorridaPrevia:
    """Una corrida ya enviada, en los términos que las reglas necesitan.

    La arma la capa de aplicación a partir de la huella que
    `CalcRunRepository` guarda y del estado que dice el historial: el
    dominio no sabe de SQLite ni de JSON.
    """

    formula: str
    calc_kind: CalcKind
    fecha: str
    estado: str = ""
    encut: Optional[int] = None
    job_name: str = ""

    @property
    def sirve_de_base(self) -> bool:
        return self.estado not in _MALOS


@dataclass(frozen=True)
class Sugerencia:
    texto: str
    # Cómo pedírselo al bot, en las palabras que el router entiende. None
    # cuando no hay nada que preparar y la sugerencia es informativa.
    pedido: Optional[str] = None


def _de(corridas: list[CorridaPrevia], kind: CalcKind) -> list[CorridaPrevia]:
    return [c for c in corridas if c.calc_kind is kind]


def sugerencias(corridas: list[CorridaPrevia]) -> list[Sugerencia]:
    """Qué le falta a este material. Lista vacía = no tengo nada que decir.

    `corridas` son las de UN material, más recientes primero. El orden
    importa para el mensaje, no para las reglas.
    """
    if not corridas:
        return []

    formula = corridas[0].formula
    fuera: list[Sugerencia] = []

    # 1. Lo primero es el estado, porque cambia el valor de todo lo demás:
    #    sugerir el siguiente paso sobre una base que no existe es peor que
    #    no sugerir nada.
    ultima = corridas[0]
    if not ultima.sirve_de_base:
        fuera.append(Sugerencia(
            f"⚠️ Tu corrida más reciente de {formula} ({ultima.job_name}) "
            f"quedó en «{ultima.estado}», así que no sirve como punto de "
            "partida. Mirá el diagnóstico que te mandé cuando terminó antes "
            "de encadenar nada encima."
        ))

    utiles = [c for c in corridas if c.sirve_de_base]

    # 2. ENCUT sin barrido. La regla habla del tag, así que la cita sale
    #    del vocabulario del manual y no de acá.
    con_encut = [c for c in utiles if c.encut and c.calc_kind is not CalcKind.ENCUT_SCAN]
    if con_encut and not _de(corridas, CalcKind.ENCUT_SCAN):
        valores = sorted({c.encut for c in con_encut})
        cuales = ", ".join(str(v) for v in valores)
        fuera.append(Sugerencia(
            f"📐 Corriste {formula} con ENCUT={cuales} pero no veo un barrido "
            f"de convergencia{cita('ENCUT')}. El corte fija el tamaño de la "
            "base de ondas planas: energías obtenidas con cortes distintos no "
            "son comparables, y sin verificar que el resultado deje de moverse "
            "al subirlo no se sabe si el número que salió es el del material o "
            "el de la base.",
            pedido=f"hacé la curva de convergencia de ENCUT para {formula}",
        ))

    # 3. Relajación sin estático posterior. Es práctica, no definición de un
    #    tag: va sin §.
    relajaciones = [c for c in utiles if c.calc_kind is CalcKind.RELAX]
    if relajaciones and not _de(utiles, CalcKind.STATIC):
        fuera.append(Sugerencia(
            f"🎯 Relajaste {formula} y no veo un estático sobre la geometría "
            "relajada. Al relajar la celda, la base de ondas planas queda "
            "definida para la celda vieja y arrastra un error que no se ve en "
            "las fuerzas; lo habitual es cerrar con un estático sobre el "
            "resultado, que además da la energía final bien convergida.",
            pedido=f"hacé un estático del {formula} relajado",
        ))

    # 4. DOS sin relajación previa.
    if _de(utiles, CalcKind.DOS) and not relajaciones:
        fuera.append(Sugerencia(
            f"🔬 Pediste la DOS de {formula} sin una relajación previa. La "
            "densidad de estados que sale describe la geometría que le diste, "
            "no la de equilibrio: si la estructura no estaba relajada, el "
            "resultado es de un cristal que no es el que querés.",
            pedido=f"relajá el {formula}",
        ))

    return fuera


def render(formula: str, corridas: list[CorridaPrevia]) -> str:
    """El texto que ve el usuario. Vive acá y no en el handler porque
    decidir qué se dice y en qué orden es parte de la regla, no del canal."""
    if not corridas:
        return (
            f"🔍 No tengo ninguna corrida tuya de {formula}, así que no puedo "
            "sugerirte el siguiente paso. Pedime una relajación y arrancamos "
            "por ahí."
        )

    fuera = sugerencias(corridas)
    hechas = ", ".join(sorted({c.calc_kind.value for c in corridas}))
    encabezado = (
        f"🧭 Sobre tu {formula} — {len(corridas)} corrida(s): {hechas}.\n"
    )
    if not fuera:
        return encabezado + (
            "\nNo veo nada pendiente de lo que sé preparar: tenés el barrido "
            "de ENCUT, la relajación y el estático. El siguiente paso ya es "
            "decisión tuya."
        )

    partes = [encabezado]
    for s in fuera:
        partes.append(f"\n{s.texto}")
    pedidos = [s.pedido for s in fuera if s.pedido]
    if pedidos:
        partes.append(
            "\n\n¿Arranco por alguno? Pedímelo así:\n"
            + "\n".join(f"  • «{p}»" for p in pedidos)
        )
    return "".join(partes)
