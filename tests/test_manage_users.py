"""Tests de `scripts/manage_users.py`, el único escritor de `users.json`.

Ese archivo es el control de acceso del bot: quien figura ahí puede operar
el cluster con su cuenta SSH y quien no figura no existe para B.E.C.A.R.I.O.
El lector (`JSONUserRegistry`) ya estaba cubierto; el escritor no tenía nada,
así que un cambio en el formato que escribe el CLI podía dejar el roster
ilegible sin que ningún test se enterara.

El test que más importa es el round-trip: lo que escribe `cmd_add` lo tiene
que poder leer `JSONUserRegistry` y devolver la `ClusterIdentity` correcta.
Si eso se rompe, el bot no reconoce a nadie — o peor, reconoce a alguien con
los datos de otro. Lo demás cubre las formas conocidas de perder gente:
pisar a los que ya estaban, borrar de más, o escribir una entrada inválida.

Todo corre contra `tmp_path`: ningún test toca el `users.json` real ni la
red ni el cluster.
"""
import argparse
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from becario.infrastructure.user_registry import JSONUserRegistry
from scripts.manage_users import (
    _load,
    _prompt,
    _prompt_int,
    _save,
    cmd_add,
    cmd_list,
    cmd_remove,
    main,
)


def _add_args(path: Path, **overrides) -> argparse.Namespace:
    """Los flags de `add` ya parseados, con datos válidos por default.

    Se arma a mano en vez de pasar por `argparse` para que cada test declare
    solo lo que le importa; el camino que sí pasa por el parser lo cubre
    `TestMainDesdeLaLineaDeComandos`.
    """
    campos = {
        "file": str(path),
        "telegram_id": 111,
        "ssh_user": "alice",
        "ssh_key": "/home/becario/.ssh/id_alice",
        "name": "Alice",
        "ssh_host": None,
    }
    campos.update(overrides)
    return argparse.Namespace(**campos)


def _roster(path: Path, *entradas: dict) -> None:
    """Escribe un roster preexistente, como si lo hubiera dejado otra corrida."""
    path.write_text(
        json.dumps({"users": list(entradas)}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _entrada(telegram_user_id: int, ssh_user: str, **extra) -> dict:
    entrada = {
        "telegram_user_id": telegram_user_id,
        "ssh_user": ssh_user,
        "ssh_key_path": f"/home/becario/.ssh/id_{ssh_user}",
        "display_name": ssh_user.capitalize(),
    }
    entrada.update(extra)
    return entrada


def _ids(path: Path) -> list[int]:
    return [u["telegram_user_id"] for u in json.loads(path.read_text())["users"]]


def _teclado(monkeypatch, *respuestas: str) -> None:
    """Simula a alguien tecleando `respuestas`, una por cada `input()`."""
    pendientes = iter(respuestas)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(pendientes))


# ---------------------------------------------------------------------------
# Round-trip escritor -> lector
# ---------------------------------------------------------------------------


class TestRoundTripConElRegistro:
    """Lo que escribe el CLI lo tiene que leer el registro del bot.

    Son dos módulos que nunca se importan entre sí: el CLI serializa con
    `model_dump_json` y el registro deserializa con `ClusterIdentity(**entry)`.
    Nadie miraba esa juntura. Si el formato se corriera de un lado, el
    registro descartaría la entrada como inválida —silenciosamente, porque
    solo loguea un error— y la persona quedaría afuera del bot sin que
    ningún test fallara.
    """

    def test_lo_que_escribe_add_lo_lee_el_registro_como_identidad(self, tmp_path):
        path = tmp_path / "users.json"
        cmd_add(_add_args(path, telegram_id=111, ssh_user="alice", name="Alice"))

        identidad = JSONUserRegistry(str(path)).get_identity(111)

        assert identidad is not None
        assert identidad.telegram_user_id == 111
        assert identidad.ssh_user == "alice"
        assert identidad.ssh_key_path == "/home/becario/.ssh/id_alice"
        assert identidad.display_name == "Alice"

    def test_el_host_propio_sobrevive_el_round_trip(self, tmp_path):
        """`ssh_host` es lo que decide a qué máquina se conecta esa persona.

        Si se perdiera al escribir, el bot la mandaría al host global y la
        sesión SSH fallaría —o, peor, entraría al cluster equivocado.
        """
        path = tmp_path / "users.json"
        cmd_add(_add_args(path, ssh_host="otro.cluster.edu.ar"))

        identidad = JSONUserRegistry(str(path)).get_identity(111)

        assert identidad.ssh_host == "otro.cluster.edu.ar"

    def test_sin_host_propio_la_clave_no_se_escribe_y_el_registro_lee_none(
        self, tmp_path
    ):
        """`exclude_none=True` deja `ssh_host` afuera del JSON.

        La entrada escrita tiene que seguir siendo válida para el registro
        aunque le falte el campo: `None` significa «usá el host global», y
        escribir un `null` explícito sería ruido innecesario en el archivo.
        """
        path = tmp_path / "users.json"
        cmd_add(_add_args(path, ssh_host=None))

        escrito = json.loads(path.read_text())["users"][0]
        assert "ssh_host" not in escrito
        assert JSONUserRegistry(str(path)).get_identity(111).ssh_host is None

    def test_el_nombre_con_acentos_viaja_legible_y_vuelve_igual(self, tmp_path):
        """El roster lo edita gente a mano: `ensure_ascii=False` importa.

        Con escapes `\\uXXXX` el round-trip igual funcionaría, pero el archivo
        dejaría de ser revisable por un humano, que es la razón de que la
        administración viva afuera de Telegram.
        """
        path = tmp_path / "users.json"
        cmd_add(_add_args(path, name="Juan Pérez", ssh_user="jperez"))

        assert "Juan Pérez" in path.read_text(encoding="utf-8")
        assert JSONUserRegistry(str(path)).get_identity(111).display_name == "Juan Pérez"

    def test_varias_altas_seguidas_quedan_todas_visibles_para_el_registro(
        self, tmp_path
    ):
        path = tmp_path / "users.json"
        cmd_add(_add_args(path, telegram_id=111, ssh_user="alice", name="Alice"))
        cmd_add(_add_args(path, telegram_id=222, ssh_user="bob", name="Bob"))
        cmd_add(_add_args(path, telegram_id=333, ssh_user="carol", name="Carol"))

        registro = JSONUserRegistry(str(path))

        assert registro.get_identity(111).ssh_user == "alice"
        assert registro.get_identity(222).ssh_user == "bob"
        assert registro.get_identity(333).ssh_user == "carol"
        assert registro.get_identity(444) is None


# ---------------------------------------------------------------------------
# Altas sobre un roster que ya tiene gente
# ---------------------------------------------------------------------------


class TestAgregarNoPisaAlRestoDelRoster:
    """Cada alta reescribe el archivo entero, así que el riesgo real no es
    que la entrada nueva falte: es que se lleve puesto al resto del grupo."""

    def test_dar_de_alta_a_alguien_conserva_a_los_que_ya_estaban(self, tmp_path):
        path = tmp_path / "users.json"
        _roster(path, _entrada(111, "alice"), _entrada(222, "bob"))

        cmd_add(_add_args(path, telegram_id=333, ssh_user="carol", name="Carol"))

        assert _ids(path) == [111, 222, 333]
        registro = JSONUserRegistry(str(path))
        assert registro.get_identity(111).ssh_user == "alice"
        assert registro.get_identity(222).ssh_user == "bob"

    def test_las_entradas_previas_quedan_byte_a_byte_iguales(self, tmp_path):
        """No alcanza con que sigan estando: tienen que seguir intactas.

        Una reescritura que «normalizara» las entradas viejas podría cambiar
        la clave SSH de alguien sin que nadie lo pidiera.
        """
        path = tmp_path / "users.json"
        previa = _entrada(111, "alice", ssh_host="viejo.cluster.edu.ar")
        _roster(path, previa)

        cmd_add(_add_args(path, telegram_id=222, ssh_user="bob"))

        assert json.loads(path.read_text())["users"][0] == previa

    def test_un_telegram_id_repetido_actualiza_la_entrada_sin_duplicarla(
        self, tmp_path
    ):
        """Comportamiento actual: `add` sobre un id existente es un upsert.

        No rechaza ni pide confirmación — pisa los datos viejos con los
        nuevos. Está fijado acá porque es la forma prevista de corregir la
        clave SSH de alguien, y porque un duplicado sí sería un problema:
        el registro se queda con la última entrada y la primera pasaría a
        ser un fantasma imposible de borrar de a una.
        """
        path = tmp_path / "users.json"
        _roster(path, _entrada(111, "alice"), _entrada(222, "bob"))

        cmd_add(
            _add_args(
                path,
                telegram_id=111,
                ssh_user="alicia",
                ssh_key="/home/becario/.ssh/id_nueva",
                name="Alicia",
            )
        )

        assert _ids(path) == [222, 111]  # la actualizada se reubica al final
        identidad = JSONUserRegistry(str(path)).get_identity(111)
        assert identidad.ssh_user == "alicia"
        assert identidad.ssh_key_path == "/home/becario/.ssh/id_nueva"
        assert JSONUserRegistry(str(path)).get_identity(222).ssh_user == "bob"


# ---------------------------------------------------------------------------
# Bajas
# ---------------------------------------------------------------------------


class TestQuitarMiembros:
    def test_borrar_a_uno_no_se_lleva_puestos_a_los_demas(self, tmp_path):
        """La baja también reescribe el archivo entero. El filtro tiene que
        sacar exactamente la entrada pedida: de más, y alguien pierde el
        acceso sin saber por qué; de menos, y alguien que se fue del grupo
        sigue pudiendo operar el cluster."""
        path = tmp_path / "users.json"
        alice, bob, carol = (
            _entrada(111, "alice"),
            _entrada(222, "bob"),
            _entrada(333, "carol"),
        )
        _roster(path, alice, bob, carol)

        cmd_remove(argparse.Namespace(file=str(path), telegram_id=222))

        assert json.loads(path.read_text())["users"] == [alice, carol]

    def test_el_registro_deja_de_reconocer_al_que_se_borro(self, tmp_path):
        path = tmp_path / "users.json"
        _roster(path, _entrada(111, "alice"), _entrada(222, "bob"))

        cmd_remove(argparse.Namespace(file=str(path), telegram_id=222))

        registro = JSONUserRegistry(str(path))
        assert registro.get_identity(222) is None
        assert registro.get_identity(111) is not None

    def test_borrar_un_id_que_no_esta_deja_el_roster_igual(self, tmp_path):
        """Un id inexistente —un dedazo, o alguien ya dado de baja— no puede
        vaciar ni alterar el roster; a lo sumo avisa que no encontró nada."""
        path = tmp_path / "users.json"
        _roster(path, _entrada(111, "alice"), _entrada(222, "bob"))

        cmd_remove(argparse.Namespace(file=str(path), telegram_id=999))

        assert _ids(path) == [111, 222]

    def test_avisa_cuando_no_encontro_a_nadie(self, tmp_path, capsys):
        path = tmp_path / "users.json"
        _roster(path, _entrada(111, "alice"))

        cmd_remove(argparse.Namespace(file=str(path), telegram_id=999))

        assert "No encontrado" in capsys.readouterr().out

    def test_borrar_dos_veces_al_mismo_es_inocuo(self, tmp_path):
        path = tmp_path / "users.json"
        _roster(path, _entrada(111, "alice"), _entrada(222, "bob"))

        cmd_remove(argparse.Namespace(file=str(path), telegram_id=222))
        cmd_remove(argparse.Namespace(file=str(path), telegram_id=222))

        assert _ids(path) == [111]


# ---------------------------------------------------------------------------
# Datos inválidos
# ---------------------------------------------------------------------------


class TestDatosInvalidosNoLleganAlArchivo:
    """El `ssh_user` termina en una línea de comando SSH del gateway.

    La validación de `ClusterIdentity` es la que impide que ahí entre un
    `;` o un `$(...)`, y tiene que correr ANTES de escribir: un roster con
    una entrada envenenada es un agujero que sobrevive al proceso que lo
    creó, aunque el registro después la descarte al leerla.
    """

    @pytest.mark.parametrize(
        "ssh_user",
        [
            "alice; rm -rf /",
            "alice$(whoami)",
            "alice`id`",
            "alice bob",
            "Alice",  # los usuarios UNIX válidos acá son en minúscula
            "../../etc/passwd",
        ],
    )
    def test_un_ssh_user_peligroso_corta_el_comando_con_codigo_de_error(
        self, tmp_path, ssh_user
    ):
        path = tmp_path / "users.json"

        with pytest.raises(SystemExit) as exc:
            cmd_add(_add_args(path, ssh_user=ssh_user))

        assert exc.value.code == 1

    @pytest.mark.parametrize(
        "flag, kwargs",
        [
            ("--ssh-user", {"ssh_user": ""}),
            ("--ssh-key", {"ssh_key": ""}),
            ("--ssh-user", {"ssh_user": "   "}),
        ],
    )
    def test_un_flag_pasado_vacio_falla_en_vez_de_ponerse_a_preguntar(
        self, tmp_path, capsys, flag, kwargs
    ):
        """Un flag vacío es un error, no un dato ausente.

        `argparse` ya distingue las dos cosas —`None` es «no lo pasaron»,
        `""` es «lo pasaron vacío»— y el CLI las confundía: caía al modo
        interactivo y pedía el dato por teclado. Para una persona eso apenas
        sorprende; para el cron que registra usuarios con una variable de
        entorno sin expandir es un proceso colgado esperando stdin, que es
        mucho peor que un error con código 1.
        """
        path = tmp_path / "users.json"

        with pytest.raises(SystemExit) as exc:
            cmd_add(_add_args(path, **kwargs))

        assert exc.value.code == 1
        assert flag in capsys.readouterr().out
        assert not path.exists()

    def test_omitir_el_flag_sigue_preguntando_por_teclado(
        self, tmp_path, monkeypatch
    ):
        """La contracara: rechazar el vacío no puede romper el modo
        interactivo, que es el que usa una persona dándose de alta."""
        path = tmp_path / "users.json"
        _teclado(monkeypatch, "alice", "")  # usuario SSH y host

        cmd_add(_add_args(path, ssh_user=None))

        assert JSONUserRegistry(str(path)).get_identity(111).ssh_user == "alice"

    def test_el_roster_previo_queda_intacto_cuando_los_datos_no_validan(
        self, tmp_path
    ):
        """El caso caro: no basta con no agregar al inválido, hay que no
        tocar a los que ya estaban. Un `_save` prematuro dejaría el archivo
        a medio escribir con el grupo entero adentro."""
        path = tmp_path / "users.json"
        _roster(path, _entrada(111, "alice"), _entrada(222, "bob"))
        antes = path.read_bytes()

        with pytest.raises(SystemExit):
            cmd_add(_add_args(path, telegram_id=333, ssh_user="carol; rm -rf /"))

        assert path.read_bytes() == antes

    def test_datos_invalidos_no_crean_el_archivo_si_no_existia(self, tmp_path):
        path = tmp_path / "users.json"

        with pytest.raises(SystemExit):
            cmd_add(_add_args(path, ssh_user="mala idea"))

        assert not path.exists()

    def test_explica_por_que_los_datos_no_sirven(self, tmp_path, capsys):
        """Quien administra el roster no lee pydantic: el mensaje tiene que
        decir cuál es el campo que rechazó."""
        with pytest.raises(SystemExit):
            cmd_add(_add_args(tmp_path / "users.json", ssh_user="alice; rm -rf /"))

        salida = capsys.readouterr().out
        assert "Datos inválidos" in salida
        assert "ssh_user" in salida


# ---------------------------------------------------------------------------
# Carga y guardado del archivo
# ---------------------------------------------------------------------------


class TestCargaDelRoster:
    def test_un_archivo_que_no_existe_es_un_roster_vacio(self, tmp_path):
        """La primera alta del proyecto ocurre sin `users.json`: si `_load`
        explotara ahí, no habría forma de empezar."""
        assert _load(tmp_path / "no_existe.json") == {"users": []}

    def test_un_json_corrupto_explota_en_vez_de_devolver_un_roster_vacio(
        self, tmp_path
    ):
        """Comportamiento buscado: tratar un archivo roto como «roster vacío»
        haría que la siguiente alta lo reescribiera con una sola persona,
        borrando al grupo entero. Mejor fallar ruidosamente."""
        path = tmp_path / "users.json"
        path.write_text('{"users": [', encoding="utf-8")

        with pytest.raises(json.JSONDecodeError):
            _load(path)

    def test_un_roster_corrupto_no_se_pisa_al_intentar_agregar(self, tmp_path):
        path = tmp_path / "users.json"
        path.write_text("esto no es json", encoding="utf-8")

        with pytest.raises(json.JSONDecodeError):
            cmd_add(_add_args(path))

        assert path.read_text(encoding="utf-8") == "esto no es json"

    def test_lo_guardado_se_vuelve_a_cargar_igual(self, tmp_path):
        path = tmp_path / "users.json"
        data = {"users": [_entrada(111, "alice")]}

        _save(path, data)

        assert _load(path) == data

    def test_el_archivo_guardado_es_legible_por_un_humano(self, tmp_path):
        """El roster se revisa y se edita a mano; una sola línea sin sangría
        vuelve imposible ver quién está adentro."""
        path = tmp_path / "users.json"

        _save(path, {"users": [_entrada(111, "alice")]})

        assert "\n" in path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Listado
# ---------------------------------------------------------------------------


class TestListado:
    def test_un_roster_vacio_lo_dice_en_vez_de_no_mostrar_nada(self, tmp_path, capsys):
        path = tmp_path / "users.json"
        _roster(path)

        cmd_list(argparse.Namespace(file=str(path)))

        assert "Roster vacío" in capsys.readouterr().out

    def test_sin_archivo_tambien_es_un_roster_vacio(self, tmp_path, capsys):
        cmd_list(argparse.Namespace(file=str(tmp_path / "no_existe.json")))

        assert "Roster vacío" in capsys.readouterr().out

    def test_muestra_id_y_usuario_ssh_de_cada_miembro(self, tmp_path, capsys):
        """Son los dos datos con los que se audita el acceso: quién es en
        Telegram y con qué cuenta entra al cluster."""
        path = tmp_path / "users.json"
        _roster(path, _entrada(111, "alice"), _entrada(222, "bob"))

        cmd_list(argparse.Namespace(file=str(path)))

        salida = capsys.readouterr().out
        assert "telegram_id=111" in salida
        assert "ssh_user=alice" in salida
        assert "telegram_id=222" in salida
        assert "ssh_user=bob" in salida

    def test_quien_no_puso_nombre_no_aparece_como_una_linea_rota(
        self, tmp_path, capsys
    ):
        path = tmp_path / "users.json"
        _roster(path, _entrada(111, "alice", display_name=""))

        cmd_list(argparse.Namespace(file=str(path)))

        assert "(sin nombre)" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Modo interactivo
# ---------------------------------------------------------------------------


class TestPreguntasPorTeclado:
    """El modo interactivo es el recomendado para gente sin experiencia.

    Si un prompt aceptara un vacío donde el dato es obligatorio, o si dejara
    pasar texto donde va un id numérico, el error saldría recién al validar
    —o peor, escribiría una entrada que el registro después descarta.
    """

    def test_devuelve_lo_tecleado_sin_espacios_de_sobra(self, monkeypatch):
        _teclado(monkeypatch, "  alice  ")

        assert _prompt("Usuario") == "alice"

    def test_repregunta_hasta_que_le_den_un_dato_obligatorio(self, monkeypatch, capsys):
        _teclado(monkeypatch, "", "   ", "alice")

        assert _prompt("Usuario") == "alice"
        assert "obligatorio" in capsys.readouterr().out

    def test_enter_acepta_el_valor_sugerido(self, monkeypatch):
        _teclado(monkeypatch, "")

        assert _prompt("Clave", default="~/.ssh/id_rsa") == "~/.ssh/id_rsa"

    def test_un_campo_opcional_admite_quedar_vacio(self, monkeypatch):
        _teclado(monkeypatch, "")

        assert _prompt("Nombre", required=False) == ""

    def test_el_id_se_devuelve_como_entero(self, monkeypatch):
        """El registro indexa por `int`: un id como string no matchea nunca
        y la persona queda afuera del bot sin ningún error visible."""
        _teclado(monkeypatch, "111")

        assert _prompt_int("id") == 111

    def test_repregunta_el_id_hasta_que_sea_numerico(self, monkeypatch, capsys):
        _teclado(monkeypatch, "@alice", "111")

        assert _prompt_int("id") == 111
        assert "no es un número" in capsys.readouterr().out


class TestAltaInteractiva:
    def test_los_datos_tecleados_terminan_en_el_roster(self, tmp_path, monkeypatch):
        """El camino completo del modo interactivo: sin flags, todo por
        teclado, y el resultado tiene que ser una identidad que el bot lee."""
        path = tmp_path / "users.json"
        _teclado(
            monkeypatch,
            "111",  # id de Telegram
            "alice",  # usuario SSH
            "/home/becario/.ssh/id_alice",  # clave
            "Alice",  # nombre
            "",  # host: Enter = usar el global
        )

        cmd_add(
            _add_args(path, telegram_id=None, ssh_user=None, ssh_key=None, name="")
        )

        identidad = JSONUserRegistry(str(path)).get_identity(111)
        assert identidad.ssh_user == "alice"
        assert identidad.display_name == "Alice"
        assert identidad.ssh_host is None

    def test_el_host_tecleado_queda_asociado_a_esa_persona(self, tmp_path, monkeypatch):
        path = tmp_path / "users.json"
        _teclado(
            monkeypatch,
            "111",
            "alice",
            "/home/becario/.ssh/id_alice",
            "",  # sin nombre
            "otro.cluster.edu.ar",
        )

        cmd_add(
            _add_args(path, telegram_id=None, ssh_user=None, ssh_key=None, name="")
        )

        identidad = JSONUserRegistry(str(path)).get_identity(111)
        assert identidad.ssh_host == "otro.cluster.edu.ar"
        assert identidad.display_name == ""

    def test_avisa_cuando_la_clave_ssh_no_esta_en_esta_maquina(
        self, tmp_path, capsys
    ):
        """Es un aviso, no un error: el bot suele correr en otro servidor.
        Frenar acá impediría preparar el roster desde una laptop."""
        path = tmp_path / "users.json"

        cmd_add(_add_args(path, ssh_key=str(tmp_path / "no_esta")))

        assert "Aviso" in capsys.readouterr().out
        assert JSONUserRegistry(str(path)).get_identity(111) is not None

    def test_no_avisa_nada_si_la_clave_existe(self, tmp_path, capsys):
        clave = tmp_path / "id_alice"
        clave.write_text("clave falsa", encoding="utf-8")

        cmd_add(_add_args(tmp_path / "users.json", ssh_key=str(clave)))

        assert "Aviso" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Línea de comandos
# ---------------------------------------------------------------------------


class TestMainDesdeLaLineaDeComandos:
    """El camino real: `argparse` armando el `Namespace` que reciben los
    comandos. Cubre que los flags documentados en el módulo sigan existiendo
    con el nombre que promete el docstring — un renombre silencioso rompería
    cualquier automatización que dé de alta gente sin intervención humana."""

    def test_add_por_flags_deja_una_identidad_que_el_bot_puede_leer(
        self, tmp_path, monkeypatch
    ):
        path = tmp_path / "users.json"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "manage_users.py",
                "--file",
                str(path),
                "add",
                "--telegram-id",
                "111",
                "--ssh-user",
                "alice",
                "--ssh-key",
                "/home/becario/.ssh/id_alice",
                "--name",
                "Alice",
            ],
        )

        main()

        identidad = JSONUserRegistry(str(path)).get_identity(111)
        assert identidad.ssh_user == "alice"
        assert identidad.display_name == "Alice"

    def test_add_por_flags_no_pregunta_nada_por_teclado(self, tmp_path, monkeypatch):
        """Con los tres datos obligatorios en la línea de comandos no puede
        quedarse esperando un `input()`: así se usa desde un script."""
        path = tmp_path / "users.json"

        def _sin_teclado(*_args, **_kwargs):
            raise AssertionError("no debería pedir nada por teclado")

        monkeypatch.setattr("builtins.input", _sin_teclado)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "manage_users.py",
                "--file",
                str(path),
                "add",
                "--telegram-id",
                "111",
                "--ssh-user",
                "alice",
                "--ssh-key",
                "/home/becario/.ssh/id_alice",
            ],
        )

        main()

        assert JSONUserRegistry(str(path)).get_identity(111).display_name == ""

    def test_remove_por_flags_saca_solo_a_quien_se_pidio(self, tmp_path, monkeypatch):
        path = tmp_path / "users.json"
        _roster(path, _entrada(111, "alice"), _entrada(222, "bob"))
        monkeypatch.setattr(
            sys,
            "argv",
            ["manage_users.py", "--file", str(path), "remove", "--telegram-id", "222"],
        )

        main()

        assert _ids(path) == [111]

    def test_remove_exige_el_id(self, tmp_path, monkeypatch):
        """Sin `--telegram-id` no hay baja posible: `argparse` tiene que
        cortar antes de que el comando decida borrar cualquier cosa."""
        path = tmp_path / "users.json"
        _roster(path, _entrada(111, "alice"))
        monkeypatch.setattr(
            sys, "argv", ["manage_users.py", "--file", str(path), "remove"]
        )

        with pytest.raises(SystemExit):
            main()

        assert _ids(path) == [111]

    def test_list_por_flags_muestra_el_roster(self, tmp_path, monkeypatch, capsys):
        path = tmp_path / "users.json"
        _roster(path, _entrada(111, "alice"))
        monkeypatch.setattr(
            sys, "argv", ["manage_users.py", "--file", str(path), "list"]
        )

        main()

        assert "telegram_id=111" in capsys.readouterr().out

    def test_sin_subcomando_no_hace_nada(self, monkeypatch):
        """El subcomando es obligatorio: un `manage_users.py` pelado no puede
        interpretarse como una acción sobre el control de acceso."""
        monkeypatch.setattr(sys, "argv", ["manage_users.py"])

        with pytest.raises(SystemExit):
            main()

    def test_el_roster_por_default_es_users_json(self, monkeypatch):
        """El default sale del `--file` del parser; si cambiara, el CLI
        escribiría un archivo que el bot no está leyendo."""
        capturado = {}
        monkeypatch.setattr(
            "scripts.manage_users.cmd_list", lambda args: capturado.update(file=args.file)
        )
        monkeypatch.setattr(sys, "argv", ["manage_users.py", "list"])

        main()

        assert capturado["file"] == "users.json"


# ---------------------------------------------------------------------------
# Regresiones de los defectos que encontró esta suite
# ---------------------------------------------------------------------------


class TestQuitarNoEscribeSiNoQuito:
    """`cmd_remove` guardaba SIEMPRE, hubiera borrado a alguien o no.

    Parecía inofensivo porque reescribir el mismo contenido no se nota. No
    lo era sobre un archivo que no existe: `_load` devuelve `{"users": []}`
    y el guardado lo materializaba, así que un `--file` con un dedazo dejaba
    un roster vacío NUEVO en disco y quien lo corrió creía haber dado de
    baja a alguien. Ninguno de los tests de borrado lo veía porque todos
    partían de un roster que ya existía.
    """

    def test_borrar_sobre_un_archivo_que_no_existe_no_lo_crea(self, tmp_path):
        path = tmp_path / "users.json"

        cmd_remove(argparse.Namespace(file=str(path), telegram_id=111))

        assert not path.exists(), "creó un roster vacío donde no había ninguno"

    def test_un_dedazo_en_la_ruta_no_deja_un_roster_fantasma(self, tmp_path, capsys):
        # El caso realista: el roster bueno está en `users.json` y alguien
        # escribe `user.json`. Ni se toca el bueno ni aparece el malo.
        bueno = tmp_path / "users.json"
        _roster(bueno, _entrada(111, "alice"))
        antes = bueno.read_bytes()

        cmd_remove(argparse.Namespace(file=str(tmp_path / "user.json"), telegram_id=111))

        assert not (tmp_path / "user.json").exists()
        assert bueno.read_bytes() == antes
        assert "No encontrado" in capsys.readouterr().out

    def test_no_toca_el_archivo_cuando_no_encontro_a_nadie(self, tmp_path):
        """No reescribir también deja la fecha de modificación quieta.

        Sobre un archivo de control de acceso, un `mtime` que se mueve sin
        que haya cambiado nada es ruido para cualquiera que audite o
        monitoree el archivo."""
        path = tmp_path / "users.json"
        _roster(path, _entrada(111, "alice"))
        mtime_antes = path.stat().st_mtime_ns

        cmd_remove(argparse.Namespace(file=str(path), telegram_id=999))

        assert path.stat().st_mtime_ns == mtime_antes

    def test_cuando_si_borra_escribe(self, tmp_path):
        # La contracara obvia, para que "no escribir" no se vuelva "no hacer".
        path = tmp_path / "users.json"
        _roster(path, _entrada(111, "alice"), _entrada(222, "bob"))

        cmd_remove(argparse.Namespace(file=str(path), telegram_id=111))

        assert _ids(path) == [222]


class TestElCLIAdministraLoQueElBotAcepta:
    """La asimetría que hacía inservible al CLI justo cuando hacía falta.

    `JSONUserRegistry.reload()` hace `raw.get("users", [])` y saltea las
    entradas inválidas logueando el error: el bot arranca igual y reporta
    «nadie registrado». El CLI hacía `data["users"]` y `u["telegram_user_id"]`
    directo, así que sobre el MISMO archivo moría con un `KeyError` pelado.
    O sea: el bot te decía que el roster estaba mal y la herramienta para
    arreglarlo era la única que no lo podía abrir.
    """

    def test_un_json_sin_la_clave_users_se_puede_dar_de_alta(self, tmp_path):
        path = tmp_path / "users.json"
        path.write_text("{}", encoding="utf-8")

        cmd_add(_add_args(path))

        assert JSONUserRegistry(str(path)).get_identity(111).ssh_user == "alice"

    def test_un_json_sin_la_clave_users_se_puede_listar(self, tmp_path, capsys):
        path = tmp_path / "users.json"
        path.write_text("{}", encoding="utf-8")

        cmd_list(argparse.Namespace(file=str(path)))

        assert "Roster vacío" in capsys.readouterr().out

    def test_una_entrada_incompleta_no_impide_dar_de_alta_a_otro(self, tmp_path):
        """Y la entrada rota NO se borra en el camino.

        Borrar en silencio algo que no entendemos es peor que dejarlo: nadie
        pidió esa baja, y el archivo es el único registro de que esa persona
        alguna vez estuvo."""
        path = tmp_path / "users.json"
        _roster(path, {"ssh_user": "sin_id"}, _entrada(222, "bob"))

        cmd_add(_add_args(path, telegram_id=333, ssh_user="carol"))

        entradas = json.loads(path.read_text())["users"]
        assert {"ssh_user": "sin_id"} in entradas
        assert JSONUserRegistry(str(path)).get_identity(333).ssh_user == "carol"

    def test_una_entrada_incompleta_no_impide_borrar_a_otro(self, tmp_path):
        path = tmp_path / "users.json"
        _roster(path, {"ssh_user": "sin_id"}, _entrada(222, "bob"))

        cmd_remove(argparse.Namespace(file=str(path), telegram_id=222))

        entradas = json.loads(path.read_text())["users"]
        assert entradas == [{"ssh_user": "sin_id"}]

    def test_listar_muestra_la_entrada_rota_marcada_en_vez_de_explotar(
        self, tmp_path, capsys
    ):
        """`list` es la herramienta con la que alguien va a diagnosticar por
        qué el bot no lo reconoce: tiene que poder MOSTRAR el problema."""
        path = tmp_path / "users.json"
        _roster(path, _entrada(111, "alice"), {"display_name": "Sin datos"})

        cmd_list(argparse.Namespace(file=str(path)))

        salida = capsys.readouterr().out
        assert "alice" in salida
        assert "entrada inválida" in salida
        assert "telegram_user_id" in salida

    def test_una_clave_users_que_no_es_lista_falla_limpio(self, tmp_path, capsys):
        """Acá sí se corta, y a propósito: normalizar a `[]` en silencio
        perdería lo que hubiera adentro, que es el mismo riesgo que motiva
        que un JSON roto explote."""
        path = tmp_path / "users.json"
        path.write_text('{"users": {"111": "alice"}}', encoding="utf-8")

        with pytest.raises(SystemExit) as exc:
            cmd_add(_add_args(path))

        assert exc.value.code == 1
        assert "no es una lista" in capsys.readouterr().out
        assert path.read_text(encoding="utf-8") == '{"users": {"111": "alice"}}'

    def test_un_json_que_no_es_objeto_falla_limpio(self, tmp_path, capsys):
        path = tmp_path / "users.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")

        with pytest.raises(SystemExit) as exc:
            cmd_list(argparse.Namespace(file=str(path)))

        assert exc.value.code == 1
        assert "no es un objeto JSON" in capsys.readouterr().out
