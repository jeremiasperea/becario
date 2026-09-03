"""La mitad SFTP del gateway SSH, que hasta hoy no tenía un solo test.

`test_infrastructure.py::TestSSHCommandConstruction` cubre los métodos que
arman un comando de shell (submit, sacct, mkdir, tree). Lo que baja y sube
archivos por SFTP —y `job_exit_code`, que sí arma comando pero nadie
miraba— quedó afuera, y ahí es donde este adaptador toma las decisiones
más finas: cuándo un fallo significa «no está» y cuándo significa «no
pude mirar».

Que el hueco era real está medido: antes de estos tests se le podían meter
dos mutaciones al módulo —`file_exists` devolviendo `False` en vez de
`None` ante un fallo de transporte, y `job_exit_code` leyendo la señal en
vez del código de salida— y la suite entera seguía en verde.

Nada de esto toca la red: se intercepta `_connection()` con un cliente de
mentira y `_run()` con una grabadora, igual que hace `RecordingGateway`.
"""
from __future__ import annotations

import paramiko
import pytest

from becario.domain.models import CommandResult, JobId
from becario.infrastructure.ssh_gateway import SSHClusterGateway


class _Handle:
    """Lo que devuelve `sftp.open(...)`, usado como context manager.

    Registra el argumento de cada `read`: el tope de bytes es una decisión
    de diseño (no bajar a memoria un OUTCAR de gigabytes) y solo se puede
    verificar mirando qué se le pidió de verdad al servidor, no el
    resultado — un archivo chico se lee igual con tope y sin tope.
    """

    def __init__(self, contenido: bytes) -> None:
        self._contenido = contenido
        self.reads: list = []

    def read(self, size=None):
        self.reads.append(size)
        return self._contenido if size is None else self._contenido[:size]

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class _SFTP:
    """Doble del canal SFTP de paramiko.

    Cada operación puede configurarse para levantar una excepción, porque
    lo que se está probando es justamente el mapeo de fallos: qué
    excepción produce qué respuesta.
    """

    def __init__(
        self,
        *,
        contenido: bytes = b"",
        entradas: list[str] | None = None,
        stat_error: BaseException | None = None,
        listdir_error: BaseException | None = None,
        open_error: BaseException | None = None,
        put_error: BaseException | None = None,
    ) -> None:
        self.handle = _Handle(contenido)
        self._entradas = entradas or []
        self._stat_error = stat_error
        self._listdir_error = listdir_error
        self._open_error = open_error
        self._put_error = put_error
        self.puestos: list[tuple[str, str]] = []
        self.cierres = 0

    def stat(self, path):
        if self._stat_error is not None:
            raise self._stat_error
        return object()

    def listdir(self, path):
        if self._listdir_error is not None:
            raise self._listdir_error
        return list(self._entradas)

    def open(self, path, mode):
        if self._open_error is not None:
            raise self._open_error
        return self.handle

    def put(self, local, remote):
        if self._put_error is not None:
            raise self._put_error
        self.puestos.append((str(local), str(remote)))

    def close(self):
        self.cierres += 1


class _Gateway(SSHClusterGateway):
    """Gateway sin red: graba los comandos y sirve un SFTP de mentira.

    `_run` guarda además el flag `reintentable` de cada llamada. No es un
    detalle: qué operación se puede repetir y cuál no es una decisión
    documentada método por método en este módulo, y un `mv` marcado como
    reintentable por accidente informaría un fallo falso sobre una corrida
    que sí se movió.
    """

    def __init__(self, sftp: _SFTP | None = None, salidas=None) -> None:
        super().__init__(host="fake", user="fake", key_path="/dev/null")
        self.commands: list[str] = []
        self.reintentables: list[bool] = []
        self._sftp = sftp
        self._salidas = list(salidas or [])

    def _run(self, command: str, *, reintentable: bool = False) -> CommandResult:
        self.commands.append(command)
        self.reintentables.append(reintentable)
        if self._salidas:
            return self._salidas.pop(0)
        return CommandResult(ok=True, stdout="ok")

    def _connection(self):
        sftp = self._sftp

        class _Client:
            def open_sftp(self):
                if sftp is None:
                    raise paramiko.SSHException("el transporte se cayó")
                return sftp

        return _Client()


# ---------------------------------------------------------------------------
# file_exists: la distinción que se pagó
# ---------------------------------------------------------------------------


class TestFileExistsDistingueNoEstaDeNoPudeMirar:
    """Tres respuestas, no dos, y el docstring del método explica por qué.

    `FileNotFoundError` es una respuesta del servidor: el archivo no está.
    Un `SSHException`/`OSError` es otra cosa —no hubo conversación— y
    devolver `False` ahí equivale a afirmar «no existe» sin haber podido
    mirar. Río arriba eso se traduce en decirle al usuario que le falta un
    POTCAR que en realidad está ahí, y abortar un cálculo por una caída de
    red de dos segundos.

    Ojo con el orden de los `except`: `FileNotFoundError` es subclase de
    `OSError`, así que si alguien los reordena, el «no está» se convierte
    en «no pude mirar» y este caso deja de distinguirse.
    """

    def test_un_archivo_que_esta_da_true(self):
        gw = _Gateway(_SFTP())

        assert gw.file_exists("/data/POTCAR") is True

    def test_un_archivo_que_no_esta_da_false(self):
        gw = _Gateway(_SFTP(stat_error=FileNotFoundError("no such file")))

        assert gw.file_exists("/data/POTCAR") is False

    def test_un_fallo_de_transporte_no_afirma_que_el_archivo_no_existe(self):
        """La mutación que la suite entera no detectaba."""
        gw = _Gateway(_SFTP(stat_error=paramiko.SSHException("canal muerto")))

        assert gw.file_exists("/data/POTCAR") is None, (
            "devolver False acá es afirmar «no existe» sin haber podido mirar"
        )

    def test_un_oserror_que_no_es_archivo_faltante_tampoco(self):
        # Permisos, disco lleno del lado remoto, socket cortado: el servidor
        # no llegó a contestar sobre ESTE archivo.
        gw = _Gateway(_SFTP(stat_error=OSError("connection reset")))

        assert gw.file_exists("/data/POTCAR") is None

    def test_no_poder_abrir_el_canal_tampoco_es_una_respuesta(self):
        gw = _Gateway(sftp=None)

        assert gw.file_exists("/data/POTCAR") is None

    def test_el_canal_se_cierra_aunque_el_archivo_no_este(self):
        # El `finally` del método: una consulta fallida no puede dejar
        # canales colgados, que en un bot de larga vida se acumulan.
        sftp = _SFTP(stat_error=FileNotFoundError("no such file"))
        gw = _Gateway(sftp)

        gw.file_exists("/data/POTCAR")

        assert sftp.cierres == 1


# ---------------------------------------------------------------------------
# job_exit_code
# ---------------------------------------------------------------------------


class TestJobExitCode:
    """`ExitCode` de sacct viene como `salida:señal`, e interesa el primero.

    Quedarse con el segundo es la segunda mutación que la suite no veía: un
    trabajo que falló con código 2 pasaría a reportarse como exitoso
    (señal 0), y el usuario se enteraría de que su cálculo murió recién al
    abrir el OUTCAR.
    """

    def _gw(self, stdout: str, ok: bool = True) -> _Gateway:
        return _Gateway(salidas=[CommandResult(ok=ok, stdout=stdout)])

    def test_se_queda_con_el_codigo_de_salida_no_con_la_senal(self):
        assert self._gw("2:0\n").job_exit_code(JobId(value="42")) == 2

    def test_una_senal_distinta_de_cero_no_se_confunde_con_el_codigo(self):
        # Trabajo matado por SIGKILL sin código de salida propio: el código
        # es 0 y la señal 9. Leer al revés reportaría "falló con 9".
        assert self._gw("0:9\n").job_exit_code(JobId(value="42")) == 0

    def test_el_exito_limpio_es_cero(self):
        assert self._gw("0:0\n").job_exit_code(JobId(value="42")) == 0

    def test_usa_la_primera_linea_que_es_la_del_trabajo(self):
        # `-X` ya limita a la línea del trabajo, pero si el cluster igual
        # devuelve los pasos, la que resume el desenlace es la primera.
        assert self._gw("1:0\n0:0\n").job_exit_code(JobId(value="42")) == 1

    def test_pregunta_solo_por_el_exitcode_y_quotea_el_id(self):
        gw = self._gw("0:0")
        gw.job_exit_code(JobId(value="42"))

        assert gw.commands == [
            "sacct -j 42 --format=ExitCode --noheader --parsable2 -X"
        ]

    def test_leer_el_desenlace_se_puede_reintentar(self):
        # Es una consulta: repetirla no cambia nada en el cluster.
        gw = self._gw("0:0")
        gw.job_exit_code(JobId(value="42"))

        assert gw.reintentables == [True]

    @pytest.mark.parametrize("crudo", ["", "   \n", "no-es-un-codigo", ":", "a:b"])
    def test_lo_que_no_se_puede_leer_no_se_inventa(self, crudo):
        """`None` es «no sé», y es distinto de 0.

        Devolver 0 ante una salida ilegible sería declarar exitoso un
        trabajo del que no sabemos nada."""
        assert self._gw(crudo).job_exit_code(JobId(value="42")) is None

    def test_un_comando_que_fallo_no_da_codigo(self):
        assert self._gw("", ok=False).job_exit_code(JobId(value="42")) is None


# ---------------------------------------------------------------------------
# Lectura remota
# ---------------------------------------------------------------------------


class TestReadFile:
    def test_devuelve_el_contenido_decodificado(self):
        gw = _Gateway(_SFTP(contenido="ENCUT = 520\n".encode("utf-8")))

        assert gw.read_file("/data/INCAR") == "ENCUT = 520\n"

    def test_el_tope_de_bytes_viaja_al_servidor(self):
        """El punto del tope es NO bajar el archivo entero.

        Si el `max_bytes` se perdiera en el camino, el test de contenido
        seguiría pasando (el archivo de prueba es chico) y en producción un
        OUTCAR de gigabytes entraría entero a la memoria del bot."""
        sftp = _SFTP(contenido=b"0123456789")
        gw = _Gateway(sftp)

        assert gw.read_file("/data/OUTCAR", max_bytes=4) == "0123"
        assert sftp.handle.reads == [4]

    def test_sin_tope_se_lee_todo(self):
        sftp = _SFTP(contenido=b"0123456789")
        gw = _Gateway(sftp)

        assert gw.read_file("/data/INCAR") == "0123456789"
        assert sftp.handle.reads == [None]

    def test_los_bytes_ilegibles_no_tiran_abajo_la_lectura(self):
        # Un OUTCAR con basura binaria no puede reventar el bot: el método
        # decodifica con `errors="replace"`.
        gw = _Gateway(_SFTP(contenido=b"OK \xff\xfe fin"))

        resultado = gw.read_file("/data/OUTCAR")

        assert resultado is not None
        assert resultado.startswith("OK ")

    def test_un_fallo_devuelve_none_y_no_una_cadena_vacia(self):
        """`""` es un archivo vacío; `None` es no haber podido leerlo."""
        gw = _Gateway(_SFTP(open_error=paramiko.SSHException("canal muerto")))

        assert gw.read_file("/data/INCAR") is None

    def test_un_archivo_que_no_esta_tambien_da_none(self):
        gw = _Gateway(_SFTP(open_error=FileNotFoundError("no such file")))

        assert gw.read_file("/data/INCAR") is None


class TestListDir:
    def test_devuelve_las_entradas_ordenadas(self):
        """El orden lo fija el gateway, no el servidor.

        `listdir` de SFTP no promete orden. Sin el `sorted`, el listado que
        ve el usuario cambia de una consulta a la otra sin que nada haya
        cambiado en el cluster."""
        gw = _Gateway(_SFTP(entradas=["OUTCAR", "INCAR", "POSCAR"]))

        assert gw.list_dir("/data/run") == ["INCAR", "OUTCAR", "POSCAR"]

    def test_un_directorio_vacio_es_una_lista_vacia(self):
        gw = _Gateway(_SFTP(entradas=[]))

        assert gw.list_dir("/data/run") == []

    def test_un_fallo_devuelve_none_y_no_una_lista_vacia(self):
        """La misma distinción que en `file_exists`.

        `[]` dice «miré y no hay nada»; `None` dice «no pude mirar». Río
        arriba, confundirlos hace que una caída de red se reporte como una
        corrida sin archivos."""
        gw = _Gateway(_SFTP(listdir_error=paramiko.SSHException("canal muerto")))

        assert gw.list_dir("/data/run") is None


class TestHomeDir:
    def test_devuelve_el_home_remoto(self):
        gw = _Gateway(salidas=[CommandResult(ok=True, stdout="/home/alice")])

        assert gw.home_dir() == "/home/alice"

    def test_se_cachea_y_no_vuelve_a_preguntar(self):
        # El home no cambia durante una sesión y se consulta cada vez que
        # hay que volver absoluta una ruta relativa: preguntarlo de nuevo es
        # una ida y vuelta SSH por cada mensaje del usuario.
        gw = _Gateway(
            salidas=[
                CommandResult(ok=True, stdout="/home/alice"),
                CommandResult(ok=True, stdout="/home/OTRO"),
            ]
        )

        assert gw.home_dir() == "/home/alice"
        assert gw.home_dir() == "/home/alice"
        assert len(gw.commands) == 1

    def test_una_respuesta_que_no_es_una_ruta_no_se_cachea(self):
        """Cachear una respuesta mala la vuelve permanente.

        Si el comando falla o devuelve basura (un banner de login, un
        mensaje de error), guardarla dejaría al gateway resolviendo rutas
        contra ella hasta que se reinicie el proceso. Al no cachearla, el
        siguiente intento puede recuperarse."""
        gw = _Gateway(
            salidas=[
                CommandResult(ok=True, stdout="bienvenido al cluster"),
                CommandResult(ok=True, stdout="/home/alice"),
            ]
        )

        assert gw.home_dir() is None
        assert gw.home_dir() == "/home/alice"

    def test_un_comando_fallado_tampoco_se_cachea(self):
        gw = _Gateway(
            salidas=[
                CommandResult(ok=False, stderr="sin conexión"),
                CommandResult(ok=True, stdout="/home/alice"),
            ]
        )

        assert gw.home_dir() is None
        assert gw.home_dir() == "/home/alice"


# ---------------------------------------------------------------------------
# Subida de directorios
# ---------------------------------------------------------------------------


class TestUploadDir:
    """Subir una corrida entera: cada archivo necesita su directorio.

    Una corrida de VASP con barrido de ENCUT tiene subdirectorios, y SFTP
    no crea el padre solo: sin el `mkdir -p` previo, el `put` falla y la
    corrida sube a medias.
    """

    def _arbol(self, tmp_path):
        base = tmp_path / "run"
        (base / "encut_500").mkdir(parents=True)
        (base / "INCAR").write_text("ENCUT = 520\n")
        (base / "encut_500" / "POSCAR").write_text("Zr\n")
        return base

    def test_sube_todos_los_archivos_del_arbol(self, tmp_path):
        sftp = _SFTP()
        gw = _Gateway(sftp)

        result = gw.upload_dir(str(self._arbol(tmp_path)), "/data/runs/zr")

        assert result.ok
        subidos = {remoto for _, remoto in sftp.puestos}
        assert subidos == {
            "/data/runs/zr/INCAR",
            "/data/runs/zr/encut_500/POSCAR",
        }

    def test_crea_el_directorio_de_cada_archivo_antes_de_subirlo(self, tmp_path):
        gw = _Gateway(_SFTP())

        gw.upload_dir(str(self._arbol(tmp_path)), "/data/runs/zr")

        assert "mkdir -p /data/runs/zr/encut_500" in gw.commands

    def test_un_destino_con_espacios_viaja_quoteado(self, tmp_path):
        # `shlex.quote` solo agrega comillas cuando hacen falta, así que el
        # quoting solo se puede observar con una ruta que las necesite. El
        # nombre con espacio es realista: sale del nombre de la corrida.
        gw = _Gateway(_SFTP())

        gw.upload_dir(str(self._arbol(tmp_path)), "/data/runs/zr relax")

        assert "mkdir -p '/data/runs/zr relax/encut_500'" in gw.commands

    def test_las_rutas_remotas_son_posix_aunque_el_local_no_lo_sea(self, tmp_path):
        # Se arman con `as_posix()`: una barra invertida en una ruta remota
        # no es un separador, es parte del nombre del archivo.
        sftp = _SFTP()
        gw = _Gateway(sftp)

        gw.upload_dir(str(self._arbol(tmp_path)), "/data/runs/zr")

        assert all("\\" not in remoto for _, remoto in sftp.puestos)

    def test_una_barra_de_mas_en_el_destino_no_duplica_separadores(self, tmp_path):
        sftp = _SFTP()
        gw = _Gateway(sftp)

        gw.upload_dir(str(self._arbol(tmp_path)), "/data/runs/zr/")

        assert all("//" not in remoto for _, remoto in sftp.puestos)

    def test_un_fallo_a_mitad_de_camino_se_informa_como_transporte(self, tmp_path):
        """No alcanza con `ok=False`: el motivo decide si se reintenta.

        Una subida cortada por la red es transitoria y el reintento la
        recupera; un error de comando no. Ese tercer estado es lo que
        `CommandFailureReason` existe para llevar."""
        from becario.domain.models import CommandFailureReason

        gw = _Gateway(_SFTP(put_error=paramiko.SSHException("se cortó")))

        result = gw.upload_dir(str(self._arbol(tmp_path)), "/data/runs/zr")

        assert result.ok is False
        assert result.reason is CommandFailureReason.TRANSPORT

    def test_el_canal_se_cierra_aunque_la_subida_falle(self, tmp_path):
        sftp = _SFTP(put_error=OSError("disco lleno"))
        gw = _Gateway(sftp)

        gw.upload_dir(str(self._arbol(tmp_path)), "/data/runs/zr")

        assert sftp.cierres == 1


# ---------------------------------------------------------------------------
# Comandos que no arma nadie más
# ---------------------------------------------------------------------------


class TestMoveRun:
    """Mover la corrida de `.pending/` a su lugar es lo que la vuelve real."""

    def test_usa_mv_con_las_dos_rutas_quoteadas(self):
        gw = _Gateway()

        gw.move_run("/data/runs/.pending/zr relax", "/data/runs/zr relax")

        assert gw.commands == [
            "mv -T '/data/runs/.pending/zr relax' '/data/runs/zr relax'"
        ]

    def test_NO_es_reintentable(self):
        """Está documentado en el método y es contraintuitivo, así que se fija.

        Si el `mv` salió bien y la respuesta se perdió, el segundo intento
        falla con «no existe» e informaría un fallo falso sobre una corrida
        que en realidad ya está en su lugar."""
        gw = _Gateway()

        gw.move_run("/a", "/b")

        assert gw.reintentables == [False]


class TestConcatFiles:
    """El POTCAR se arma concatenando los de la biblioteca, del lado remoto."""

    def test_quotea_lo_que_necesita_comillas(self):
        # `shlex.quote` es quirúrgico: solo entrecomilla lo que el shell
        # partiría o interpretaría. Por eso el test mezcla una ruta con
        # espacio con una que no lo tiene — verificar solo la limpia no
        # probaría nada, y esperar comillas en las dos sería describir mal
        # la herramienta.
        gw = _Gateway()

        gw.concat_files(
            ["/pot/Zr sv/POTCAR", "/pot/O/POTCAR"], "/data/runs/zr relax/POTCAR"
        )

        assert gw.commands == [
            "cat '/pot/Zr sv/POTCAR' /pot/O/POTCAR > '/data/runs/zr relax/POTCAR'"
        ]

    def test_respeta_el_orden_recibido(self):
        # El orden del POTCAR tiene que coincidir con el de las especies del
        # POSCAR: permutarlo produce un cálculo que corre y da fruta.
        gw = _Gateway()

        gw.concat_files(["/pot/O/POTCAR", "/pot/Zr/POTCAR"], "/dest")

        assert gw.commands[0].index("/pot/O/POTCAR") < gw.commands[0].index("/pot/Zr/POTCAR")

    def test_sin_fuentes_no_corre_nada(self):
        """Un `cat > dest` sin fuentes se queda leyendo stdin para siempre.

        En un canal SSH sin terminal eso es un comando que nunca vuelve, o
        —peor— un POTCAR vacío escrito sobre el destino."""
        gw = _Gateway()

        result = gw.concat_files([], "/data/runs/zr/POTCAR")

        assert result.ok is False
        assert gw.commands == []
