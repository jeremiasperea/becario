"""Handlers de intents de B.E.C.A.R.I.O.

Cada módulo agrupa los casos de uso de un área; las funciones reciben la
fachada (`BecarioService`) como primer argumento `svc`. La privacidad ahora
la marca el módulo, no el prefijo `_` de los nombres.

Importar los submódulos directamente (`from .handlers import jobs`), como
hace `services.py`: no hay namespace plano re-exportado acá para no
mantener a mano un inventario paralelo de funciones.
"""
