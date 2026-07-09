# ADR-0003: Confirmación humana para acciones destructivas

## Contexto

El enrutador es un LLM interpretando lenguaje natural (a veces transcrito
de audio). Dos acciones tienen consecuencias que no se pueden deshacer
sobre recursos compartidos del grupo: enviar un cálculo (`sbatch`, que
consume cupo/tiempo de cómputo asignado) y cancelar un trabajo (`scancel`,
que puede tirar horas de cálculo en curso). Ejecutar directo confiando en
que el LLM interpretó bien la orden es asumir un riesgo innecesario:
errores de transcripción de audio, ambigüedad del pedido, o una
"alucinación" del modelo tienen aquí un costo real de cómputo, no solo un
mensaje de texto equivocado.

## Decisión

`enviar_slurm` y `cancelar_calculo` nunca ejecutan directo. El caso de uso
correspondiente (`_prepare_submit` / `_prepare_cancel` en
`BecarioService`) valida los parámetros, arma una `PendingAction` con una
descripción legible del comando exacto que se ejecutaría, y la deja en un
`ConfirmationStore` con un token de un solo uso y vencimiento (TTL, 10
minutos por defecto). El bot muestra la descripción con botones ✅/❌; solo
al confirmar se llama al `ClusterGateway`. El resto de las acciones
(`revisar_estado`, `consultar_db`, `modificar_estructura`) son de solo
lectura o generan un archivo local sin tocar la cola de Slurm, y ejecutan
directo.

## Consecuencias

**A favor:** un test (`test_confirm_executes_exactly_once`) prueba que
confirmar dos veces el mismo token ejecuta una sola vez — el token se
consume, no hay reenvíos accidentales. La descripción mostrada usa los
valores ya sanitizados por el dominio, así que lo que el usuario confirma
es exactamente lo que se va a ejecutar, no una re-interpretación.

**En contra:** agrega una vuelta de mensaje para cada acción destructiva;
en un flujo de trabajo repetitivo (enviar la misma clase de cálculo muchas
veces) puede sentirse pesado. No se implementó una opción de "confiar en
mí para este tipo de pedido" — quedó deliberadamente afuera para no
reintroducir el riesgo que esta decisión busca evitar.
