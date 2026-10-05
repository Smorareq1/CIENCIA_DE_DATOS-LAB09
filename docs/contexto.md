# Contexto del laboratorio 9

Fuente para responder preguntas y para la defensa oral. Los números salieron de una sola corrida de `laboratorio9_particion.ipynb` sobre `pedidos_ruta_verde.csv`. No hace falta volver a ejecutar el notebook para contestar.

## Curso y grupo

Universidad Rafael Landívar, Facultad de Ingeniería, Ciencia de Datos, sección 2, segundo semestre 2026. Docente: Ing. Max Cerna. Laboratorio 9: «¿Importa cómo partimos los datos?».

| Integrante | Carné |
|---|---|
| Erick Eduardo Rivas Avalos | 1116323 |
| Sebastián Alejandro Morales De La Cruz | 1057123 |
| Jackeline Raquel Albizures Quevedo | 1122323 |
| David Stuardo Monje Palomo | 1019123 |
| Christopher Javier Yuman Valdez | 1160223 |

El enunciado pide tres roles (quien programa, quien registra, quien cuestiona). El grupo los cubrió en conjunto. En la defensa cualquiera puede explicar cualquier parte. El criterio de calificación es el razonamiento, no el valor de la métrica. Una predicción equivocada con buena explicación vale más que un número correcto sin justificación. Defensa oral: 30 puntos. Notebook A–F con predicciones: 35. Las 17 preguntas: 20. Tabla comparativa: 10. Para llevar a la clase: 5.

Uso de IA: Cursor ordenó el notebook, ejecutó el código y redactó `docs/`. El razonamiento se reconstruye sin la herramienta: lo que se mueve no es el modelo, es el corte.

## Regla del experimento

El modelo no se modifica entre partes. Solo cambia cómo se separan train y test.

- M1: `LogisticRegression(max_iter=1000, class_weight='balanced')` dentro de un `Pipeline`. Numéricas (`distancia_km`, `total_pedido_gtq`): mediana y `StandardScaler`. Categóricas (`zona`, `categoria_restaurante`, `metodo_pago`): `OneHotEncoder(handle_unknown='ignore')`.
- M0: `DummyClassifier(strategy='most_frequent')`. Siempre predice la clase más común.
- Cada F1 de M1 se reporta al lado del F1 de M0.
- Métrica de clasificación: F1 de la clase positiva. Más alto es mejor.
- La pregunta de regresión (tiempo de entrega, MAE) se plantea pero no se entrena. El laboratorio mide solo la clasificación.
- `class_weight='balanced'` empuja a marcar la clase rara. Por eso el recall a veces es decente y la precisión es baja (entre 0.05 y 0.16). El F1 se queda bajo aunque M1 le gane a M0.

## Datos

Archivo: `pedidos_ruta_verde.csv`.

- 626 pedidos. 7 restaurantes. Del 1 de enero de 2026 al 31 de agosto de 2026.
- Nulos: `distancia_km` 20, `calificacion` 35. El resto completo.
- `tiempo_entrega_min`: mínimo −9.5, máximo 196.5. Cuatro tiempos negativos (−9.5, −5.8, −3.9, −3.0), imposibles. 196.5 minutos es extremo; el filtro de regresión (`> 0` y `< 200`) lo conserva. Esos cuatro negativos quedan fuera de esa pregunta.
- `metodo_pago` duplicado por mayúsculas: Tarjeta 309, Efectivo 223, Transferencia 79, EFECTIVO 7, TARJETA 6, TRANSFERENCIA 2. No se unifica: cambiar el texto cambiaría el modelo y la comparación entre grupos.
- `total_pedido_gtq` tiene cola larga (máximo 636.17; 8 pedidos por encima de 200). No se toca.
- Calificaciones: 1 estrella 17, 2 estrellas 37, 3 estrellas 78, 4 estrellas 171, 5 estrellas 288, sin nota 35.
- Pregunta de clasificación: `cal = df.dropna(subset=['calificacion'])`, 591 pedidos. `baja = 1` si calificación ≤ 2. Son 54 pedidos, el **9.14 %**.
- Pedidos por restaurante (todos / con nota): Tacos El Chapín 98/91, Burger Xelajú 97/91, Pollo Real 93/87, Café Chuwa 91/87, Pupusas Doña Lupe 91/85, Sushi Ixchel 85/82, Pizza Volcán 71/68.
- Tasa de baja entre los que tienen nota: Pollo Real 4.6 %, Pupusas Doña Lupe 5.9 %, Café Chuwa 6.9 %, Burger Xelajú 8.8 %, Pizza Volcán 8.8 %, Sushi Ixchel 13.4 %, Tacos El Chapín 15.4 %.

### Pregunta 1

Para predecir el tiempo de entrega cuando entra el pedido no se pueden usar `tiempo_entrega_min` (todavía no ocurrió; es el objetivo) ni `calificacion` (el cliente califica después). Fecha, ubicación, restaurante, categoría, método de pago, total y distancia sí existen. La distancia se estima con las dos direcciones antes de que salga el repartidor.

### Pregunta 2

Predicción: F1 de M0 = 0, porque «no baja» es mayoría y el dummy nunca predice la clase positiva. Resultado: 0 en todas las partes. Acertó.

## Parte A — azar, 80/20, semillas 0, 1, 7, 42, 2026

Predicción: F1 parecido, brecha de unas centésimas. Falló: la brecha fue 0.150.

Test de 119 filas en las cinco.

| Semilla | Positivos | % | F1 M1 | Recall | Precisión | F1 M0 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 15 | 12.6 % | 0.226 | 0.400 | 0.158 | 0 |
| 1 | 12 | 10.1 % | 0.246 | 0.583 | 0.156 | 0 |
| 7 | 11 | 9.2 % | 0.113 | 0.273 | 0.071 | 0 |
| 42 | 14 | 11.8 % | 0.167 | 0.357 | 0.109 | 0 |
| 2026 | 3 | 2.5 % | 0.095 | 0.667 | 0.051 | 0 |

Promedio F1: 0.169. Mínimo 0.095, máximo 0.246.

Matriz de la semilla 2026 (filas reales, columnas predichas): 79 verdaderos negativos, 37 falsos positivos, 1 falso negativo, 2 verdaderos positivos. Atrapó 2 de 3 bajas y disparó 37 alarmas. Por eso el recall se ve alto y el F1 es el peor.

Pregunta 3: 0.150 de diferencia. Es mucho. El mismo modelo parece más del doble de bueno solo por la semilla.

Pregunta 4: si otro grupo reporta otro F1, los dos pueden tener razón. No se reporta el mejor. Se reporta el rango o el promedio (0.169) y el F1 de M0.

Pregunta 5: no es «más positivos, mejor F1». La semilla 0 tiene 15 positivos y F1 0.226; la semilla 1 tiene 12 y F1 0.246. Lo que sí pasa es que un test que no conserva el 9 % deja de ser comparable. La semilla 2026 midió otro problema.

## Parte B — las mismas semillas con stratify

Predicción: el promedio no sube de forma clara; baja la dispersión y el % de positivos queda fijo. Acertó en lo importante. El promedio subió poco y la brecha sí bajó.

Las cinco semillas dejan 11 positivos (9.2 %).

| Semilla | F1 M1 | Recall | F1 M0 |
|---|---:|---:|---:|
| 0 | 0.145 | 0.364 | 0 |
| 1 | 0.213 | 0.455 | 0 |
| 7 | 0.156 | 0.455 | 0 |
| 42 | 0.230 | 0.636 | 0 |
| 2026 | 0.140 | 0.364 | 0 |

Promedio 0.177. Brecha 0.089 (de 0.140 a 0.230).

Pregunta 6: cambian promedio y brecha, pero no igual. Promedio 0.169 → 0.177. Brecha 0.150 → 0.089. Desaparece el test de 3 positivos.

Pregunta 7: stratify no enseña nada nuevo. Controla cuántos positivos hay en el test, no cuáles. El F1 sigue moviéndose.

Pregunta 8: es indispensable si la clase es escasa y el test es chico (aquí 9 % y 119 filas). Da casi igual si las clases están cerca de 50/50 y el test es grande.

## Parte C — por fecha

Tasa de baja por mes (pedidos con nota):

| Mes | n | % baja |
|---|---:|---:|
| 2026-01 | 70 | 7.1 % |
| 2026-02 | 81 | 9.9 % |
| 2026-03 | 79 | 10.1 % |
| 2026-04 | 77 | 7.8 % |
| 2026-05 | 60 | 6.7 % |
| 2026-06 | 75 | 12.0 % |
| 2026-07 | 76 | 1.3 % (1 pedido) |
| 2026-08 | 73 | 17.8 % (13 pedidos) |

Enero–junio anda entre 6.7 % y 12.0 %. Julio y agosto juntos vuelven a ~9 % y esconden el quiebre.

Predicción, después de ver la gráfica: el F1 va a salir peor que el azar. Acertó, y el golpe fue más fuerte de lo esperado.

Train: fecha < 2026-07-01, 442 pedidos, 9.05 % de bajas. Test: julio–agosto, 149 pedidos, 9.40 % de bajas, 14 positivos.

F1 M1 = 0.037. Recall = 0.071 (1 de 14). Precisión ≈ 0.025. Matriz: 96 verdaderos negativos, 39 falsos positivos, 13 falsos negativos, 1 verdadero positivo. F1 M0 = 0.

Pregunta 9: no falló porque la tasa global cambiara (9.4 % contra 9.0 %). Falló porque julio está casi vacío y agosto concentra 13 de las 14 bajas. El corte al azar mete agosto en el train y el modelo ve el futuro.

Pregunta 10: para septiembre, la medición de fecha se parece más. El azar contesta «qué pasa si el futuro es una mezcla del pasado».

Pregunta 11: un pedido del 8 de agosto en el train y uno del 3 de febrero en el test invierten el tiempo. El examen estudia con el futuro. Por eso el F1 aleatorio es optimista.

## Parte D — por restaurante

Predicción: el F1 baja si dos restaurantes completos quedan fuera. Acertó en la dirección.

Corte al azar, semilla 42: los 7 restaurantes del test también están en el train. No simula un local nuevo.

`GroupShuffleSplit(n_splits=1, test_size=2/7, random_state=42)` dejó en el test a Burger Xelajú y Café Chuwa (los de tasa media-baja). Train: Pizza Volcán, Pollo Real, Pupusas Doña Lupe, Sushi Ixchel, Tacos El Chapín. Test: 178 pedidos, 14 positivos, tasa 7.9 %. Train: tasa 9.7 %.

F1 M1 = 0.103. Recall = 0.214. F1 M0 = 0.

Pregunta 12: bajó frente al promedio aleatorio (0.169), sin llegar al 0.037 de la fecha. El modelo no usa el nombre del restaurante, así que zona, categoría, pago, distancia y total sí se transfieren. Aun así cada local tiene su tasa, y el train incluye a los dos con más quejas.

Pregunta 13: para un restaurante nuevo, la parte D es la referencia (0.103), no el 0.246. Azar, stratify y fecha siguen usando los mismos siete locales.

## Parte E — riesgo_rest

Columna nueva: promedio de `baja` por restaurante.

- Versión 1: el promedio usa todos los pedidos y después se parte. Fuga: el test entra en el insumo.
- Versión 2: primero `train_test_split` con semilla 42 y stratify; el promedio solo usa el train y se aplica al test.

Predicción: la versión 1 iba a ganar. Falló. Las dos dieron F1 0.230, recall 0.636, M0 = 0, y las predicciones del test fueron idénticas. Es el mismo F1 de la parte B con semilla 42, que no usaba la columna.

Diferencia del promedio (todo el archivo menos solo el train): Burger +0.009, Café −0.008, Pizza −0.015, Pollo +0.002, Pupusas −0.010, Sushi −0.020, Tacos +0.034. Chica, porque los siete locales están en los dos lados y el train se lleva el 80 %.

Pregunta 14: empataron. La versión 1 sí vio etiquetas del test. En este corte no alcanzó para cambiar una predicción. Si un restaurante hubiera quedado solo en el test, la versión 1 le habría calculado el riesgo con las notas del examen.

Pregunta 15: en producción un pedido nuevo no tiene calificación. El promedio válido es el de la versión 2. Reportar la versión 1 no se sostiene el día en que train y archivo completo se separen más.

Pregunta 16: la mediana del `SimpleImputer` y la media y la desviación del `StandardScaler` tendrían la misma fuga si se calcularan antes de partir. Dentro de M1 no pasa: están en el `Pipeline` y el `fit` solo ve el train.

## Qué número se reporta

Al gerente: **F1 = 0.037** de la parte C, con M0 en 0. Es el corte que se parece a septiembre. No se reporta 0.246. Hay que decir que atrapó 1 de 14 bajas: le gana al dummy y no está listo para operar.

| Escenario | Corte |
|---|---|
| (a) Cada noche, sobre el día siguiente | Por fecha |
| (b) Restaurantes nuevos | Por restaurante (parte D) |
| (c) Comparar dos modelos con los datos actuales | Azar con stratify, promedio de varias semillas, no la mejor |

Pregunta 17: la más alta es A semilla 1, F1 0.246. La más confiable para pedidos que no existen es C, F1 0.037. No son la misma. La más alta es la más optimista.

## Tabla comparativa

| Experimento | F1 M1 | F1 M0 | Qué simula |
|---|---:|---:|---|
| A, mejor semilla (1) | 0.246 | 0 | Meses mezclados; el corte que más favoreció |
| A, peor semilla (2026) | 0.095 | 0 | El test se quedó con 3 bajas |
| B, stratify, promedio de 5 | 0.177 | 0 | Azar que conserva el 9.2 % de bajas. Sirve para comparar modelos |
| C, jul–ago | 0.037 | 0 | Septiembre, pedidos que todavía no existen |
| D, por restaurante | 0.103 | 0 | Un local que no aportó pedidos al train |
| E, versión 1 | 0.230 | 0 | Historial calculado espiando el test |
| E, versión 2 | 0.230 | 0 | El mismo historial, solo con el train |

## Para la clase

Sorpresa principal: julio–agosto tiene casi la misma tasa global que enero–junio (9.4 % contra 9.0 %) y el F1 igual cayó a 0.037, porque julio está en 1.3 % y agosto en 17.8 %. Segunda sorpresa: la fuga de la parte E no subió el F1. Una medición inválida puede verse igual que la válida.

Pregunta abierta: no se hizo un corte que camine mes a mes (entrenar hasta junio y probar julio; entrenar hasta julio y probar agosto). Agosto concentra casi todas las bajas del segundo período. Tampoco se midió reentrenar cada semana.

Regla, sin tecnicismos: no califiquemos el modelo con pedidos que ya pasaron mezclados con los que queremos adivinar; califiquémoslo solo con lo que todavía no ha pasado.

## Dónde está cada cosa

- Notebook ejecutado: `laboratorio9_particion.ipynb` y `docs/notebook.html`.
- Reporte con figuras: `docs/README.md`.
- Este archivo: `docs/contexto.md`.
- Gráficas: `docs/capturas/`.
- CSV de la tabla: `docs/tabla_comparativa.csv`.
- Enunciado: `enunciado.pdf`.
- Datos: `pedidos_ruta_verde.csv`.

No hace falta crear el entorno ni abrir Jupyter para responder. Si alguien pide rehacer una cifra que no está en este archivo, se dice que no se midió, en lugar de inventarla.
