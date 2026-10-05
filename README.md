# Laboratorio 9 — ¿Importa cómo partimos los datos?

Ciencia de Datos · Sección 2 · Segundo semestre 2026 · Ing. Max Cerna
Universidad Rafael Landívar

El modelo no cambia. Cambia el corte entre entrenamiento y prueba, y con él cambia el F1.

**Grupo:** Erick Eduardo Rivas Avalos (1116323), Sebastián Alejandro Morales De La Cruz (1057123), Jackeline Raquel Albizures Quevedo (1122323), David Stuardo Monje Palomo (1019123), Christopher Javier Yuman Valdez (1160223).

| Qué | Dónde |
|---|---|
| Notebook con las partes A–F, las 17 respuestas y las predicciones | [`laboratorio9_particion.ipynb`](laboratorio9_particion.ipynb) |
| Reporte con gráficas y tablas | [`docs/README.md`](docs/README.md) |
| Notebook ya ejecutado, para abrirlo en el navegador | [`docs/notebook.html`](docs/notebook.html) |
| Enunciado | [`enunciado.pdf`](enunciado.pdf) |
| Datos | [`pedidos_ruta_verde.csv`](pedidos_ruta_verde.csv) |

El número que se le reporta a Ruta Verde es el del corte por fecha: **F1 = 0.037**. El más alto del laboratorio, 0.246, es el corte al azar que más favoreció al modelo. No son la misma medición.

## Cómo volver a correrlo

```text
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute laboratorio9_particion.ipynb --inplace
```
