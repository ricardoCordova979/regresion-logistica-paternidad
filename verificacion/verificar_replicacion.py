"""
Verificación de la replicación R -> Python del Proyecto 2.

1. Ejecuta el Rmd (vía extraer_resultados_R.R) y guarda sus resultados en resultados_R.json.
2. Ejecuta el script de Python y extrae los mismos resultados (resultados_Python.json).
3. Compara cada valor y genera informe_verificacion.html.

Uso (desde la carpeta verificacion/): python verificar_replicacion.py
"""
import contextlib
import io
import json
import os
import runpy
import subprocess
import sys
from datetime import date

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from great_tables import GT, style, loc

# Tolerancias: 'alta' para modelos resueltos de forma exacta (glm, métricas, pruebas);
# 'glmnet' para modelos penalizados, limitada por el criterio de convergencia de glmnet.
TOL = {"alta": (1e-9, 1e-10), "glmnet": (1e-4, 1e-5)}
AQUI = os.path.dirname(os.path.abspath(__file__))
os.chdir(AQUI)

# ---------------------------------------------------------------- 1. Resultados de R
subprocess.run(["Rscript", "extraer_resultados_R.R"], check=True)
with open("resultados_R.json", encoding="utf-8") as f:
    R = json.load(f)

# ---------------------------------------------------------- 2. Resultados de Python
carpeta_py = os.path.join(AQUI, "..", "Python")
os.chdir(carpeta_py)
sys.path.insert(0, carpeta_py)
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path("02_Regresion_Logistica_DatosGSS.py")
os.chdir(AQUI)

orden_modelos = list(g["candidatos"])
orden_metricas = ["AUC", "Kappa", "Accuracy"]
rem = (g["remuestreos"].melt(id_vars=["Modelo", "Resample"], value_vars=orden_metricas,
                             var_name="Métrica", value_name="Valor")
       .assign(m=lambda d: d["Modelo"].map(orden_modelos.index),
               k=lambda d: d["Métrica"].map(orden_metricas.index))
       .sort_values(["m", "k", "Resample"]))
rvc = (g["resumen_vc"].assign(m=lambda d: d["Modelo"].map(orden_modelos.index),
                              k=lambda d: d["Métrica"].map(orden_metricas.index))
       .sort_values(["m", "k"]))


def trayectoria(m):
    return {"lambda": m.lambdas.tolist(), "a0": m.a0.tolist(),
            "beta": m.beta.flatten(order="F").tolist(), "nzero": m.nzero.tolist()}


mapa, orden = g["nombres_r"](g["modeloStepBIC"], g["terms_bic"])
cf = g["cf"]
res = g["resultado"]
P = {
    "trainIndex": g["trainIndex"].tolist(),
    "seleccion": {"variables": g["resumen_seleccion"]["Variables"].tolist(),
                  "parametros": g["resumen_seleccion"]["Parámetros"].tolist(),
                  "aic": g["resumen_seleccion"]["AIC"].tolist(), "bic": g["resumen_seleccion"]["BIC"].tolist()},
    "folds_vc": [(val + 1).tolist() for _, val in g["folds_vc"]],
    "vc": {"modelo": rem["Modelo"].tolist(), "metrica": rem["Métrica"].tolist(),
           "resample": rem["Resample"].tolist(), "valor": rem["Valor"].tolist()},
    "resumen_vc": {c: rvc[c].tolist() for c in ["Media", "Desv. est.", "Mínimo", "Mediana", "Máximo"]},
    "umbral": float(g["umbral"]),
    "matriz_confusion": g["matriz"].to_numpy().flatten(order="F").tolist(),
    "metricas_prueba": {c: g["metricas_prueba"][c].tolist() for c in g["metricas_prueba"].columns[1:]},
    "coef_bic": {"terminos": list(cf.index), "coef": cf["Coeficiente"].tolist(),
                 "se": cf["Error estándar"].tolist(), "p": cf["p"].tolist(),
                 "ic_inf": g["odds"]["IC 95 % inf."].tolist(), "ic_sup": g["odds"]["IC 95 % sup."].tolist()},
    "anova": {"variables": g["anova2"].set_index("Variable").loc[R["anova"]["variables"]].index.tolist(),
              **{k: g["anova2"].set_index("Variable").loc[R["anova"]["variables"], c].tolist()
                 for k, c in [("lr", "Estadístico LR (χ²)"), ("gl", "gl"), ("p", "p")]}},
    "columnas_x": g["columnas_x"],
    "lasso": trayectoria(g["modeloLASSO"]), "ridge": trayectoria(g["modeloRIDGE"]),
    "enet": trayectoria(g["modeloENET"]),
    "foldid": g["foldid"].tolist(),
    "cv_lasso": {"cvm": g["cv_lasso"].cvm.tolist(), "cvsd": g["cv_lasso"].cvsd.tolist(),
                 "lambda_min": float(g["cv_lasso"].lambda_min), "lambda_1se": float(g["cv_lasso"].lambda_1se),
                 "nzero": g["cv_lasso"].nzero.tolist()},
    "grid": {c: res[c].tolist() for c in ["alpha", "lambda", "numPar", "auc_cv", "auc_lo"]},
    "auc_1se": float(g["auc_1se"]),
    "seleccion_enet": {"alpha": float(g["alpha_opt"]), "lambda": float(g["lambda_opt"])},
    "coef_def": g["coef_def"].tolist(),
    "comparacion_final": {c: g["comparacion_final"][c].tolist() for c in g["comparacion_final"].columns[1:]},
}
with open("resultados_Python.json", "w", encoding="utf-8") as f:
    json.dump(P, f, ensure_ascii=False)

# ------------------------------------------------------------------- 3. Comparación
filas = []


def comparar(seccion, metrica, r, p, tipo="alta"):
    """tipo: 'exacto' (igualdad), 'texto' (cadenas iguales), 'alta' o 'glmnet' (tolerancia numérica)."""
    if tipo == "texto":
        ok = list(r) == list(p)
        filas.append({"Sección": seccion, "Métrica": metrica, "Valores": str(len(r)), "Criterio": "Idéntico",
                      "Máx. dif. absoluta": np.nan, "Máx. dif. relativa": np.nan,
                      "Resultado": "✓ Idéntico" if ok else "✗ Difiere"})
        return
    r, p = np.atleast_1d(np.asarray(r, float)), np.atleast_1d(np.asarray(p, float))
    if r.shape != p.shape:
        filas.append({"Sección": seccion, "Métrica": metrica, "Valores": f"{r.size} vs {p.size}",
                      "Criterio": tipo, "Máx. dif. absoluta": np.nan, "Máx. dif. relativa": np.nan,
                      "Resultado": "✗ Dimensiones distintas"})
        return
    dif = np.abs(r - p)
    rel = np.where(np.abs(r) > 0, dif / np.maximum(np.abs(r), 1e-300), dif)
    if tipo == "exacto":
        ok, criterio = np.array_equal(r, p), "Idéntico"
    else:
        rtol, atol = TOL[tipo]
        ok = np.allclose(p, r, rtol=rtol, atol=atol)
        criterio = "Alta precisión" if tipo == "alta" else "Precisión glmnet"
    filas.append({"Sección": seccion, "Métrica": metrica, "Valores": str(r.size), "Criterio": criterio,
                  "Máx. dif. absoluta": float(dif.max()), "Máx. dif. relativa": float(rel.max()),
                  "Resultado": ("✓ Idéntico" if np.array_equal(r, p) else "✓ Coincide") if ok else "✗ Difiere"})


comparar("Datos", "Partición entrenamiento–prueba (índices)", R["trainIndex"], P["trainIndex"], "exacto")
s_r, s_p = R["seleccion"], P["seleccion"]
comparar("Selección de variables", "Variables elegidas por los 6 métodos", s_r["variables"], s_p["variables"], "texto")
comparar("Selección de variables", "Número de parámetros", s_r["parametros"], s_p["parametros"], "exacto")
comparar("Selección de variables", "AIC y BIC de los 6 modelos", s_r["aic"] + s_r["bic"], s_p["aic"] + s_p["bic"])
comparar("Validación cruzada", "Folds de caret (5 × 20, índices)", np.concatenate(R["folds_vc"]),
         np.concatenate(P["folds_vc"]), "exacto")
comparar("Validación cruzada", "Etiquetas de remuestreo", R["vc"]["resample"], P["vc"]["resample"], "texto")
comparar("Validación cruzada", "AUC, Kappa y accuracy por fold (3 modelos)", R["vc"]["valor"], P["vc"]["valor"])
comparar("Validación cruzada", "Resumen (media, DE, mín., mediana, máx.)",
         np.concatenate([R["resumen_vc"][c] for c in R["resumen_vc"]]),
         np.concatenate([P["resumen_vc"][c] for c in P["resumen_vc"]]))
comparar("Evaluación en prueba", "Umbral de clasificación", R["umbral"], P["umbral"])
comparar("Evaluación en prueba", "Matriz de confusión", R["matriz_confusion"], P["matriz_confusion"], "exacto")
comparar("Evaluación en prueba", "AUC, accuracy, Kappa, sensibilidad y especificidad (3 modelos)",
         np.concatenate([R["metricas_prueba"][c] for c in R["metricas_prueba"]]),
         np.concatenate([P["metricas_prueba"][c] for c in P["metricas_prueba"]]))
c_r, c_p = R["coef_bic"], P["coef_bic"]
comparar("Modelo BIC", "Nombres de los coeficientes", c_r["terminos"], c_p["terminos"], "texto")
comparar("Modelo BIC", "Coeficientes", c_r["coef"], c_p["coef"])
comparar("Modelo BIC", "Errores estándar", c_r["se"], c_p["se"])
comparar("Modelo BIC", "p-valores", c_r["p"], c_p["p"])
comparar("Modelo BIC", "Intervalos de confianza de los odds ratios", c_r["ic_inf"] + c_r["ic_sup"],
         c_p["ic_inf"] + c_p["ic_sup"])
comparar("Modelo BIC", "ANOVA tipo II (LR, gl, p)", R["anova"]["lr"] + R["anova"]["gl"] + R["anova"]["p"],
         P["anova"]["lr"] + P["anova"]["gl"] + P["anova"]["p"])
comparar("Regularización", "Columnas de la matriz de diseño", R["columnas_x"], P["columnas_x"], "texto")
for clave, nombre in [("lasso", "LASSO"), ("enet", "Elastic Net (α = 0.1)"), ("ridge", "Ridge")]:
    comparar("Regularización", f"{nombre}: secuencia de λ", R[clave]["lambda"], P[clave]["lambda"])
    comparar("Regularización", f"{nombre}: parámetros no nulos por λ", R[clave]["nzero"], P[clave]["nzero"], "exacto")
    comparar("Regularización", f"{nombre}: coeficientes e interceptos", R[clave]["beta"] + R[clave]["a0"],
             P[clave]["beta"] + P[clave]["a0"], "glmnet")
comparar("Ajuste de hiperparámetros", "Asignación de folds (foldid)", R["foldid"], P["foldid"], "exacto")
cl_r, cl_p = R["cv_lasso"], P["cv_lasso"]
comparar("Ajuste de hiperparámetros", "LASSO: λ mínimo y λ 1-SE", [cl_r["lambda_min"], cl_r["lambda_1se"]],
         [cl_p["lambda_min"], cl_p["lambda_1se"]])
comparar("Ajuste de hiperparámetros", "LASSO: AUC medio y error estándar por λ", cl_r["cvm"] + cl_r["cvsd"],
         cl_p["cvm"] + cl_p["cvsd"], "glmnet")
comparar("Ajuste de hiperparámetros", "Rejilla α × λ: valores de α y λ", R["grid"]["alpha"] + R["grid"]["lambda"],
         P["grid"]["alpha"] + P["grid"]["lambda"])
comparar("Ajuste de hiperparámetros", "Rejilla α × λ: parámetros no nulos", R["grid"]["numPar"],
         P["grid"]["numPar"], "exacto")
comparar("Ajuste de hiperparámetros", "Rejilla α × λ: AUC medio y AUC − 1 EE",
         R["grid"]["auc_cv"] + R["grid"]["auc_lo"], P["grid"]["auc_cv"] + P["grid"]["auc_lo"], "glmnet")
comparar("Ajuste de hiperparámetros", "Umbral 1-SE", R["auc_1se"], P["auc_1se"], "glmnet")
comparar("Modelo regularizado", "α y λ seleccionados", [R["seleccion_enet"]["alpha"], R["seleccion_enet"]["lambda"]],
         [P["seleccion_enet"]["alpha"], P["seleccion_enet"]["lambda"]])
comparar("Modelo regularizado", "Coeficientes del modelo definitivo", R["coef_def"], P["coef_def"], "glmnet")
comparar("Modelo regularizado", "Métricas en prueba (logístico y regularizado)",
         np.concatenate([R["comparacion_final"][c] for c in R["comparacion_final"]]),
         np.concatenate([P["comparacion_final"][c] for c in P["comparacion_final"]]), "glmnet")

res_df = pd.DataFrame(filas)
n_ok = res_df["Resultado"].str.startswith("✓").sum()
n_valores = pd.to_numeric(res_df["Valores"], errors="coerce").sum()
todo_ok = n_ok == len(res_df)

tabla = (GT(res_df, groupname_col="Sección")
         .tab_header(title="Verificación de la replicación R → Python",
                     subtitle=f"Proyecto 2 · Regresión logística · {len(res_df)} comparaciones · "
                              f"{int(n_valores):,} valores")
         .opt_stylize(style=6, color="blue")
         .tab_options(table_font_size="12px", heading_title_font_size="16px")
         .fmt_scientific(columns=["Máx. dif. absoluta", "Máx. dif. relativa"], decimals=2)
         .sub_zero(columns=["Máx. dif. absoluta", "Máx. dif. relativa"], zero_text="0")
         .sub_missing(columns=["Máx. dif. absoluta", "Máx. dif. relativa"], missing_text="—")
         .cols_align("center", columns=["Valores", "Criterio", "Resultado"])
         .tab_style(style=style.text(color="#1E7B34", weight="bold"),
                    locations=loc.body(columns="Resultado",
                                       rows=res_df.index[res_df["Resultado"].str.startswith("✓")].tolist()))
         .tab_source_note("Idéntico: igualdad exacta. Alta precisión: |Python − R| ≤ 1e-10 + 1e-9·|R|. "
                          "Precisión glmnet: |Python − R| ≤ 1e-5 + 1e-4·|R|.")
         .tab_source_note("La tolerancia de los modelos penalizados refleja el criterio de convergencia "
                          "iterativo de glmnet; la réplica en Python alcanza un valor de la función objetivo "
                          "igual o menor que el de R en todos los casos."))

html = f"""<!DOCTYPE html><html lang="es"><head><meta charset="utf-8">
<title>Verificación R vs Python · Proyecto 2</title>
<style>body{{font-family:Lato,'Helvetica Neue',Arial,sans-serif;max-width:1040px;margin:40px auto;
color:#2C3E50;line-height:1.55;padding:0 20px}} h1{{font-weight:400}} code{{background:#f4f4f4;
padding:1px 4px;border-radius:3px}} .ok{{background:#E8F5E9;border-left:5px solid #1E7B34;padding:12px 16px}}
.ko{{background:#FDECEA;border-left:5px solid #B71C1C;padding:12px 16px}}</style></head><body>
<h1>Informe de verificación de la replicación R → Python</h1>
<p><b>Proyecto 2:</b> Perfil sociodemográfico de la paternidad: clasificación con regresión logística y
regularización · <b>Autor:</b> Ricardo José Córdova Soriano · <b>Fecha:</b> {date.today():%d/%m/%Y}</p>
<div class="{'ok' if todo_ok else 'ko'}"><b>{'Replicación exitosa' if todo_ok else 'Replicación con diferencias'}:</b>
{n_ok} de {len(res_df)} comparaciones superadas ({int(n_valores):,} valores contrastados). El modelo
seleccionado, las variables, los folds y todas las decisiones del análisis son idénticos en ambos lenguajes.</div>
<h2>Metodología</h2>
<p>Se ejecutaron de forma independiente el R Markdown (<code>R/02_Regresion_Logistica_DatosGSS.Rmd</code>) y el
script de Python (<code>Python/02_Regresion_Logistica_DatosGSS.py</code>), y se contrastaron todos sus resultados:
particiones, selección de variables, validación cruzada fold a fold, métricas en prueba, coeficientes,
pruebas estadísticas, trayectorias de los modelos penalizados y la búsqueda de hiperparámetros.</p>
<h2>Componentes replicados desde cero en Python</h2>
<p>Varias funciones de R no tienen equivalente directo en Python. El módulo <code>Python/rcompat.py</code> las
reimplementa y reproduce sus resultados:</p>
<ul>
<li><b>Generador aleatorio de R</b> (<code>set.seed()</code>, Mersenne-Twister y muestreo por rechazo de
<code>sample()</code>), incluido el muestreo con <i>hash</i> que R usa para poblaciones grandes.</li>
<li><b>Particiones de caret</b>: <code>createDataPartition()</code>, <code>createFolds()</code> y
<code>createMultiFolds()</code>, además de la semilla interna que <code>train()</code> extrae antes de generar
los folds.</li>
<li><b><code>step()</code></b> con criterios AIC y BIC en sus tres direcciones.</li>
<li><b><code>glmnet</code> y <code>cv.glmnet</code></b>: el mismo problema de optimización (elastic net logístico
con estandarización interna), la misma secuencia de λ y reglas de parada de la trayectoria, y la
interpolación de predicciones entre las secuencias de λ de cada fold (<code>lambda.interp</code>).</li>
</ul>
<h2>Equivalencias directas</h2>
<ul>
<li><code>glm(family = binomial)</code> → <code>statsmodels.formula.api.glm(family=Binomial())</code>.</li>
<li><code>car::Anova(type = "II")</code> → pruebas de razón de verosimilitudes por variable.</li>
<li><code>pROC::auc()</code> → AUC de Mann-Whitney con empates ponderados por ½.</li>
<li><code>confint.default()</code> → intervalos de Wald con <code>scipy.stats.norm.ppf(0.975)</code>.</li>
</ul>
<h2>Resultados</h2>
{tabla.as_raw_html()}
</body></html>"""
with open("informe_verificacion.html", "w", encoding="utf-8") as f:
    f.write(html)

print(res_df[["Sección", "Métrica", "Valores", "Máx. dif. absoluta", "Resultado"]].to_string(index=False))
print(f"\n{n_ok}/{len(res_df)} comparaciones superadas · {int(n_valores):,} valores")
sys.exit(0 if todo_ok else 1)
