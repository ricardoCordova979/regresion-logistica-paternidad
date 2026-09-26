# Ejecuta el código del Rmd y exporta los resultados numéricos clave a JSON
# Uso (desde la carpeta verificacion/): Rscript extraer_resultados_R.R
suppressMessages(library(jsonlite))

rmd <- normalizePath("../R/02_Regresion_Logistica_DatosGSS.Rmd")
script <- tempfile(fileext = ".R")
knitr::purl(rmd, output = script, quiet = TRUE)

env <- new.env()
dir_original <- setwd(dirname(rmd))
invisible(capture.output(suppressMessages(sys.source(script, envir = env))))
setwd(dir_original)

with(env, {
  trayectoria <- function(m) list(lambda = m$lambda, a0 = unname(m$a0),
                                  beta = unname(as.vector(as.matrix(m$beta))), nzero = unname(m$df))
  rem <- remuestreos |> arrange(Modelo, Métrica, Resample)
  cfb <- summary(modeloStepBIC)$coefficients
  resultados <- list(
    trainIndex = as.integer(trainIndex),
    seleccion = list(variables = resumen_seleccion$Variables, parametros = resumen_seleccion$Parámetros,
                     aic = resumen_seleccion$AIC, bic = resumen_seleccion$BIC),
    folds_vc = unname(lapply(vcrTodosModelos[[1]]$control$indexOut, as.integer)),
    vc = list(modelo = as.character(rem$Modelo), metrica = as.character(rem$Métrica),
              resample = rem$Resample, valor = rem$Valor),
    resumen_vc = as.list(resumen_vc |> arrange(Modelo, Métrica) |> select(-Modelo, -Métrica)),
    umbral = umbral,
    matriz_confusion = as.vector(cm_test$table),
    metricas_prueba = as.list(metricas_prueba[, -1]),
    coef_bic = list(terminos = rownames(cfb), coef = unname(cfb[, 1]), se = unname(cfb[, 2]),
                    p = unname(cfb[, 4]), ic_inf = odds$`IC 95 % inf.`, ic_sup = odds$`IC 95 % sup.`),
    anova = list(variables = rownames(anova2), lr = anova2$`LR Chisq`, gl = anova2$Df,
                 p = anova2$`Pr(>Chisq)`),
    columnas_x = colnames(x),
    lasso = trayectoria(modeloLASSO), ridge = trayectoria(modeloRIDGE), enet = trayectoria(modeloENET),
    foldid = as.integer(foldid),
    cv_lasso = list(cvm = cv.lasso$cvm, cvsd = cv.lasso$cvsd, lambda_min = cv.lasso$lambda.min,
                    lambda_1se = cv.lasso$lambda.1se, nzero = unname(cv.lasso$nzero)),
    grid = list(alpha = resultado$alpha, lambda = resultado$lambda, numPar = unname(resultado$numPar),
                auc_cv = resultado$auc_cv, auc_lo = resultado$auc_lo),
    auc_1se = auc_1se,
    seleccion_enet = list(alpha = alpha_opt, lambda = lambda_opt),
    coef_def = unname(coef_def[, 1]),
    comparacion_final = as.list(comparacion_final[, -1])
  )
  write_json(resultados, "resultados_R.json", digits = NA, auto_unbox = TRUE)
})
cat("Resultados de R exportados a resultados_R.json\n")
