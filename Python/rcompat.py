"""
rcompat: réplicas en Python de funciones de R usadas en el portafolio.

Permiten reproducir resultados de R con exactitud numérica:
- RRandom: generador de números aleatorios de R (set.seed, Mersenne-Twister, sample()).
- create_data_partition, create_folds, create_multi_folds: particiones de {caret}.
- caret_repeatedcv_folds: índices de train(method = "repeatedcv"), incluida la semilla interna de caret.
- step_glm: selección de variables de step() para modelos logísticos (AIC/BIC).
- Glmnet / cv_glmnet: regresión logística penalizada (elastic net) equivalente a {glmnet}.

Autor: Ricardo José Córdova Soriano
"""
import math

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


# ------------------------------------------------------------------ Generador de R
class RRandom:
    """Réplica del generador de R: set.seed() + Mersenne-Twister + sample() (sample.kind = 'Rejection')."""

    def __init__(self, seed):
        s = seed & 0xFFFFFFFF
        for _ in range(50):                               # mezcla inicial de la semilla en R
            s = (69069 * s + 1) & 0xFFFFFFFF
        key = np.empty(625, dtype=np.uint32)
        for j in range(625):
            s = (69069 * s + 1) & 0xFFFFFFFF
            key[j] = s
        self.bg = np.random.MT19937()
        # key[0] corresponde a 'mti', que R fija en 624 (FixupSeeds)
        self.bg.state = {"bit_generator": "MT19937", "state": {"key": key[1:], "pos": 624}}

    def unif_rand(self):
        v = float(self.bg.random_raw()) * 2.3283064365386963e-10
        eps = 2.328306437080797e-10
        if v <= 0.0:
            return 0.5 * eps
        if 1.0 - v <= 0.0:
            return 1.0 - 0.5 * eps
        return v

    def _rbits(self, bits):
        v = 0
        for _ in range(0, bits + 1, 16):
            v = 65536 * v + int(math.floor(self.unif_rand() * 65536))
        return v & ((1 << bits) - 1)

    def unif_index(self, dn):
        if dn <= 0:
            return 0
        bits = int(math.ceil(math.log2(dn)))
        while True:
            dv = self._rbits(bits)
            if dn > dv:
                return dv

    def sample_int(self, n, size=None):
        """sample.int(n, size) sin reemplazo (índices base 1)."""
        size = n if size is None else size
        if n > 1e7 and size <= n / 2:                      # R usa sample2() (muestreo con hash)
            vistos, salida = set(), []
            for _ in range(size):
                while True:
                    v = self.unif_index(n)
                    if v not in vistos:
                        vistos.add(v)
                        salida.append(v + 1)
                        break
            return salida
        x, salida = list(range(n)), []
        for _ in range(size):
            j = self.unif_index(n)
            salida.append(x[j] + 1)
            n -= 1
            x[j] = x[n]
        return salida

    def sample(self, valores, size=None):
        """sample(valores, size) sin reemplazo (valores de longitud > 1)."""
        valores = list(valores)
        return [valores[i - 1] for i in self.sample_int(len(valores), size)]


# --------------------------------------------------------------- Particiones caret
def create_data_partition(y, p, rng, groups=5):
    """caret::createDataPartition(): índices base 1 del conjunto de entrenamiento."""
    y = pd.Series(y).reset_index(drop=True)
    if pd.api.types.is_numeric_dtype(y) and not isinstance(y.dtype, pd.CategoricalDtype):
        cortes = np.unique(np.quantile(y.astype(float), np.linspace(0, 1, groups)))
        estratos = pd.cut(y.astype(float), bins=cortes, include_lowest=True, right=True)
    else:
        estratos = y.astype("category")
    indices = []
    for nivel in estratos.cat.categories:                  # dlply() recorre los niveles en orden
        miembros = list(np.where(estratos == nivel)[0] + 1)
        if len(miembros) == 1:
            indices += miembros
        elif len(miembros) > 1:
            indices += rng.sample(miembros, math.ceil(len(miembros) * p))
    return np.sort(np.array(indices))


def create_folds(y, k, rng):
    """caret::createFolds(list = FALSE) para una respuesta categórica: vector de fold (1..k)."""
    y = pd.Series(y).astype(str).reset_index(drop=True)
    niveles = sorted(y.unique())                            # factor(as.character(y)) ordena los niveles
    fold = np.zeros(len(y), dtype=int)
    for nivel in niveles:
        pos = np.where(y == nivel)[0]
        n_clase = len(pos)
        min_reps = n_clase // k
        if min_reps > 0:
            sobrantes = n_clase % k
            seq = list(range(1, k + 1)) * min_reps
            if sobrantes > 0:
                seq += rng.sample(range(1, k + 1), sobrantes)
            fold[pos] = rng.sample(seq)
        else:
            fold[pos] = rng.sample(range(1, k + 1), n_clase)
    return fold


def create_multi_folds(y, k, times, rng):
    """caret::createMultiFolds(): lista de pares (entrenamiento, validación) con índices base 0."""
    particiones = []
    for _ in range(times):
        fold = create_folds(y, k, rng)
        for f in range(1, k + 1):                          # split() ordena los folds 1..k
            particiones.append((np.where(fold != f)[0], np.where(fold == f)[0]))
    return particiones


def caret_repeatedcv_folds(y, number, repeats, seed):
    """Índices que usa caret::train(trControl = trainControl('repeatedcv')) tras set.seed(seed).

    train() extrae primero una semilla interna con sample.int(.Machine$integer.max, 1) y genera los
    folds con ella (withr::with_seed).
    """
    semilla_interna = RRandom(seed).sample_int(2147483647, 1)[0]
    return create_multi_folds(y, number, repeats, RRandom(semilla_interna))


# ---------------------------------------------------------------- step() para glm
def _ajustar_glm(respuesta, terminos, datos):
    formula = f"{respuesta} ~ " + (" + ".join(terminos) if terminos else "1")
    return smf.glm(formula, data=datos, family=sm.families.Binomial()).fit()


def _criterio(modelo, k):
    """extractAIC() de R para glm: deviance + k * número de parámetros (equivale a AIC con k = 2)."""
    return modelo.deviance + k * (modelo.df_model + 1)


def step_glm(respuesta, datos, inicio, alcance, direccion, k):
    """Réplica de step() de R para regresión logística.

    inicio: lista de términos del modelo inicial; alcance: términos del modelo completo.
    direccion: 'both', 'forward' o 'backward'. Devuelve la lista de términos en el orden de R.
    """
    actuales = list(inicio)
    modelo = _ajustar_glm(respuesta, actuales, datos)
    valor = _criterio(modelo, k)
    while True:
        candidatos = []
        if direccion in ("both", "backward"):
            for t in actuales:
                terms = [x for x in actuales if x != t]
                candidatos.append((_criterio(_ajustar_glm(respuesta, terms, datos), k), terms))
        if direccion in ("both", "forward"):
            for t in alcance:
                if t not in actuales:
                    terms = actuales + [t]
                    candidatos.append((_criterio(_ajustar_glm(respuesta, terms, datos), k), terms))
        if not candidatos:
            break
        mejor_valor, mejores = min(candidatos, key=lambda c: c[0])
        if mejor_valor >= valor - 1e-7:                    # misma regla de parada que step()
            break
        actuales, valor = mejores, mejor_valor
    return actuales


# --------------------------------------------------------------------- glmnet
class Glmnet:
    """Regresión logística penalizada (elastic net) equivalente a glmnet(family = "binomial").

    Resuelve  min  -(1/N)·logverosimilitud + λ·[(1-α)/2·||β||² + α·||β||₁]
    con las variables estandarizadas internamente (standardize = TRUE) e intercepto sin penalizar,
    usando la misma secuencia de λ y las mismas reglas de parada de la trayectoria que glmnet.
    Los coeficientes se devuelven en la escala original de las variables.
    """

    FDEV, DEVMAX, MNLAM = 1e-5, 0.999, 5                   # glmnet.control() por defecto

    def __init__(self, alpha=1.0, nlambda=100, lambda_min_ratio=None, lambdas=None, tol=1e-13):
        self.alpha, self.nlambda, self.lambda_min_ratio = alpha, nlambda, lambda_min_ratio
        self.lambdas_usuario, self.tol = lambdas, tol

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        n, p = X.shape
        self.medias = X.mean(axis=0)
        self.desv = np.sqrt(((X - self.medias) ** 2).mean(axis=0))
        Z = (X - self.medias) / self.desv
        ybar = y.mean()
        dev_nula = -2 * n * (ybar * math.log(ybar) + (1 - ybar) * math.log(1 - ybar))

        if self.lambdas_usuario is None:
            g = np.abs(Z.T @ (y - ybar)) / n
            lam_max = g.max() / max(self.alpha, 1e-3)
            ratio = self.lambda_min_ratio or (1e-4 if n > p else 1e-2)
            lambdas = lam_max * ratio ** (np.arange(self.nlambda) / (self.nlambda - 1))
            truncar = True
        else:
            lambdas, truncar = np.asarray(self.lambdas_usuario, float), False

        b0, beta = math.log(ybar / (1 - ybar)), np.zeros(p)
        betas, b0s, devs = [], [], []
        for m, lam in enumerate(lambdas):
            if m == 0 and truncar:
                # glmnet resuelve el primer punto con un λ "infinito" (9.9e35) y lo etiqueta con λ_max
                b0, beta = self._resolver(Z, y, 9.9e35, b0, beta)
            else:
                b0, beta = self._resolver(Z, y, lam, b0, beta)
            eta = b0 + Z @ beta
            dev = 2 * np.sum(np.logaddexp(0, eta) - y * eta)
            betas.append(beta.copy()); b0s.append(b0); devs.append(1 - dev / dev_nula)
            if truncar and m + 1 >= self.MNLAM:
                if devs[-1] > self.DEVMAX or devs[-1] - devs[-2] < self.FDEV:
                    break
        k = len(betas)
        self.lambdas = lambdas[:k]
        B = np.array(betas).T                               # p × k en escala estandarizada
        self.beta = B / self.desv[:, None]                  # escala original
        self.a0 = np.array(b0s) - self.medias @ self.beta
        self.dev_ratio = np.array(devs)
        self.nzero = (self.beta != 0).sum(axis=0)
        return self

    def _resolver(self, Z, y, lam, b0, beta):
        """Newton proximal: aproximación cuadrática (IRLS) + descenso por coordenadas."""
        n, p = Z.shape
        a = self.alpha
        for _ in range(200):
            eta = b0 + Z @ beta
            prob = 1 / (1 + np.exp(-eta))
            w = np.clip(prob * (1 - prob), 1e-10, None)
            zt = eta + (y - prob) / w
            wn = w / n
            sw = wn.sum()
            # Eliminación del intercepto: centrado ponderado
            zc = Z - (wn @ Z) / sw
            rc = zt - (wn @ zt) / sw
            G = (zc * wn[:, None]).T @ zc
            c = (zc * wn[:, None]).T @ rc
            beta_old = beta.copy()
            b = beta.copy()
            diag = np.diag(G)
            for _ in range(10000):
                cambio = 0.0
                for j in range(p):
                    grad = c[j] - G[j] @ b + diag[j] * b[j]
                    nuevo = np.sign(grad) * max(abs(grad) - lam * a, 0.0) / (diag[j] + lam * (1 - a))
                    d = nuevo - b[j]
                    if d != 0.0:
                        b[j] = nuevo
                        cambio = max(cambio, abs(d))
                if cambio < self.tol:
                    break
            beta = b
            b0_old = b0
            b0 = ((wn @ zt) - (wn @ Z) @ beta) / sw
            if np.max(np.abs(beta - beta_old)) < 1e-12 and abs(b0 - b0_old) < 1e-12:
                break
        return b0, beta

    def coef_s(self, s):
        """Coeficientes en valores arbitrarios de λ, interpolando como glmnet:::lambda.interp()."""
        lam = self.lambdas
        s = np.atleast_1d(np.asarray(s, float))
        if len(lam) == 1:
            return np.repeat(self.a0, len(s)), np.repeat(self.beta, len(s), axis=1)
        k = len(lam)
        sfrac = (lam[0] - s) / (lam[0] - lam[k - 1])
        lnorm = (lam[0] - lam) / (lam[0] - lam[k - 1])
        sfrac = np.clip(sfrac, lnorm.min(), lnorm.max())
        coord = np.interp(sfrac, lnorm, np.arange(1, k + 1))
        izq, der = np.floor(coord).astype(int) - 1, np.ceil(coord).astype(int) - 1
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = (sfrac - lnorm[der]) / (lnorm[izq] - lnorm[der])
        frac[izq == der] = 1.0
        frac[np.abs(lnorm[izq] - lnorm[der]) < np.finfo(float).eps] = 1.0
        a0 = self.a0[izq] * frac + self.a0[der] * (1 - frac)
        beta = self.beta[:, izq] * frac + self.beta[:, der] * (1 - frac)
        return a0, beta

    def predict_proba_s(self, X, s):
        a0, beta = self.coef_s(s)
        return 1 / (1 + np.exp(-(a0 + _eta_por_perfil(X, beta))))

    def coef(self, indice):
        return self.a0[indice], self.beta[:, indice]

    def predict_proba(self, X, indice=None):
        eta = self.a0 + np.asarray(X, float) @ self.beta
        prob = 1 / (1 + np.exp(-eta))
        return prob if indice is None else prob[:, indice]


def auc(y, prob):
    """AUC de Mann-Whitney (empates = 1/2), equivalente a pROC::auc() y survival::concordance()."""
    y = np.asarray(y).astype(bool)
    rangos = pd.Series(prob).rank(method="average").to_numpy()
    n1, n0 = y.sum(), (~y).sum()
    return (rangos[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


class CVGlmnet:
    """Equivalente a cv.glmnet(type.measure = "auc", foldid = foldid) para family = "binomial"."""

    def __init__(self, alpha, foldid):
        self.alpha, self.foldid = alpha, np.asarray(foldid)

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.glmnet_fit = Glmnet(alpha=self.alpha).fit(X, y)
        self.lambdas = self.glmnet_fit.lambdas
        nfolds = self.foldid.max()
        cvraw = np.empty((nfolds, len(self.lambdas)))
        pesos = np.empty(nfolds)
        for f in range(1, nfolds + 1):
            val = self.foldid == f
            # Como cv.glmnet: cada fold calcula su propia secuencia de λ y las predicciones se
            # interpolan en los λ del modelo completo (alignment = "lambda", lambda.interp)
            modelo = Glmnet(alpha=self.alpha).fit(X[~val], y[~val])
            prob = modelo.predict_proba_s(X[val], self.lambdas)
            cvraw[f - 1] = [auc(y[val], prob[:, j]) for j in range(len(self.lambdas))]
            pesos[f - 1] = val.sum()
        self.cvm = (cvraw * pesos[:, None]).sum(axis=0) / pesos.sum()
        var = (((cvraw - self.cvm) ** 2) * pesos[:, None]).sum(axis=0) / pesos.sum()
        self.cvsd = np.sqrt(var / (nfolds - 1))
        self.cvlo, self.cvup = self.cvm - self.cvsd, self.cvm + self.cvsd
        self.nzero = self.glmnet_fit.nzero
        # getOptcv.glmnet (para AUC se maximiza)
        idmin = np.where(self.cvm >= self.cvm.max())[0]
        self.index_min = idmin[np.argmax(self.lambdas[idmin])]
        semin = self.cvm[self.index_min] - self.cvsd[self.index_min]
        id1se = np.where(self.cvm >= semin)[0]
        self.index_1se = id1se[np.argmax(self.lambdas[id1se])]
        self.lambda_min, self.lambda_1se = self.lambdas[self.index_min], self.lambdas[self.index_1se]
        return self


# ------------------------------------------------------------ Matriz de diseño
def model_matrix(datos, respuesta):
    """model.matrix(respuesta ~ ., datos)[, -1]: variables en el orden de las columnas y codificación
    de tratamiento (una indicadora por cada nivel distinto del de referencia), como en R."""
    columnas = {}
    for var in datos.columns:
        if var == respuesta:
            continue
        s = datos[var]
        if isinstance(s.dtype, pd.CategoricalDtype):
            for nivel in s.cat.categories[1:]:
                columnas[f"{var}{nivel}"] = (s == nivel).astype(float).to_numpy()
        else:
            columnas[var] = s.astype(float).to_numpy()
    return pd.DataFrame(columnas, index=datos.index)


def _eta_por_perfil(X, coef):
    """Producto X·coef calculado una sola vez por fila única de X.

    Las operaciones vectorizadas de NumPy pueden redondear distinto filas idénticas (diferencias en el
    último bit), lo que rompe empates que en R son exactos y altera métricas basadas en rangos (AUC)."""
    X = np.asarray(X, float)
    unicas, inversa = np.unique(X, axis=0, return_inverse=True)
    return (unicas @ coef)[inversa.ravel()]


# --------------------------------------------------------- glm.fit de R (IRLS)
class GlmLogitR:
    """Regresión logística ajustada con el mismo algoritmo IRLS que glm.fit() de R.

    Reproduce los valores iniciales (mustart = (y + 0.5) / 2), el criterio de convergencia
    |dev − dev_anterior| / (|dev| + 0.1) < 1e-8 y el máximo de 25 iteraciones. Las probabilidades
    ajustadas coinciden con las de R hasta la precisión de la máquina, lo que importa cuando se
    calculan métricas basadas en rangos (AUC) sobre probabilidades casi empatadas.
    """

    def __init__(self, formula, datos, epsilon=1e-8, maxit=25):
        self.modelo_sm = smf.glm(formula, data=datos, family=sm.families.Binomial())
        X, y = self.modelo_sm.exog, self.modelo_sm.endog
        mu = (y + 0.5) / 2
        eta = np.log(mu / (1 - mu))
        dev_old = self._devianza(y, mu)
        for self.iteraciones in range(1, maxit + 1):
            w = np.sqrt(mu * (1 - mu))
            z = eta + (y - mu) / (mu * (1 - mu))
            coef, *_ = np.linalg.lstsq(X * w[:, None], z * w, rcond=None)
            eta = X @ coef
            mu = 1 / (1 + np.exp(-eta))
            dev = self._devianza(y, mu)
            if abs(dev - dev_old) / (abs(dev) + 0.1) < epsilon:
                break
            dev_old = dev
        self.params, self.deviance = coef, dev

    @staticmethod
    def _devianza(y, mu):
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = np.where(y > 0, y * np.log(y / mu), 0.0)
            t0 = np.where(y < 1, (1 - y) * np.log((1 - y) / (1 - mu)), 0.0)
        return 2 * np.sum(t1 + t0)

    def predict(self, datos_nuevos):
        from patsy import build_design_matrices
        (X,) = build_design_matrices([self.modelo_sm.data.model_spec], datos_nuevos,
                                     return_type="matrix")
        return 1 / (1 + np.exp(-_eta_por_perfil(X, self.params)))
