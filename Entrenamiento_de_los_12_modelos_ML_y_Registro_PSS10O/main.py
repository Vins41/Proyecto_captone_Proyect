import os
import pandas as pd
import pyodbc
from dotenv import load_dotenv

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_sample_weight
from sklearn.model_selection import learning_curve

# Modelos
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    AdaBoostClassifier, RandomForestClassifier,
    GradientBoostingClassifier, ExtraTreesClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Métricas
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

import joblib # Importamos joblib aquí para que esté disponible para guardar

# Nombres de archivos para carga local
CSV_PATH_LOCAL = "Registros.csv"
CSV_PATH_PROCESSED = "data_processed/pss10_clean_labeled.csv"
cols_preguntas = ['p1','p2','p3','p4','p5','p6','p7','p8','p9','p10']

# -------------------------
# FUNCIÓN DE CARGA DE DATOS (Con Fallback a CSV Local)
# -------------------------

def load_data_with_fallback():
    """Intenta cargar datos desde Azure SQL; si falla, carga el CSV local."""
    load_dotenv()
    df = None

    # Intentar conexión a Azure SQL (Descomentar para uso remoto)
    """
    server = os.getenv('DB_SERVER')
    database = os.getenv('DB_DATABASE')
    username = os.getenv('DB_USER')
    password = os.getenv('DB_PASSWORD')
    driver = '{ODBC Driver 18 for SQL Server}'
    
    if all([server, database, username, password]):
        connection_string = f"DRIVER={driver};SERVER=tcp:{server},1433;DATABASE={database};UID={username};PWD={password};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=10;"
        query = "SELECT * FROM respuestas_pss10"
        
        try:
            print("Conectando a Azure SQL para obtener datos...")
            cnxn = pyodbc.connect(connection_string)
            df = pd.read_sql(query, cnxn)
            cnxn.close()
            print(f"✅ Conexión exitosa. Se cargaron {len(df)} registros desde Azure.")
            return df
        except Exception as e:
            print(f"⚠️ ADVERTENCIA: Falló la conexión a la base de datos ({e}).")
            print(f"       Intentando cargar datos desde el archivo CSV local de respaldo...")
    else:
        print("⚠️ ADVERTENCIA: Faltan credenciales en .env o conexión comentada. Intentando cargar CSV local.")
    """

    # FALLBACK: Cargar desde CSV local (Usando os.path.join para rutas seguras)
    try:
        # Asegurarse de que el CSV_PATH_LOCAL apunte a la ubicación correcta
        csv_full_path = os.path.join(os.path.dirname(__file__), CSV_PATH_LOCAL)
        df = pd.read_csv(csv_full_path, encoding='utf-8')
        print(f"✅ Datos cargados exitosamente desde: {CSV_PATH_LOCAL}. Registros: {len(df)}.")
        return df
    except FileNotFoundError:
        print(f"❌ ERROR CRÍTICO: No se pudo conectar a Azure ni encontrar el archivo local: {CSV_PATH_LOCAL}.")
        print("   Asegúrese de que el archivo 'Registros.csv' exista en el directorio del script.")
        return None

# --- Inicio de la Ejecución Principal ---
df = load_data_with_fallback()
if df is None:
    exit() # Termina si no se pudieron cargar los datos

# -------------------------
# LIMPIEZA DE DATOS 
# -------------------------
print("\n⚙️ Generando DataFrame limpio para entrenamiento (sin tocar BD)...")

df_clean = df.copy()

# Eliminar registros con valores fuera de rango o nulos
cond_valid = df_clean[cols_preguntas].notnull().all(axis=1)
for c in cols_preguntas:
    cond_valid &= df_clean[c].between(0,4)
df_clean = df_clean[cond_valid]

# Eliminar registros donde todas las respuestas son iguales
df_clean = df_clean[~df_clean[cols_preguntas].apply(lambda row: len(set(row)) == 1, axis=1)]

# Eliminar duplicados
df_clean = df_clean.drop_duplicates(subset=cols_preguntas).reset_index(drop=True)

print(f"✅ DataFrame limpio generado: {len(df_clean)} registros válidos.")

# -------------------------
# TRANSFORMACIÓN (invertir ítems y crear etiqueta)
# -------------------------
print("\n🔄 Invirtiendo ítems correctos (P4, P5, P7, P8) y calculando score/label...")

items_invertidos = ['p4', 'p5', 'p7', 'p8']
df_clean[items_invertidos] = 4 - df_clean[items_invertidos]

df_clean["score_total"] = df_clean[cols_preguntas].sum(axis=1)

def clasificar_estres(score):
    # 0 = Eustrés (score <= 20), 1 = Distrés (score > 20)
    return 0 if score <= 20 else 1

df_clean["label"] = df_clean["score_total"].apply(clasificar_estres)

print("✅ Etiquetas generadas (conteo):")
print(df_clean["label"].value_counts().to_string())


# -------------------------
# GUARDAR DATASET PREPROCESADO
# -------------------------
os.makedirs("data_processed", exist_ok=True)
# Se guarda el dataframe ya limpio, invertido y con las etiquetas de Eustrés/Distrés
df_clean.to_csv(CSV_PATH_PROCESSED, index=False)
print(f"✅ Dataset limpio y etiquetado guardado en: {CSV_PATH_PROCESSED}")

# -------------------------
# ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# -------------------------
# ... (Bloque de EDA es idéntico al original)
os.makedirs("results/graficos", exist_ok=True)
print("\n🔍 Generando Gráficos EDA para Eustrés (0) y Distrés (1)...")

# 1. Histograma del Score Total con umbral
plt.figure(figsize=(7, 5))
sns.histplot(df_clean['score_total'], kde=True, bins=20, color='skyblue')
plt.axvline(x=20, color='red', linestyle='--', linewidth=2, label='Umbral de Distrés (Score=20)')
plt.title('Distribución del Score Total PSS-10')
plt.xlabel('Score Total')
plt.ylabel('Frecuencia')
plt.legend()
plt.tight_layout()
plt.savefig("results/graficos/EDA_01_score_distribution.png")
plt.close()

# 2. Boxplots de Preguntas vs. Etiqueta (para identificar discriminadores)
plt.figure(figsize=(15, 8))
for i, col in enumerate(cols_preguntas):
    plt.subplot(2, 5, i + 1)
    sns.boxplot(x='label', y=col, data=df_clean, palette=['#66c2a5', '#fc8d62'])
    plt.title(f'Distribución {col}', fontsize=10)
    plt.xlabel('Label (0/1)', fontsize=8)
    plt.ylabel(f'{col} (Respuesta)', fontsize=8)
plt.suptitle('Boxplots de Respuestas por Pregunta vs. Etiqueta (0=Eustrés, 1=Distrés)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("results/graficos/EDA_02_questions_vs_label_boxplots.png")
plt.close()

# 3. Conteo de Etiquetas por Género
plt.figure(figsize=(6, 5))
sns.countplot(x='genero', hue='label', data=df_clean, palette=['#8da0cb', '#e78ac3'])
plt.title('Distribución de Etiquetas (Eustrés/Distrés) por Género')
plt.xlabel('Género (0=Hombre, 1=Mujer)')
plt.ylabel('Conteo de Registros')
plt.legend(title='Label', labels=['Eustrés (0)', 'Distrés (1)'])
plt.tight_layout()
plt.savefig("results/graficos/EDA_03_gender_vs_label_count.png")
plt.close()

# 4. Mapa de Calor de Correlación
df_corr = df_clean[['score_total'] + cols_preguntas + ['label', 'genero']].copy()
plt.figure(figsize=(10, 8))
sns.heatmap(df_corr.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, linecolor='black')
plt.title('Matriz de Correlación entre Features y Label')
plt.tight_layout()
plt.savefig("results/graficos/EDA_04_correlation_heatmap.png")
plt.close()

print("✅ Gráficos EDA generados y guardados en 'results/graficos/'.")

# -------------------------
# PREPARACION FEATURES, SPLIT, ESCALADO
# -------------------------
print("\n⚙️ Preparando datos para ML, split 70/30 y escalado...")

features = ['genero'] + cols_preguntas
X = df_clean[features].copy()
y = df_clean['label'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

scaler = StandardScaler()
X_train_num = X_train[cols_preguntas]
X_test_num = X_test[cols_preguntas]
scaler.fit(X_train_num)

X_train_scaled_num = pd.DataFrame(scaler.transform(X_train_num), index=X_train.index, columns=cols_preguntas)
X_test_scaled_num = pd.DataFrame(scaler.transform(X_test_num), index=X_test.index, columns=cols_preguntas)

X_train_proc = pd.concat([X_train['genero'].to_frame(), X_train_scaled_num], axis=1)
X_test_proc = pd.concat([X_test['genero'].to_frame(), X_test_scaled_num], axis=1)

print(f"✅ Train: {len(X_train_proc)} rows. Test: {len(X_test_proc)} rows.")

# -------------------------
# DEFINIR LOS 12 MODELOS
# -------------------------
models = {
    "Naive Bayes": GaussianNB(),
    "Linear Discriminant Analysis (LDA)": LDA(),
    "Quadratic Discriminant Analysis (QDA)": QDA(),
    "K-Vecinos más Cercanos (K-NN)": KNeighborsClassifier(n_neighbors=5),
    "Regresión Logística": LogisticRegression(max_iter=1000, random_state=42),
    "Support Vector Machine (SVM)": SVC(probability=True, random_state=42),
    "Árbol de Decisión": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "AdaBoost": AdaBoostClassifier(random_state=42, n_estimators=100),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Extra Trees": ExtraTreesClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss')
}

# -------------------------
# FUNCIONES AUXILIARES (Sin Cambios)
# -------------------------
# (Las funciones de métricas, gráficos, etc. son idénticas al original)

def safe_roc_auc(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score)
    except Exception:
        return np.nan

def evaluate_model_on_test(name, model, X_test_loc, y_test_loc):
    # ... (idéntico a tu original)
    y_pred = model.predict(X_test_loc)
    acc = accuracy_score(y_test_loc, y_pred)
    rec = recall_score(y_test_loc, y_pred, zero_division=0)
    f1v = f1_score(y_test_loc, y_pred, zero_division=0)

    roc_val = np.nan
    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test_loc)[:, 1]
            roc_val = safe_roc_auc(y_test_loc, y_proba)
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test_loc)
            roc_val = safe_roc_auc(y_test_loc, y_score)
    except Exception:
        roc_val = np.nan

    return {"Modelo": name, "Accuracy": acc, "Recall": rec, "F1-Score": f1v, "ROC_AUC": roc_val, "y_pred": y_pred}

os.makedirs("results/graficos", exist_ok=True)
os.makedirs("results", exist_ok=True)

def plot_confusion(cm, model_name):
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Matriz - {model_name}")
    plt.xlabel("Predicho"); plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(f"results/graficos/{model_name}_confusion.png")
    plt.close()

def plot_roc_curve(model, X_t, y_t, model_name):
    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_t)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_t)
        else:
            return
        fpr, tpr, _ = roc_curve(y_t, y_proba)
        plt.figure(figsize=(4,3))
        plt.plot(fpr, tpr, label=f"AUC={safe_roc_auc(y_t,y_proba):.4f}")
        plt.plot([0,1],[0,1],'--', color='gray')
        plt.title(f"ROC - {model_name}"); plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.legend(loc="lower right"); plt.tight_layout()
        plt.savefig(f"results/graficos/{model_name}_roc.png")
        plt.close()
    except Exception:
        pass

def plot_learning_curve_generic(estimator, X_loc, y_loc, model_name):
    try:
        train_sizes, train_scores, test_scores = learning_curve(estimator, X_loc, y_loc, cv=5, scoring="accuracy", n_jobs=-1)
        train_mean = np.mean(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        plt.figure(figsize=(4,3))
        plt.plot(train_sizes, train_mean, marker='o', label='Train')
        plt.plot(train_sizes, test_mean, marker='s', label='Val')
        plt.title(f"Learning curve - {model_name}")
        plt.xlabel("Train size"); plt.ylabel("Accuracy"); plt.legend()
        plt.tight_layout()
        plt.savefig(f"results/graficos/{model_name}_learning.png")
        plt.close()
    except Exception:
        pass

def plot_feature_importance_if_exists(model, X_loc, model_name):
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        cols = X_loc.columns
        idx = np.argsort(imp)[::-1]
        
        # Filtrar solo características con importancia > 0 si hay más de 5
        if len(imp) > 5:
            imp_sorted = imp[idx]
            cols_sorted = cols[idx]
            valid_idx = imp_sorted > 0
            imp_sorted = imp_sorted[valid_idx][:10] # Mostrar top 10 si hay muchos
            cols_sorted = cols_sorted[valid_idx][:10]
            
            # Reordenar los índices para el gráfico
            idx_plot = np.argsort(imp_sorted)[::-1]
            imp_plot = imp_sorted[idx_plot]
            cols_plot = cols_sorted[idx_plot]
            
            if len(imp_plot) == 0:
                return # No hay features importantes
        else:
            imp_plot = imp[idx]
            cols_plot = cols[idx]

        plt.figure(figsize=(6, 4))
        sns.barplot(x=imp_plot, y=cols_plot)
        plt.title(f"Feature importance - {model_name}")
        plt.tight_layout()
        plt.savefig(f"results/graficos/{model_name}_features.png")
        plt.close()
    # Para modelos lineales como Regresión Logística
    elif hasattr(model, "coef_"):
        try:
            imp = model.coef_[0]
            cols = X_loc.columns
            
            # Convertir a valores absolutos para mostrar importancia
            abs_imp = np.abs(imp)
            idx = np.argsort(abs_imp)[::-1]
            
            plt.figure(figsize=(6, 4))
            sns.barplot(x=abs_imp[idx], y=cols[idx])
            plt.title(f"Coeficientes Absolutos - {model_name}")
            plt.tight_layout()
            plt.savefig(f"results/graficos/{model_name}_features_coef.png")
            plt.close()
        except Exception:
            pass


def plot_metrics_bar(name, test_df_row):
    metrics = ["Accuracy","Recall","F1-Score","ROC_AUC"]
    values = [test_df_row[m].values[0] for m in metrics]
    plt.figure(figsize=(4,3))
    sns.barplot(x=metrics, y=values, palette='viridis')
    plt.ylim(0.80,1.00)
    plt.title(f"Métricas - {name}"); plt.tight_layout()
    plt.savefig(f"results/graficos/{name}_metrics.png")
    plt.close()


# -------------------------
# ENTRENAMIENTO, EVALUACIÓN Y GRÁFICOS
# -------------------------
print("\n⚙️ PASO 7: Entrenando modelos y evaluando en conjunto de prueba (split 70/30)...")

test_results = []
for name, model in models.items():
    print(f"\n--- Entrenando {name} ---")
    try:
        sample_w = compute_sample_weight(class_weight='balanced', y=y_train)
    except Exception:
        sample_w = None

    try:
        if sample_w is not None and name not in ["XGBoost", "Gradient Boosting"]:
            model.fit(X_train_proc, y_train, sample_weight=sample_w)
        else:
            model.fit(X_train_proc, y_train)
    except TypeError:
        model.fit(X_train_proc, y_train)
    except Exception as e:
        print(f"Error al entrenar {name}: {e}")
        continue


    res = evaluate_model_on_test(name, model, X_test_proc, y_test)
    test_results.append(res)

    cm = confusion_matrix(y_test, res["y_pred"])
    plot_confusion(cm, name)
    plot_roc_curve(model, X_test_proc, y_test, name)
    plot_learning_curve_generic(model, pd.concat([X_train_proc, X_test_proc]), pd.concat([y_train, y_test]), name)
    plot_feature_importance_if_exists(model, X_train_proc, name)
    plot_metrics_bar(name, pd.DataFrame([res]))

test_df = pd.DataFrame(test_results).sort_values("F1-Score", ascending=False).reset_index(drop=True)
print("\n--- RESULTADOS EN TEST (Split 70/30) ---")
print(test_df[["Modelo","Accuracy","Recall","F1-Score","ROC_AUC"]].round(4))
test_df.to_csv("results/resultados_test_split.csv", index=False)

# -------------------------
# RESUMEN FINAL
# -------------------------
totales = df_clean['label'].value_counts().rename_axis('label').reset_index(name='count')
totales_map = {'Eustres (0)': int((df_clean['label']==0).sum()), 'Distres (1)': int((df_clean['label']==1).sum())}
por_genero = df_clean.groupby(['genero','label']).size().unstack(fill_value=0)
por_genero = por_genero.rename(columns={0:'Eustres',1:'Distres'})

print("\n\n RESUMEN FINAL PARA INFORME")
print("---------------------------")
print(f"Total registros válidos: {len(df_clean)}")
print(f"Total Eustrés (label=0): {totales_map['Eustres (0)']}")
print(f"Total Distrés (label=1): {totales_map['Distres (1)']}")
print("\nDistribución por género (0=hombre,1=mujer):")
print(por_genero)

por_genero.to_csv("results/resumen_por_genero.csv")
totales.to_csv("results/resumen_totales.csv", index=False)

print("\n✅ PROCESO COMPLETADO. Resultados guardados en 'results/' y gráficos en 'results/graficos/'.")


# -------------------------

# -------------------------
# GUARDAR MODELOS ENTRENADOS (Regresión Logística y SVM)
# -------------------------
os.makedirs("modelos", exist_ok=True)

# Buscar los modelos específicos dentro del diccionario
modelo_log = models.get("Regresión Logística")
modelo_svm = models.get("Support Vector Machine (SVM)")

if modelo_log:
    joblib.dump(modelo_log, "modelos/modelo_reglog.joblib")
    print("   - modelos/modelo_reglog.joblib")
else:
    print("   - Advertencia: Regresión Logística no fue encontrado/entrenado.")

if modelo_svm:
    joblib.dump(modelo_svm, "modelos/modelo_svm.joblib")
    print("   - modelos/modelo_svm.joblib")
else:
    print("   - Advertencia: SVM no fue encontrado/entrenado.")

# Guardar el scaler usado para transformar los datos
joblib.dump(scaler, "modelos/scaler.joblib")

print("   - modelos/scaler.joblib")
print("✅ Modelos y scaler guardados correctamente en la carpeta 'modelos/'.")