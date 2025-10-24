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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Métricas
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)


""""
# -------------------------
# Conexion a la base de datos en Azure SQL Database
# -------------------------
load_dotenv()

server = os.getenv('DB_SERVER')
database = os.getenv('DB_DATABASE')
username = os.getenv('DB_USER')
password = os.getenv('DB_PASSWORD')
driver = '{ODBC Driver 18 for SQL Server}'

if not all([server, database, username, password]):
    print("❌ ERROR: Faltan credenciales en el archivo .env. Revisa el archivo.")
    exit()

connection_string = f"DRIVER={driver};SERVER=tcp:{server},1433;DATABASE={database};UID={username};PWD={password};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"

try:
    print("Conectando a la base de datos en Azure...")
    cnxn = pyodbc.connect(connection_string)
    query = "SELECT * FROM respuestas_pss10"
    df = pd.read_sql(query, cnxn)
    cnxn.close()
    print(f"✅ PASO 2: ¡Conexión exitosa! Se cargaron {len(df)} registros.")
except Exception as e:
    print(f"❌ ERROR al conectar a la base de datos: {e}")
    exit()

"""

# -------------------------
# MODO LOCAL 
# -------------------------
csv_path = os.path.join(os.path.dirname(__file__), "Registros.csv")
print("📂 Cargando datos desde archivo CSV local...")
df = pd.read_csv(csv_path, encoding='utf-8')
print(f"✅ Se cargaron {len(df)} registros desde CSV.")

# -------------------------
# LIMPIEZA DE DATOS 
# -------------------------
print("\n⚙️ Generando DataFrame limpio para entrenamiento (sin tocar BD)...")

df_clean = df.copy()

cols_preguntas = ['p1','p2','p3','p4','p5','p6','p7','p8','p9','p10']

#Eliminar registros con valores fuera de rango o nulos
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
#TRANSFORMACIÓN (invertir ítems y crear etiqueta)
# -------------------------
print("\n🔄 Invirtiendo ítems correctos (P4, P5, P7, P8) y calculando score/label...")

items_invertidos = ['p4', 'p5', 'p7', 'p8']
df_clean[items_invertidos] = 4 - df_clean[items_invertidos]

df_clean["score_total"] = df_clean[cols_preguntas].sum(axis=1)

def clasificar_estres(score):
    return 0 if score <= 20 else 1

df_clean["label"] = df_clean["score_total"].apply(clasificar_estres)

print("✅ Etiquetas generadas (conteo):")
print(df_clean["label"].value_counts().to_string())

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
# PASO 6: DEFINIR MODELOS (6 MODELOS CLAVE)
# -------------------------
models = {
    "Naive Bayes": GaussianNB(),
    "LDA": LDA(),
    "K-NN": KNeighborsClassifier(n_neighbors=5),
    "AdaBoost": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1, random_state=0),
        random_state=0, n_estimators=100
    ),
    "XGBoost": XGBClassifier(random_state=0, eval_metric='logloss'),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=0, n_jobs=-1
    )
}

# -------------------------
# FUNCIONES AUXILIARES
# -------------------------
def safe_roc_auc(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score)
    except Exception:
        return np.nan

def evaluate_model_on_test(name, model, X_test_loc, y_test_loc):
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
        plt.plot(fpr, tpr, label=f"AUC={safe_roc_auc(y_t,y_proba):.3f}")
        plt.plot([0,1],[0,1],'--', color='gray')
        plt.title(f"ROC - {model_name}"); plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.legend(loc="lower right"); plt.tight_layout()
        plt.savefig(f"results/graficos/{model_name}_roc.png")
        plt.close()
    except Exception:
        pass

def plot_learning_curve_generic(estimator, X_loc, y_loc, model_name):
    try:
        train_sizes, train_scores, test_scores = learning_curve(estimator, X_loc, y_loc, cv=5, scoring="accuracy")
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
        plt.figure(figsize=(6,4))
        plt.bar(range(len(imp)), imp[idx])
        plt.xticks(range(len(imp)), cols[idx], rotation=90)
        plt.title(f"Feature importance - {model_name}")
        plt.tight_layout()
        plt.savefig(f"results/graficos/{model_name}_features.png")
        plt.close()

def plot_metrics_bar(name, test_df_row):
    metrics = ["Accuracy","Recall","F1-Score","ROC_AUC"]
    values = [test_df_row[m].values[0] for m in metrics]
    plt.figure(figsize=(4,3))
    sns.barplot(x=metrics, y=values)
    plt.ylim(0,1)
    plt.title(f"Métricas - {name}")
    plt.tight_layout()
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
        if sample_w is not None:
            model.fit(X_train_proc, y_train, sample_weight=sample_w)
        else:
            model.fit(X_train_proc, y_train)
    except TypeError:
        model.fit(X_train_proc, y_train)

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
