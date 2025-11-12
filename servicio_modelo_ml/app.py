from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

# Cargar modelo y scaler entrenados
MODELO = joblib.load('modelo_reglog.joblib')
SCALER = joblib.load('scaler.joblib')

# Orden de columnas esperado por el modelo
FEATURES = ['genero','p1','p2','p3','p4','p5','p6','p7','p8','p9','p10']

# Preguntas que se deben invertir
PREGUNTAS_INVERTIDAS = ['p4','p5','p7','p8']

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Convertir a float
        data_processed = {k: float(v) for k, v in data.items()}

        # Validar que las preguntas estén dentro del rango 0-4
        for p in [f'p{i}' for i in range(1,11)]:
            if p not in data_processed or not (0 <= data_processed[p] <= 4):
                return jsonify({
                    "prediccion": "Inválido",
                    "mensaje": f"Pregunta {p} fuera de rango (0-4) o ausente."
                }), 400

        # Validar que no todas las respuestas sean iguales
        valores = [data_processed[f'p{i}'] for i in range(1,11)]
        if len(set(valores)) == 1:
            return jsonify({
                "prediccion": "Inválido",
                "mensaje": "Todas las respuestas tienen el mismo valor. Test no confiable."
            }), 400

        # Invertir ítems según diseño del test
        for p in PREGUNTAS_INVERTIDAS:
            data_processed[p] = 4 - data_processed[p]

        # Crear DataFrame en el orden correcto
        df_pred = pd.DataFrame([data_processed], columns=FEATURES)

        # Escalar solo las preguntas (no genero)
        X_numeric = df_pred.drop(columns=['genero'])
        X_scaled_vals = SCALER.transform(X_numeric)
        X_scaled_df = pd.DataFrame(X_scaled_vals, columns=X_numeric.columns, index=df_pred.index)

        # Combinar con genero (no influye)
        X_proc = pd.concat([df_pred['genero'].to_frame(), X_scaled_df], axis=1)

        # Predicción y probabilidad
        pred_binaria = MODELO.predict(X_proc)[0]

        # Asegurarse que predict_proba exista
        if hasattr(MODELO, "predict_proba"):
            probabilidad = MODELO.predict_proba(X_proc)[0][1]
        else:
            # Para SVM sin probabilidad, usar decision_function
            probabilidad = (MODELO.decision_function(X_proc)[0] + 1) / 2  # Normaliza a 0-1

        # Aplicar margen mínimo/máximo para estabilidad (opcional)
        margen = 0.05
        probabilidad = max(min(probabilidad, 1-margen), margen)

        resultado = "Distrés" if pred_binaria == 1 else "Eustrés"

        return jsonify({
            "prediccion": resultado,
            "probabilidad": round(float(probabilidad)*100, 2)  # Porcentaje estable
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
