🎓 Capstone Project: Sistema de Clasificación de Estrés (PSS-10) con Machine Learning
Este proyecto integral tiene como objetivo identificar y clasificar los niveles de estrés (Eustrés y Distrés) en estudiantes mediante el Test PSS-10. La solución combina una plataforma web para la toma de datos y un motor de Inteligencia Artificial basado en múltiples modelos de clasificación.

🏗️ Arquitectura del Sistema
El repositorio está organizado en tres componentes fundamentales:

1. 🌐 FRONTEND (/frontend)
Tecnologías: HTML5, CSS3, JavaScript.

Descripción: Interfaz de usuario donde se desarrolla el Test de Estrés. Permite una captura de datos fluida y el envío de respuestas al servidor de forma asíncrona.

2. ⚙️ BACKEND (/backend)
Tecnologías: Node.js, Express.

Nube: Microsoft Azure SQL Database.

Descripción: API REST que gestiona la lógica de negocio, recibe las respuestas del frontend y garantiza la persistencia de la información en una base de datos en la nube.

3. 🤖 MACHINE LEARNING (/Entrenamiento_de_los_12_modelos_ML_y_Registro_PSS10)
Esta es la carpeta núcleo del proyecto de investigación, donde se implementa la inteligencia del sistema.

Tecnologías: Python, Scikit-Learn, Pandas.

Análisis: Implementación y evaluación comparativa de 12 modelos de Machine Learning para encontrar el algoritmo con mayor precisión predictiva.

Archivos clave:

main.py: Script principal de ejecución y entrenamiento.

Registros.csv: Dataset base utilizado para el entrenamiento.

requirements.txt: Dependencias necesarias para el entorno.

LEEME - PASOS.txt: Documentación técnica para la puesta en marcha de los modelos.

🛠️ Stack Tecnológico Completo
Desarrollo Web: JavaScript (Node.js/Express)

Ciencia de Datos: Python (Scikit-Learn)

Infraestructura Cloud: Microsoft Azure

Control de Versiones: Git / GitHub
