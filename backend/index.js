const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const sql = require('mssql');
const axios = require('axios'); // para llamar al servicio Python

const app = express();
app.use(cors());
app.use(bodyParser.json());

// Configuración SQL Server
const config = {
    server: 'capstone-bd-sqlserver.database.windows.net',
    database: 'rpcapstone',
    user: 'admincapstone',
    password: 'korosensei6979@',
    options: {
        encrypt: true,
        trustServerCertificate: false
    }
};

let pool;

async function startServer() {
    try {
        pool = await sql.connect(config);
        console.log('✅ Conectado a SQL Server');

        // Endpoint principal
        app.post('/respuestas', async (req, res) => {
            const { genero, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 } = req.body;

            try {
                // 1️⃣ Guardar datos crudos en SQL Server
                const result = await pool.request()
                    .input('genero', sql.TinyInt, genero)
                    .input('p1', sql.TinyInt, p1)
                    .input('p2', sql.TinyInt, p2)
                    .input('p3', sql.TinyInt, p3)
                    .input('p4', sql.TinyInt, p4)
                    .input('p5', sql.TinyInt, p5)
                    .input('p6', sql.TinyInt, p6)
                    .input('p7', sql.TinyInt, p7)
                    .input('p8', sql.TinyInt, p8)
                    .input('p9', sql.TinyInt, p9)
                    .input('p10', sql.TinyInt, p10)
                    .query(`
                        INSERT INTO respuestas_pss10 
                        (genero, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, fecha)
                        VALUES (@genero, @p1, @p2, @p3, @p4, @p5, @p6, @p7, @p8, @p9, @p10, GETDATE())
                    `);

                console.log('✅ Respuestas guardadas correctamente');

                // 2️⃣ Enviar al servicio de predicción en Azure (Flask)
                const pythonResponse = await axios.post(
                    'https://ml-capst-api-e5a3hrhqbhhbc7gr.brazilsouth-01.azurewebsites.net/predict',
                    { genero, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 }
                );

                // 3️⃣ Devolver al frontend el resultado del modelo
                return res.json({
                    message: '✅ Respuestas guardadas y analizadas correctamente',
                    prediccion: pythonResponse.data.prediccion,
                    probabilidad: pythonResponse.data.probabilidad
                });

            } catch (err) {
                console.error('❌ Error en /respuestas:', err.message);
                res.status(500).json({ error: 'Error al procesar las respuestas', detalle: err.message });
            }
        });

        // Iniciar servidor
        const PORT = process.env.PORT || 3000;
        app.listen(PORT, () => console.log(`🚀 Servidor corriendo en puerto ${PORT}`));

    } catch (err) {
        console.error('❌ No se pudo conectar a SQL Server:', err);
    }
}

startServer();
