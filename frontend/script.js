const preguntas = [
    { texto: "P1: ¿Con qué frecuencia se ha sentido estresado la última semana?", invertida: false },
    { texto: "P2: ¿Con qué frecuencia ha sentido que no puede controlar lo importante en su vida?", invertida: false },
    { texto: "P3: ¿Con qué frecuencia se ha sentido nervioso o estresado?", invertida: false },
    { texto: "P4: ¿Con qué frecuencia ha sentido que no puede afrontar todas las cosas que debe hacer?", invertida: true },
    { texto: "P5: ¿Con qué frecuencia se ha sentido que las cosas le superan?", invertida: true },
    { texto: "P6: ¿Con qué frecuencia ha sentido que no puede controlar lo que le pasa?", invertida: false },
    { texto: "P7: ¿Con qué frecuencia ha sentido que todo le sale mal?", invertida: true },
    { texto: "P8: ¿Con qué frecuencia se ha sentido agobiado?", invertida: true },
    { texto: "P9: ¿Con qué frecuencia se ha sentido incapaz de hacer frente a las dificultades?", invertida: false },
    { texto: "P10: ¿Con qué frecuencia se ha sentido bajo presión?", invertida: false }
];

const respuestasTexto = ["Nunca", "Casi nunca", "Algunas veces", "Casi siempre", "Siempre"];
const preguntasDiv = document.getElementById('preguntas');

// Generar las preguntas dinámicamente
preguntas.forEach((preg, i) => {
    const div = document.createElement('div');
    div.className = 'pregunta';

    const label = document.createElement('label');
    label.textContent = preg.texto;
    div.appendChild(label);

    const radioGroup = document.createElement('div');
    radioGroup.className = 'radio-group';

    const valores = preg.invertida ? [4,3,2,1,0] : [0,1,2,3,4];

    respuestasTexto.forEach((text, j) => {
        const input = document.createElement('input');
        input.type = 'radio';
        input.name = `p${i+1}`;
        input.value = valores[j];
        input.id = `p${i+1}_${j}`;

        const inputLabel = document.createElement('label');
        inputLabel.htmlFor = input.id;
        inputLabel.textContent = text;

        radioGroup.appendChild(input);
        radioGroup.appendChild(inputLabel);
    });

    div.appendChild(radioGroup);
    preguntasDiv.appendChild(div);
});

// Enviar respuestas al backend SQL Server
document.getElementById('formEncuesta').addEventListener('submit', async (e) => {
    e.preventDefault();
    const data = { genero: parseInt(document.getElementById('genero').value) };

    for (let i = 1; i <= 10; i++) {
        const selected = document.querySelector(`input[name="p${i}"]:checked`);
        if (!selected) {
            alert(`Por favor responde la pregunta ${i}`);
            return;
        }
        data[`p${i}`] = parseInt(selected.value);
    }

    try {
        const res = await fetch('https://capstone-backend-vins.azurewebsites.net/respuestas', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        const result = await res.json();
        alert(result.message); // tu backend devuelve rowsAffected, no id
        document.getElementById('formEncuesta').reset();
    } catch (err) {
        console.error(err);
        alert('Error al enviar la encuesta');
    }
});
