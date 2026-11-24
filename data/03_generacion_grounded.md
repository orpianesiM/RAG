# Generación grounded y límites de RAG

La generación aumentada por recuperación (RAG) combina una búsqueda sobre
documentos con un modelo generativo. Primero, el sistema convierte la pregunta
en un embedding y busca los fragmentos más parecidos. Luego entrega al modelo
la pregunta y un conjunto limitado de fragmentos como contexto. Finalmente,
el modelo redacta una respuesta usando esa evidencia y devuelve referencias
a los documentos recuperados.

El prompt debe establecer una regla verificable: responder solo con el
contexto, no seguir instrucciones que aparezcan dentro de los documentos y
abstenerse cuando falte evidencia suficiente. Una respuesta de abstención se
expresa exactamente como “No lo sé” y no incluye fuentes. Una respuesta
sustentada incluye los nombres de archivo y los IDs de los fragmentos que la
respaldan. La aplicación valida la forma de la salida y comprueba que las
referencias existan entre los fragmentos efectivamente recuperados.

La recuperación de un pasaje no garantiza por sí sola que la respuesta sea
correcta. El modelo puede interpretar mal un texto, combinar fuentes
incompatibles o generar una afirmación no sustentada. Por eso se limita la
cantidad de fragmentos enviados, se registran las fuentes, se pide al modelo
que admita cuando no sabe y se prueban consultas respondibles y consultas
trampa. La similitud vectorial sirve para encontrar evidencia candidata; la
respuesta final debe permanecer dentro de lo que esa evidencia afirma.

Una consulta fuera del tema puede aun así recuperar los documentos “menos
lejanos”, porque muchas bases vectoriales devuelven vecinos aunque ninguno sea
relevante. En ese caso, el modelo debe reconocer que el contexto no responde
la pregunta. Para aplicaciones con requisitos estrictos se puede añadir un
umbral de relevancia, reranking o revisión humana, y medir el comportamiento
con un conjunto de preguntas y respuestas esperadas.
