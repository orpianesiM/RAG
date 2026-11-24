# Embeddings y búsqueda semántica

Un embedding es una representación numérica de un texto en forma de vector.
Un modelo de embeddings asigna vectores a consultas y documentos de modo que
los textos con significados relacionados tiendan a quedar cerca en el espacio
vectorial. La búsqueda semántica compara esos vectores, por ejemplo mediante
similitud coseno, y recupera pasajes relevantes aunque la consulta no repita
exactamente las palabras del documento.

Un embedding no es una respuesta ni una base de conocimiento por sí mismo:
solo representa contenido para facilitar comparaciones. La calidad depende
del modelo, del idioma, del dominio y del tamaño de los fragmentos. Para que
las distancias tengan sentido, los documentos y las consultas deben utilizar
el mismo modelo y la misma configuración de embeddings. No se deben comparar
vectores producidos por modelos distintos como si pertenecieran al mismo
espacio.

En un índice vectorial se almacena el vector junto con el texto original y
metadatos, como el nombre del archivo, la página y el identificador de
fragmento. Al recibir una pregunta, el sistema calcula su embedding, busca los
vectores más cercanos y recupera el contenido asociado. La similitud es una
señal de relevancia, no una prueba de que una afirmación sea verdadera; el
contenido recuperado debe revisarse como evidencia antes de redactar una
respuesta.

Los resultados también dependen de la consulta. Una pregunta concreta que
incluye el concepto y las restricciones relevantes suele recuperar mejor que
una frase vaga. Un filtro por metadatos puede limitar la búsqueda a un manual,
versión o categoría, pero esos filtros no sustituyen la comparación semántica.
