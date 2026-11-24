# ChromaDB, persistencia y fragmentación

ChromaDB es una base de datos vectorial que conserva embeddings, documentos y
metadatos en colecciones. Una aplicación local puede indicar un directorio de
persistencia para que el índice siga disponible después de cerrar el proceso.
Cada fragmento debe tener un identificador estable: esto permite reconocer
contenido que ya fue indexado, evitar duplicados y retirar fragmentos que
dejaron de existir en los archivos de origen.

Antes de indexar, documentos extensos se dividen en fragmentos pequeños
(chunks). Un fragmentador recursivo intenta conservar primero párrafos,
líneas y frases, y solo luego corta a nivel de caracteres. El tamaño se puede
medir en tokens para aproximarlo a los límites del modelo. Un solapamiento
entre fragmentos vecinos conserva contexto que podría quedar justo en un
borde. El tamaño apropiado depende de la estructura del material: fragmentos
muy grandes mezclan temas, mientras que fragmentos muy pequeños separan
definiciones de sus explicaciones.

La ingesta debe ser idempotente. En cada ejecución, se calcula un identificador
a partir del archivo, la posición y el contenido de cada fragmento. Los IDs
que ya están en la colección se conservan; solo se calculan embeddings para
nuevos fragmentos y se eliminan los IDs obsoletos. Así, volver a ejecutar el
script sin cambios en los datos no genera copias ni costos de embeddings
innecesarios.

El índice debe recordar el modelo de embeddings con el que fue construido. Si
cambia el modelo, la dimensión o el espacio vectorial puede no ser compatible.
La aplicación debe detectar el cambio y solicitar una reindexación explícita,
en vez de comparar silenciosamente vectores incompatibles. Los metadatos de
origen y el ID del chunk acompañan cada fragmento recuperado para poder citar
las fuentes.
