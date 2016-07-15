# ApiCultor: Extrae la miel de RedPanal

Implementado para realizar una una performance multidisciplinaria basada en los sonidos de la plataforma http://redpanal.org

El código da soporte para el procesamiento de los mismos utilizando técnicas de MIR (Music Information Retrieval) para la "extracción" de parámetros que los caractericen para luego clasficarlos, segmentarlos y manipularlos según los criterios elegidos.

Servidor OSC + API REST para la consulta externa de archivos y descriptores desde SuperCollider, pd o cualquier otro tipo de entorno.


## Componentes
* API REST para realizar consultas sobre redpanal (url audios+valores de descriptores)
* Algoritmos MIR para extraer descriptores de las pistas
* Algoritmos para segmentar los archivos de audio con distintos criterios
* Algoritmos para clasificar y agrupar los archivos de la base de datos redpanalera
* Server OSC
* Ejemplos de uso con SuperCollider

