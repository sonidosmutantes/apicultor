# ApiCultor: Extrae la miel de RedPanal

Implementado para realizar una una performance multidisciplinaria basada en los sonidos de la plataforma http://redpanal.org

El código da soporte para el procesamiento de los mismos utilizando técnicas de MIR (Music Information Retrieval) para la "extracción" de parámetros que los caractericen para luego clasficarlos, segmentarlos y manipularlos según los criterios elegidos.

Servidor OSC + API REST para la consulta externa de archivos y descriptores desde SuperCollider, pd o cualquier otro tipo de entorno.

Prueba de concepto (versión inicial) http://www.youtube.com/watch?v=pMBl_4y6UmE

## Componentes
* API REST para realizar consultas sobre redpanal (url audios+valores de descriptores)
* Algoritmos MIR para extraer descriptores de las pistas
* Algoritmos para segmentar los archivos de audio con distintos criterios
* Algoritmos para clasificar y agrupar los archivos de la base de datos redpanalera
* Server OSC
* Ejemplos de uso con SuperCollider

# Uso (temporal)

## Bajar archivos de prueba y aplicar MIR
$ cd data && ./download-test-data.sh
$ ./run_MIR_analysis.py

$ cd data 
$ ./WebScrapingDownload.py 
$ cd /carpetadondeesta/apicultor
$ ./run_MIR_analysis.py

#Docu
* WebScrapingDownload.py descarga los primeros diez archivos de la base de datos redpanalera tomando en cuenta el tag. Solamente hay que establecer el tag de archivos para buscar y tener la carpeta creada con el nombre del tag.
* DoSegmentation.py segmenta los archivos de audio en cuadros de corta duración.
* RandomSegmentation.py segmenta los archivos de audio en cuadros de duración aleatoria
* run_MIR_analysis.py te muestra los valores de las pistas en base a conceptos de interés como el ataque, la frecuencia del centroide espectral, los BPM, y muestra muchas cosas mas!
* SoundSimilarity.py lee los descriptores e intenta encontrar similaridades entre los sonidos para encontrar los sonidos más compatibles para tu experimento

#SuperCollider
SuperCollider code in "examples/" 
## Correr webservice (api rest)
$ ./MockRedPanalAPI_service.py

(escucha en localhost, puerto 5000)

Ver ejemplos de uso en tests/API_Tests.py

## Generar documentación HTML sobre la API REST
$ cd doc/ && ./update-api-doc.sh

Resultado: API-Documentation.html

## Supercollider

Ver ejemplos en examples/


