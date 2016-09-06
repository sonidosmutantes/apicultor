# ApiCultor: Extrae la miel de RedPanal

Implementado para realizar una una performance multidisciplinaria basada en los sonidos de la plataforma http://redpanal.org

El código da soporte para el procesamiento de los mismos utilizando técnicas de MIR (Music Information Retrieval) para la "extracción" de parámetros que los caractericen para luego clasficarlos, segmentarlos y manipularlos según los criterios elegidos.

Servidor OSC + API REST para la consulta externa de archivos y descriptores desde SuperCollider, pd o cualquier otro tipo de entorno.

Pruebas de concepto (versión inicial):
* Demo máquina de estados + MIR data + OSC + API + Tests: http://www.youtube.com/watch?v=pMBl_4y6UmE
* Integración con controlador MIDI + Supercollider + ApicultorWebService: https://www.youtube.com/watch?v=cTnQQONjXrY 

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

## Descargar por tag, aplicar MIR, procesar señales de salida para guardar los sonidos en base a clusters
$ cd data 
$ ./WebScrapingDownload.py 
$ cd /carpetadondeesta/apicultor
$ ./run_MIR_analysis.py
$./SoundSimilarity.py

#Docu
* WebScrapingDownload.py descarga los primeros diez archivos de la base de datos redpanalera tomando en cuenta el tag. Solamente hay que especificar el tag de archivos para buscar.
* DoSegmentation.py segmenta los archivos de audio en cuadros de corta duración.
* RandomSegmentation.py segmenta los archivos de audio en cuadros de duración aleatoria.
* run_MIR_analysis.py te muestra los valores de las pistas en base a conceptos de interés como el ataque, la frecuencia del centroide espectral, los BPM, y otros conceptos. También procesa los sonidos en base a las descripciones hechas.
* SoundSimilarity.py muestra clusters entre los sonidos para encontrar similitud basandose en descriptores seleccionados, luego guarda esos sonidos en carpetas de clusters.

#SuperCollider
SuperCollider code in "examples/" 

## Correr webservice (requiere api rest)
$ ./MockRedPanalAPI_service.py

(escucha en localhost, puerto 5000)

Ver ejemplos de uso en tests/API_Tests.py

## Generar documentación HTML sobre la API REST
$ cd doc/ && ./update-api-doc.sh

Resultado: API-Documentation.html


