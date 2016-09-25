# ApiCultor: Extrae la miel de RedPanal

Implementado para realizar una una performance multidisciplinaria basada en los sonidos de la plataforma http://redpanal.org

El código da soporte para el procesamiento de los mismos utilizando técnicas de MIR (Music Information Retrieval) para la "extracción" de parámetros que los caractericen para luego clasficarlos, segmentarlos y manipularlos según los criterios elegidos.

Servidor OSC + API REST para la consulta externa de archivos y descriptores desde SuperCollider, pd o cualquier otro tipo de entorno.

Pruebas de concepto (versión inicial):
* Demo máquina de estados + MIR data + OSC + API + Tests: http://www.youtube.com/watch?v=pMBl_4y6UmE
* Integración con controlador MIDI + Supercollider + ApicultorWebService: https://www.youtube.com/watch?v=X0M_gTOZnNQ

Sonidos Mutantes, propone performances basadas en el uso artístico de bases de datos preexistentes, las mismas construidas por audios o músicas libres, por ejemplo aquellas de la plataforma colaborativa de RedPanal.org, la cuál es de acceso público vía internet. Estos sonidos, analizados y procesados en tiempo real se combinan en una improvisación colectiva con músicos en vivo bajo consignas preestablecidas, dando lugar a composiciones que mutan a lo largo del tiempo y en función de los propios músicos y de la comunidad toda. Ya que el público podrá participar de la performance subiendo audios y haciendo búsquedas o comentarios en la plataforma web de RedPanal.


## Componentes
* API REST para realizar consultas sobre redpanal (url audios+valores de descriptores)
* Algoritmos MIR para extraer descriptores de las pistas
* Algoritmos para segmentar los archivos de audio con distintos criterios
* Algoritmos para clasificar y agrupar los archivos de la base de datos redpanalera
* Server OSC
* Ejemplos de uso con SuperCollider

# Uso (estado experimental)

## Descargar pistas/sonidos de RedPanal.org por tag
$ cd data 
$ #./download-test-data.sh # Predefinidos (testing)
$ ./WebScrapingDownload.py [TAG]

## Segmentar sonidos
$ ./RandomSegmentation.py

## Aplicar MIR
$ ./run_MIR_analysis.py

### Procesar señales de salida para guardar los sonidos en base a clusters
$ cd /carpetadondeesta/apicultor
$ ./SoundSimilarity.py

## SuperCollider
Performance and helper scripts in "supercollider/" 

## Correr webservice (requiere api rest)
$ ./MockRedPanalAPI_service.py

By default:
* Listen IP: 0.0.0.0
* Port: 5000

Ver ejemplos de uso en tests/Test_API.py

## Generar documentación HTML sobre la API REST
$ cd doc/ && ./update_api_doc.sh

Resultado: API-Documentation.html

## Sonificación
$ cd /apicultor
$ python Sonification.py carpetadeltag
## MÁQUINA DE ESTADOS EMOCIONALES DE LA MÚSICA
$ cd /apicultor
$ python MusicEmotionMachine.py directoriodondeestadata multitag(verdadero, clasifica Todos los audios descargados/ninguno, clasifica audios de un tag específico/falso, después de haber hecho la clasificación, correr de nuevo para llamar a Johnny (la máquina de estados emocionales) para que comienzen las transiciones emocionales con remixes en tiempo real de Todos los sonidos)

#Dependencias
* pip install bs4
* pip install regex
* pip install wget
* pip install colorama
* pip install matplotlib
* Essentia (ver instrucciones para compilar aquí: http://essentia.upf.edu/documentation/installing.html)
* pip install numpy scipy scikit-learn
* pip install smst
* pip install librosa
* pip install transitions

# Descripción de algunos scripts
* WebScrapingDownload.py descarga los primeros diez archivos de la base de datos redpanalera tomando en cuenta el tag. Solamente hay que especificar el tag de archivos para buscar.
* DoSegmentation.py segmenta los archivos de audio en cuadros de corta duración.
* RandomSegmentation.py segmenta los archivos de audio en cuadros de duración aleatoria.
* run_MIR_analysis.py te muestra los valores de las pistas en base a conceptos de interés como el ataque, la frecuencia del centroide espectral, los BPM, y otros conceptos
* SoundSimilarity.py muestra clusters entre los sonidos para encontrar similitud basandose en descriptores seleccionados, luego guarda esos sonidos en carpetas de clusters.
* Sonification.py Procesa los sonidos en base a las descripciones hechas.
* MusicEmotionMachine.py clasifica los sonidos en base a sus emociones. Si la clasificación es multitag (de todo el audio redpanalero), luego se puede correr la máquina de estados emocionales musicales (Johnny) para remixar todos los sonidos y reproducirlos en tiempo real

