# ApiCultor: Extrae la miel de RedPanal

Implementado para realizar una una performance multidisciplinaria basada en los sonidos de la plataforma http://redpanal.org

El código da soporte para el procesamiento de los mismos utilizando técnicas de MIR (Music Information Retrieval) para la "extracción" de parámetros que los caractericen para luego clasficarlos, segmentarlos y manipularlos según los criterios elegidos.

## Componentes
* Descarga archivos de audio por tag
* Algoritmos MIR para extraer descriptores de las pistas
* Algoritmos para segmentar los archivos de audio con distintos criterios
* Algoritmos para clasificar y agrupar los archivos de la base de datos redpanalera

#Dependencias
* pip install bs4
* pip install regex
* pip install wget
* pip install matplotlib
* Essentia (ver instrucciones para compilar aquí: http://essentia.upf.edu/documentation/installing.html)
* pip install numpy scipy scikit-learn
* pip install smst

# Uso 
## Bajar archivos de prueba y aplicar MIR
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
