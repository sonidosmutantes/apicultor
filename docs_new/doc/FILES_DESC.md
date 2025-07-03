# Files
* `WebScrapingDownload.py` descarga los primeros diez archivos de la base de datos redpanalera tomando en cuenta el tag. Solamente hay que especificar el tag de archivos para buscar.
* `DoSegmentation.py` segmenta los archivos de audio en cuadros de corta duración.
* `RandomSegmentation.py` segmenta los archivos de audio en cuadros de duración aleatoria.
* `run_MIR_analysis.py` te muestra los valores de las pistas en base a conceptos de interés como el ataque, la frecuencia del centroide espectral, los BPM, y otros conceptos. También procesa los sonidos en base a las descripciones hechas.
* `SoundSimilarity.py` muestra clusters entre los sonidos para encontrar similitud basandose en descriptores seleccionados, luego guarda esos sonidos en carpetas de clusters y posteriormente los sonidos son remixados en base a la similaridad sonora
* `Sonification.py` También procesa los sonidos en base a las descripciones hechas.
* `MusicEmotionMachine.py` clasifica los sonidos en base a sus emociones. Luego se puede correr la máquina de estados emocionales musicales (Johnny) para remixar todos los sonidos y reproducirlos en tiempo real
