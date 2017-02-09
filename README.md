# ¿Qué es?

ApiCultor fue implementado para realizar performances multidisciplinarias basadas en los sonidos de la plataforma [http://redpanal.org](http://redpanal.org) pero sirve para trabajar con cualquier otra base de datos sonora en internet o disponible localmente.

El código da soporte para el procesamiento de sonidos de la web utilizando técnicas de MIR (Music Information Retrieval) para la "extracción" de parámetros que los caractericen para luego clasficarlos, segmentarlos y manipularlos según los criterios elegidos.

Funciona a través de una API REST para la consulta externa de archivos y descriptores desde SuperCollider, pyo, pd o cualquier otro tipo de entorno que maneje protocolos estándar.

## Performances utilizando estas herramientas:

### Cierre de Taller de Experimentación Sonora:

[http://redpanal.org/a/banda-de-mutantes-cierre-taller/](http://redpanal.org/a/banda-de-mutantes-cierre-taller/)


### La Noche de los Museos La Casa del Bicentenario 29/10/2016
[http://redpanal.org/a/performance-casa-tomada/](http://redpanal.org/a/performance-casa-tomada/)

[https://www.youtube.com/watch?v=eKcvkgtJIEo](https://www.youtube.com/watch?v=eKcvkgtJIEo)

Con visuales 3D (Blender + game engine)

**Reseña**: [http://blog.enjambrelab.com.ar/enjambrebit-y-la-banda-de-mutantes/](http://blog.enjambrelab.com.ar/enjambrebit-y-la-banda-de-mutantes/)

## Sonidos Mutantes
*Sonidos Mutantes, propone performances basadas en el uso artístico de bases de datos preexistentes, las mismas construidas por audios o músicas libres, por ejemplo aquellas de la plataforma colaborativa de RedPanal.org, la cuál es de acceso público vía internet. Estos sonidos, analizados y procesados en tiempo real se combinan en una improvisación colectiva con músicos en vivo bajo consignas preestablecidas, dando lugar a composiciones que mutan a lo largo del tiempo y en función de los propios músicos y de la comunidad toda. Ya que el público podrá participar de la performance subiendo audios y haciendo búsquedas o comentarios en la plataforma web de RedPanal.*

[https://www.facebook.com/SonidosMutantes](https://www.facebook.com/SonidosMutantes)

### Pruebas de concepto (demos viejas):
* Demo máquina de estados + MIR data + OSC + API + Tests: [http://www.youtube.com/watch?v=pMBl_4y6UmE](http://www.youtube.com/watch?v=pMBl_4y6UmE)
* Integración con controlador MIDI + Supercollider + ApicultorWebService: [https://www.youtube.com/watch?v=X0M_gTOZnNQ](https://www.youtube.com/watch?v=X0M_gTOZnNQ)

## Componentes
* Descarga archivos de audio por tag
* Algoritmos MIR para extraer descriptores de las pistas
* Algoritmos para segmentar los archivos de audio con distintos criterios
* Algoritmos para clasificar y agrupar los archivos de la base de datos redpanalera
* API REST para realizar consultas sobre redpanal (url audios+valores de descriptores)
* Server OSC
* Ejemplos de uso con Supercollider

Ver la [descripción de archivos](FILES_DESC.md) para más detalles.

# Dependencias

Ver [INSTALL.md](INSTALL.md)

# Uso (estado experimental)
## Bajar los sonidos redpanaleros y aplicar MIR
```
$ cd apicultor/data 
$ ./download-test-data.sh # Predefinidos (testing)
$ python WebScrapingDownload.py <nombre_del_tag>
$ cd ..
$ python run_MIR_analysis.py <directorio_de_sonidos_del_tag>
```
## Segmentar sonidos
```
$ python RandomSegmentation.py
```
## Similaridad Sonora
Procesar señales de salida para guardar los sonidos en base a clusters
```
$ python SoundSimilarity.py carpetadeltag
```
## Sonificación
```
$ python Sonification.py carpetadeltag
```
## SuperCollider
Performance and helper scripts in "supercollider/".

## Correr webservice (requiere api rest)
```
$ python MockRedPanalAPI_service.py
```

By default:
* Listen IP: 0.0.0.0
* Port: 5000

Ver ejemplos de uso en tests/Test_API.py

## Generar documentación HTML sobre la API REST
```
$ cd doc/ && ./update_api_doc.sh
```
Resultado: API-Documentation.html

## Máquina de estados emocionales de la música (MusicEmotionMachine)
```
$ python MusicEmotionMachine.py directoriodondeestadata multitag
```

(verdadero, clasifica Todos los audios descargados/ninguno, clasifica audios de un tag específico/falso, después de haber hecho la clasificación, correr de nuevo para llamar a Johnny (la máquina de estados emocionales) para que comienzen las transiciones emocionales con remixes en tiempo real de Todos los sonidos)

### Sobre el aprendizaje profundo de la MEM:

Con la intención de obtener la mejor clasificación posible de los sonidos basándose en las emociones que son capaces de transmitirnos, la tarea profunda consiste en este caso particular de reveer las activaciones con capas de máquinas de soporte vectorial para dar con la clasificación correcta. Las clasificaciones son en negativo o positivo, de acuerdo a la estimulación (arousal, no "activation"). Como la información del MIR es importante, el aprendizaje se hace respetando lo mejor posible las descripciones, lo que permite reveer las clasificaciones hechas hasta dar con las correctas.

## Docker

Ver tutorial sobre [docker](docker.md).

## Build (TODO)

~~Si ya tenés instaladas todas las dependencias se puede correr: 
```
$ sudo python setup.py install
```
y tener Apicultor instalado en el sistema~~
