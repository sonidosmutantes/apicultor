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

(True, clasifica Todos los audios descargados. Después de haber hecho la clasificación, correr de nuevo con la opcion multitag en False o en None para llamar a Johnny (la máquina de estados emocionales) para que comienzen las transiciones emocionales con remixes en tiempo real de Todos los sonidos)

En los remixes utilizamos, ademas de la data acerca de las correspondientes clases, las variables de decision en el entrenamiento del DSVM para reconstruir movimientos de scratching. Ademas se pone enfasis en la busqueda de componentes que se puedan separar (utilizando un metodo que intenta encontrar las n fuentes en una mezcla automaticamente) para generar remixes simples, donde el sonido puede marcar el ritmo y el sonido b puede seguir un rol harmonico. Nos proveemos de varias utilidades (métodos de scratching, segmentaciones, etc) para que el remix resulte divertido

### Sobre el aprendizaje profundo de la MEM:

Con la intención de obtener la mejor clasificación posible de los sonidos basándose en las emociones que son capaces de transmitirnos, la tarea profunda consiste en este caso particular de reveer las activaciones con capas de máquinas de soporte vectorial para dar con la clasificación correcta. Por el momento solamente se utilizan cuatro clases de emociones para la clasificación. Como la información del MIR es importante, el aprendizaje se hace respetando lo mejor posible las descripciones, lo que permite reveer las clasificaciones hechas hasta dar con las correctas. Es por esto que contamos con nuestros modulos de validación cruzada y con nuestras utilidades matemáticas para clasificar los sonidos sin perder información.
## Docker

Ver tutorial sobre [docker](docker.md) y [Dockerfile](Dockerfile).

Servicio (API) escuchando en puerto 5000:
```
$ docker build -t apicultor_v0.9 .
$ docker run -p 5000:5000 --name apicultor  -it --net="host"  apicultor_v0.9
```

## Build (TODO)


~~Si ya tenés instaladas todas las dependencias se puede correr: 
```
$ sudo python setup.py install
```
y tener Apicultor instalado en el sistema~~
