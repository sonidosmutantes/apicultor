# About the project

* Interactive __DEMO__: [Experimental session with sounds from the cloud](https://www.youtube.com/watch?v=2sMsKvfZKGA) ([youtube link](https://www.youtube.com/watch?v=2sMsKvfZKGA))
* Publication: [Sound recycling from public databases](https://www.researchgate.net/publication/317388443_Sound_recycling_from_public_databases) @ [Audio Mostly](http://audiomostly.com) 2017 
* "Forgotten pieces from unknown people" and "Dialectic in suspense" constructed using APICultor, were showed in [NIPS 2017 Art Gallery](http://nips4creativity.com/music/) (Machine Learning for Creativity and Design, Long Beach, California, USA)
* "Sound recycling from public databases" was presented at [Technarte Los Angeles 2017](http://www.technarte.org/speakers/).
* Recycled sounds: [Sound miniatures](http://redpanal.org/p/reciclado-de-samples/)

APICultor born to realize interdisciplinary performances based on sounds of the web platform [http://redpanal.org](http://redpanal.org). The system is also useful to use with any other sound database on the internet or even running it locally.

The sound is processed digitally using different live-coding techniques. A pre-analysis based on Music Information Retrieval (MIR) stored in a database and accessed via a web-service REST API is combined with real-time processing and synthesis, random processes and human control via external interfaces.

![](doc/InstrCloudIT_play.png)

Examples available with SuperCollider, Pyo and pure data.

Spanish version: [README_es.md](README_es.md)

## Cloud Instrument

See [cloud_instrument/README.md](cloud_instrument/README.md)

Interactive __DEMO__: [https://www.youtube.com/watch?v=2sMsKvfZKGA](https://www.youtube.com/watch?v=2sMsKvfZKGA) 

![](doc/retrieve_ui.png)


## Performances

### Sonidos Mutantes

Interdisciplinary performances based on sounds of the web platform [Redpanal.org](Redpanal.org)

## Proofs of concept:

* [APICultor + No Input + Granular synthesis live performance](https://soundcloud.com/sonidosmutantes/apicultor-no-input).

* Compositions & recycled samples
  * [API-Cultor compositions](http://redpanal.org/p/apicultor/)
  * [Proyecto de reciclado de sonidos libres de la web en RedPanal.org](http://redpanal.org/p/reciclado-de-samples/) (sonido + referencia al original).

* Música generativa con máquina de estados MIR y sonidos libres de Freesound.org: 
  * "[Feature Thinking](https://soundcloud.com/hern-n-ordiales/feature-thinking)" (con sonidos libres Creative Commons de Freesound.org) by hordia.
  * State Machine @ CASo (Centro de Arte Sonoro) https://www.youtube.com/watch?v=sG1YUc8PQV4

* Performances en vivo utilizando estas herramientas:
  * Jam con guitarras + fx chain y sintes analógicos: [Perfo mutante en La Siesta del Fauno](https://soundcloud.com/hern-n-ordiales/perfo-mutante-mobile)
  * Closing performance of the Workshop of Experimental Sound:
[http://redpanal.org/a/banda-de-mutantes-cierre-taller/](http://redpanal.org/a/banda-de-mutantes-cierre-taller/)
  * La Noche de los Museos La Casa del Bicentenario: [Performance 29/10/2016](http://redpanal.org/a/performance-casa-tomada/) [Con visuales 3D (Blender game engine)](https://www.youtube.com/watch?v=eKcvkgtJIEo) ) **Reseña**: [enjambrebit-y-la-banda-de-mutantes/](http://blog.enjambrelab.com.ar/enjambrebit-y-la-banda-de-mutantes/)

* Remixes que toman audios libres de [RedPanal.org](http://redpanal.org/) para categorizarlos según diferentes tipos de emociones. Luego se elige una y se sincronizan las pistas, cambiando las tonalidades. De ser posible se separan de fuentes dentro de las mismas (by Mars Crop)
  * [Beats intro jazz](http://redpanal.org/a/sm-beats-remix/)
  * [Bass & DJ] (http://redpanal.org/a/sm-bass-guitar-plays-with-dj/)


## Components

* Mock web service with API REST to provide audio samples using MIR descriptors as parameters
* State machine, with each state defined by several MIR descriptors.
* Interaction with the free internet sound database [http://redpanal.org](http://redpanal.org)
 * API REST
 * Webscrapping by tag
* Algorithms MIR to extract mean values or by frame of audio samples
* Segmentation algorithms using different criteria.
* Classify algorithms and clustering of samples of the sound database
* Server OSC
* Examples in Supercollider, pyo
* Examples with MIDI and OSC controller. Locale and remote.

![](doc/Apicultor_chain.png)

# Dependencies

Tested under Linux, Mac OS (>10.11) and Windows 10.

Debian, Ubuntu 15.04 and 16.04 (and .10). And [Docker](module/docker.md) images.
Raspian @ Raspberry Pi

See [INSTALL.md](INSTALL.md)
