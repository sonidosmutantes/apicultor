# API description

ApiCultor born to realize interdisciplinary performances based on sounds of the web platform [http://redpanal.org](http://redpanal.org). The system is also useful to use with any other sound database on internet or even running it locally.

Sound is processed digitally using different live-coding techniques. A pre-analysis based on MIR (Music Information Retrieval) stored in a database and accessed via a web-service (REST API) is combined with real-time processing and synthesis, random processes and human control via external interfaces.

Examples available with SuperCollider, pyo and pd.


## Performances

### Sonidos Mutantes

Interdisciplinary performances based on sounds of the web platform [Redpanal.org](Redpanal.org)

## Closing performance of the Workshop of Experimental Sounda:

[http://redpanal.org/a/banda-de-mutantes-cierre-taller/](http://redpanal.org/a/banda-de-mutantes-cierre-taller/)

## La Noche de los Museos La Casa del Bicentenario 29/10/2016
[http://redpanal.org/a/performance-casa-tomada/](http://redpanal.org/a/performance-casa-tomada/)

[https://www.youtube.com/watch?v=eKcvkgtJIEo](https://www.youtube.com/watch?v=eKcvkgtJIEo)

3D visuals (Blender + game engine)

**Article**: [http://blog.enjambrelab.com.ar/enjambrebit-y-la-banda-de-mutantes/](http://blog.enjambrelab.com.ar/enjambrebit-y-la-banda-de-mutantes/)

[https://www.facebook.com/SonidosMutantes](https://www.facebook.com/SonidosMutantes)

## Proofs of concept (old demos):

* Demo máquina de estados + MIR data + OSC + API + Tests: [http://www.youtube.com/watch?v=pMBl_4y6UmE](http://www.youtube.com/watch?v=pMBl_4y6UmE)
* Integración con controlador MIDI + Supercollider + ApicultorWebService: [https://www.youtube.com/watch?v=X0M_gTOZnNQ](https://www.youtube.com/watch?v=X0M_gTOZnNQ)

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

# Dependencias

Tested under Linux, Mac OS (>10.11) and Windows 10.

Debian, Ubuntu 15.04 and 16.04 (and .10). And Docker images.
Raspian @ Raspberry Pi

See [INSTALL.md](INSTALL.md)
