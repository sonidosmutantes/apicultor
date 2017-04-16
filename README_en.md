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

See [INSTALL.md](INSTALL.md)

## Music Emotion State Machine

```
$ python MusicEmotionMachine.py wherethere'sdata multitag[True/False/None]
```

(True, Classify all downloaded sounds in a directory. After classification has been performed, you must run again with the multitag option in False or None to call Johnny (the Music Emotion State Machine) to start emotional transitions and start the remix

To remix we use (from Deep Learning results) classification data to make emotive remixes and the decision variables to reconstruct scratching movements. We also emphasize on components searching (using a method to find out how many possible sound sources can be found) to generate simple remixes, where sound a can just take the beat and sound b has the harmonic role. We provide ourselves of many utilities (scratching methods, segmentations, etc) to make fun remixes

### About the Music Emotion Machine's Deep Learning:

Since we aim to classify all the sounds in RedPanal based on emotions, our task is to review activations in many runs using layers of support vector machine (this is known as Multi-Layer SVMs or Deep SVMs). By the time we only find four emotions in music. As the MIR information is important for all the other tasks, we do our best to respect the data we have, this allows us to review everything the best possible way. This is why we use Cross Validation and other tools to get the best information.
## Docker
