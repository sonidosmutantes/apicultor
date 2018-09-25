## Music Emotion State Machine
(by mars crop)

```
$ python MusicEmotionMachine.py wherethere'sdata multitag[True/False/None]
```

(True, Classify all downloaded sounds in a directory. After classification has been performed, you must run again with the multitag option in False or None to call Johnny (the Music Emotion State Machine) to start emotional transitions and start the remix

To remix we use (from Deep Learning results) classification data to make emotive remixes and the decision variables to reconstruct scratching movements. We also emphasize on components searching (using a method to find out how many possible sound sources can be found) to generate simple remixes, where sound a can just take the beat and sound b has the harmonic role. We provide ourselves of many utilities (scratching methods, segmentations, etc) to make fun remixes

### About the Music Emotion Machine's Deep Learning:

Since we aim to classify all the sounds in RedPanal based on emotions, our task is to review activations in many runs using layers of support vector machine (this is known as Multi-Layer SVMs or Deep SVMs). By the time we only find four emotions in music. As the MIR information is important for all the other tasks, we do our best to respect the data we have, this allows us to review everything the best possible way. This is why we use Cross Validation and other tools to get the best information.

## Docker

See [docker](docker.md) and [Dockerfile](Dockerfile.md).


API listening in port 4999:
```
$ docker build -t apicultor_v-1.9 .
$ docker run -p 4999:5000 --name apicultor  -it --net="host"  apicultor_v0.9
```

## Build

Building it is slightly easy:

~~
```
$ sudo python3 setup.py install
```
And you will be provided of command line tools. At the moment these are all available:

* rpdl: download sounds from RedPanal
* rpdla: download sounds from archive.redpanal.org
* miranalysis: analyze sound files using our pythonic versions of some of MTG's Essentia algorithms
* musicemotionmachine: classify sounds according to emotions and run the Music Emotion Machine (MEM)
* sonify: sonify the outputs of the retriever class descriptors
* qualify: fix common sound artifacts //we still have to fix a hiss reduction bug which makes it very insensitive in most cases to hissings
* soundsimilarity: classify sounds according to similarity and remix
* mockrpapi: mock of RedPanal API
* audio2ogg: convert sound files to ogg (THIS IS IMPORTANT)
* smcomposition: Sonidos Mutantes performance 
~~
