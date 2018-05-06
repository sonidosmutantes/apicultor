# MIDI Specification


## General

| Ctrl                | MIDI     | Description      |
|---------------------|----------|------------------| 
| Master              | CC 7             | General volume   |
| Pan                 | CC 10            | General panning  |
|---------------------|------------------|------------------| 
| Voice1              | NoteOn 60+Bank2*4| Sound 1 or 5 |
| Voice2              | NoteOn 61+Bank2*4| Sound 2 or 6 |
| Voice3              | NoteOn 62+Bank2*4| Sound 3 or 7 |
| Voice4              | NoteOn 63+Bank2*4| Sound 4 or 8 |
|---------------------|------------------|------------------| 
| Distance Sensor     | CC 38    | Amount of grains |
| Bank1               | -        | (internal switch)|
| Bank2               | -        | (internal switch)|
| MIR                 | -        | (internal switch)|
| SYNTH               | -        | (internal switch)|


## Modes

| MIR mode               | MIDI     | SYNTH  mode               | MIDI     | Description      |
|------------------------|----------|---------------------------|----------|------------------| 
| BPM   | CC 6  | LFO Amount | CC 67 | |
| Key   | CC 66 | Rate       | CC 76 | |
| Duration | CC 102 | Gate | CC 65 | |
| Pitch    | CC 103 | Reverb | CC 91 | |
| Inharmonicity	| CC 104	| Delay	| CC 78 | |
| Dissonance	| CC 105	| Tone	| CC 74 | |
| HFC	| CC 106	| LP/BP/HP Filter	| CC 75 | |
| Pitch Salience	| CC 107	| Cutoff	| CC 73 | |
| Spectral Centroid	| CC 108	| Ressonance|	CC 71 | |
| Spectral Complexity|	CC 109|	Factor|	CC 77 | |
| Cluster X	|CC 110|	Spat X|	CC 114 | |
| Cluster Y	|CC 111|	Spat Y|	CC 115 | |
| Control L/R|	CC 112	|Control L/R	|CC 116 | |
| Retrieve / Reset	|CC 113	|Control UP/Down Reset FX|	CC 117 | |