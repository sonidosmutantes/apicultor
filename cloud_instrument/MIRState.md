# MIR state

    enabled_descriptors = [
        "duration",
        "bpm",
        "hfc.mean",
        "spectral_complexity.mean",
        "spectral_centroid.mean",
        "pitch_centroid.mean",
        "pitch.mean",
        "inharmonicity.mean",
        "dissonance.mean",
        "pitch_salience.mean",
        "chords_strength.mean",
    ]

    default_sign = {
        "duration": "<",
        "bpm": "<",
        "hfc.mean": "=",
        "spectral_complexity.mean": "=",
        "spectral_centroid.mean": "=",
        "pitch_centroid.mean": "=",
        "pitch.mean": "=", #TODO: add a range of tolerance
        "inharmonicity.mean": "=",
        "dissonance.mean": "<",
        "pitch_salience.mean": "<",
        "chords_strength.mean": "<",
    }
    