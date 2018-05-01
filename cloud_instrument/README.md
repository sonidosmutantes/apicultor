# Usage

* Install dependencies
* Create a config file
        {
            "sound.synth": "supercollider",
            "api": "freesound",
            "Freesound.org": [
                { "API_KEY": ""
                }
            ]
        }

* If you are going to use SuperCollider run synth/osc_server_and_player.scd first
* Run ./CloudInstrument.py
* Run OpenStageControl with ui/apicultor.js (sending OSC to localhost and 9001 port)
* Tweak controlls and press the righ button to make the search on Cloud


 Dependencies: [INSTALL.md](INSTALL.md)

### Linux: jackd-no-disconnect-config 
~/.jackdrc

    /usr/local/bin/jackd -P75 -t2000 -dalsa -dhw:S2 -p4096 -n7 -r44100 -s