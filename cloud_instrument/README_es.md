# Instrumento colaborativo basado en Tecnologías de la Información y la Nube (Internet)

Frente a la cantidad de información de uso libre disponible en Internet y la necesidad de explorar más allá de las fronteras estéticas impuestas por el software de uso habitual en la música, se desarrolló un nuevo flujo de trabajo que explota las bases de datos de sonidos online en cuasi tiempo real. El mismo se apoya en herramientas preexistentes de Software Libre y protocolos estándares de comunicación, así como aprovecha tecnologías disponibles gracias al auge actual de Big Data, Analytics y Machine Learning. El resultado es un sistema distribuido que utiliza descriptores sonoros basados en algoritmos de extracción de características para explorar de forma inteligente bases de datos locales o remotas y encontrar sonidos con propiedades tímbricas bien definidas. Luego, estos son transformados, mezclados y sintetizados en tiempo real.
Más allá de facilitar la reutilización de sonidos, también promueve la colaboración entre artistas permitiendo que múltiples usuarios o clientes accedan a la API de forma concurrente.
Entre las posibles aplicaciones, además de funcionar como herramienta de búsqueda o instrumento experimental en una agrupación, puede ser usado de forma automatizada e independiente, por ejemplo como parte de una instalación multimedia.


![](../doc/controller.jpg)

# Licencia

Software Libre, compartido con GPL v3 ([LICENSE](LICENSE)).

# Uso

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
* Run OpenStageControl with ui/nube_sinte_simple.js
* Tweak controlls and press the righ button to make the search on Cloud



![](doc/InstrNubeTI_repr.png)

![](doc/Apicultor_chain.png)

![](doc/retrieve_ui.png)

# Dependencias

Testeado en Linux, Mac OS (>10.11) y Windows 10.

Debian, Ubuntu 15.04 y 16.04 (y .10). E imágenes Docker.
Raspian @ Raspberry Pi

Para dependencias, ver [INSTALL.md](INSTALL.md).
