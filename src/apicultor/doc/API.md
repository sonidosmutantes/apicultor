# APICultor

Es una API en desarrollo, pensada para acceder a bases de datos propias y probada con sonidos de [RedPanal.org](http://redpanal.org/).

Para más información ver: [https://github.com/sonidosmutantes/apicultor](https://github.com/sonidosmutantes/apicultor)

Correr el servicio
```
$DEV_PATH/apicultor$ ./MockRedPanalAPI_service.py 
```

Por defecto escucha en el puerto 5000.

Notación de punto fijo para evitar la coma. Los segundos se multiplican por mil (Ej: 0.1 se transforma en 100 y 1 segundo es 1000)

## Curl

Nota: En las llamadas reemplazar localhost por la IP y puerto donde esta efectivamente corriendo el servicio. 

```
$ curl http://localhost:5000/search/mir/samples/duration/greaterthan/2000/5 -o desc.tmp
```

### Get 5 new sample file from APICultor (HFC < 1 )
```
http://localhost:5000/search/mir/samples/HFC/lessthan/1000/5
```

### Search MIR desc greater than (SAMPLES) HFC>40 / 5 valores

```
http://localhost:5000/search/mir/samples/HFC/greaterthan/40000/5
```

### Duration <1 (10 samples)
```
http://localhost:5000/search/mir/samples/duration/lessthan/1000/10
```

### Duration >2s (8 samples)
```
http://%:5000/search/mir/samples/duration/greaterthan/2000/8
```

## SuperCollider

```
~ip = "127.0.0.1"; // localhost or WebService IP
~tmp_path = "/tmp";
~samples_path = "../apicultor/samples" //set your download path

m = 15; //Length of sound list

//Get a new sample file from APICultor (HFC < 1 )
format("curl http://%:5000/search/mir/samples/HFC/lessthan/1000/% -o %/desc.tmp", ~ip,m,~tmp_path).unixCmd;

f = FileReader.read(~tmp_path++"/desc.tmp".standardizePath); //array
v = f.at(m.rand)[0]; //select a random value from array (0..10 range)
v.postln(); //selected file

v = (~samples_path++v.replace("./samples/","/")).replace(" ",""); //trim spaces and ./samples
~buf = Buffer.read(s, v ); // new buffer A

~buf.play
```

## Python3 simple API call

```
import http.client
conn = http.client.HTTPConnection("apicultor:5000")

payload = "filePath=%2Fmedia%2Fmnt%2Ffiles%2Ftest2222-file.pdf"

headers = {
    'cache-control': "no-cache",
    }

conn.request("GET", "/search/mir/samples/HFC/greaterthan/40000/5", payload, headers)

res = conn.getresponse()
data = res.read()

print(data.decode("utf-8"))
```
