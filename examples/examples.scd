// Supercollider IDE: Exec=scide

r = NetAddr.new("127.0.0.1", 5005);
r.sendMsg('/search',"bajo");
r.sendMsg('/search',"platillo");



// Get value from REST API
"wget http://127.0.0.1:5000/pistas/126/descriptor/lowlevel.hfc.mean -O desc.tmp".unixCmd;
f = FileReader.read("./desc.tmp".standardizePath);
v = f[0][0]
