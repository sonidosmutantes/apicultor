// Supercollider IDE: Exec=scide

r = NetAddr.new("127.0.0.1", 5005);
r.sendMsg('/search',"bajo");
r.sendMsg('/search',"platillo");