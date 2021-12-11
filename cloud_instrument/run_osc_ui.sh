UI=apicultor-ui # cloud instrument

echo "UI: ui/$UI.json"
open-stage-control -l ui/$UI.json -s 127.0.0.1:57120 -p 8080 &

# No external UI, with -n flag (chrome browser only)
#open-stage-control -n  -l ui/$UI.json -s 127.0.0.1:57120 -p 7000
#open-stage-control -l ui/$UI.json -s 127.0.0.1:57120 -p 8080 --no-gui &
