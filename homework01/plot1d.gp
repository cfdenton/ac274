set term png
set key off
set offsets 0, 0, 0.05, 0.05
set yrange [0:1]
outname = filename[0:strlen(filename)-4].'.png'
set output outname
plot filename with lines
