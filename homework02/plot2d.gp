#set terminal x11 size 800,500 enhanced font 'Verdana,10' persist
set term png
set offsets 0, 0, 0.05, 0.05
set style data pm3d
set zrange [0:1]
set pm3d implicit at s

set samples 10

set output filename[0:strlen(filename)-4].'.png'
splot filename
