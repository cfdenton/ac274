set terminal x11 size 800,500 enhanced font 'Verdana,10' persist
set offsets 0, 0, 0.05, 0.05
set style data pm3d
set pm3d implicit at s

set samples 10
splot filename
pause -1
