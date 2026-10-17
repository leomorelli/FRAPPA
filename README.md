# FRAPPA
Python FRAP analysis from .tiff files


### Info

1. Nella cartella TIFF, ci sono i TIFF :)
2. I TIFF sono dei timelapse con 21 frames (il primo frame è il pre-bleaching, dal secondo in poi c'è la sequenza post-bleaching). In ogni FOV ho bleachato una decina di droplets.
3. Le ROI d'interesse sono: a) bleached [le regioni bleachate delle droplets]; b) TOT [ROI che includa tutta la droplet, sia area bleachata che non]; c) Reference [droplets NON bleachate].
4. La normalizzazione che faccio è:

`A = [bleached(t)]/[Reference(t)]`

`B = [A(t)]/[A(pre)]`

`C = [TOT(t)]/[Reference(t)]`

`D = [C(t)]/[C(pre)]`

`Normalizzazione finale = [B(t)]/[D(t)]`

5. E' possibile ottenere un .csv che includa anche i raw values delle intensità delle ROI (bleached, TOT, reference) per ogni timepoint e non solo la normalizzazione?
