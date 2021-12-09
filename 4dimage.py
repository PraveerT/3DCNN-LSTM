import visvis as vv
vv.use('wx')

import numpy as np
from matplotlib.image import imread
from matplotlib.cbook import get_sample_data

imgdata = imread(get_sample_data('lena.png'))

nr, nc = imgdata.shape[:2]
x,y = np.mgrid[:nr, :nc]
z = np.ones((nr, nc))

for ii in xrange(5):
    vv.functions.surf(x, y, z*ii*100, imgdata, aa=3)