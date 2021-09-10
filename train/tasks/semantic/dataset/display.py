#!/usr/bin/env python
import numpy as np
from mayavi import mlab

import mayavi
#mayavi.use("Agg")

fp = open("./sequences/00/velodyne/000201.bin", "rb")
#fp = open("./sequences/101/velodyne/010.bin", "rb")

pointcloud = np.fromfile(fp, dtype=np.float32, count=-1).reshape([-1,4])
print(pointcloud.shape)
x = pointcloud[:,0]
y = pointcloud[:,1]
z = pointcloud[:,2]
rem = pointcloud[:,3]

ranges = np.sqrt(x**2 + y**2 + z**2)

print(np.max(ranges))
print(np.min(ranges))

print(np.max(z))
print(np.min(z))

d = np.sqrt(x**2 + y**2)

vals = "a"
if vals == "height":
    col = z
else:
    col = d

fig = mlab.figure(bgcolor=(0,0,0), size=(640, 500))

#x,y,z,rem = np.random.random((4,40))
mlab.points3d(x,y,z, rem, mode="point", colormap="spectral", figure=fig)
mlab.colorbar()
mlab.axes()
#x = np.linspace(5,5,50)
#y = np.linspace(0,0,50)
#z = np.linspace(0,5,50)
#mlab.plot3d(x,y,z)
mlab.show()
