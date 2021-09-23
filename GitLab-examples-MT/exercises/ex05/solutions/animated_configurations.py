#
# Configuration plot script for the HPCSE I course
# (c) 2014 Andreas Hehn <hehn@phys.ethz.ch>, ETH Zurich
#

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import re
import sys

posx_idx = 0
posy_idx = 1
mass_idx = 2
velx_idx = 3
vely_idx = 4
accx_idx = 5
accy_idx = 6

if len(sys.argv) != 2:
    print "usage:", sys.argv[0], "<sim_output_file>"
    sys.exit(1)

filename = sys.argv[1]

def get_particlenumber(filename):
    f = open(filename,'r')
    f.readline()
    scndline = f.readline()
    f.close()
    match = re.search('nparticles = (\d+)',scndline)
    if match:
        return int(match.group(1))
    else:
        raise RuntimeError("nparticles not found in second line of input file!")

fig = plt.figure()
nparticles= get_particlenumber(filename)
fcontent = np.loadtxt(filename, float)
num_per_row = len(fcontent[0])
fcontent = np.reshape(fcontent, (-1,nparticles,num_per_row))

print "Plotting", len(fcontent), "frames with", nparticles, "particles..."
data = []
maxforce = 0.0
for ap in fcontent:
    a = ap.transpose()
    force = np.sqrt(a[accx_idx]*a[accx_idx]+a[accy_idx]*a[accy_idx])
    f = force.max()
    if f > maxforce:
        maxforce = f
#    data += [np.array([a[posx_idx], a[posy_idx], log(a[mass_idx]), force])]
    data += [np.array([a[posx_idx], a[posy_idx], 5, force])]
 
ims = []
pd = data[0]
ppd = data[0]
for d in data:
    ims.append( (plt.scatter(d[posx_idx], d[posy_idx], s=pd[mass_idx], c=d[3]/maxforce, cmap=plt.cm.cool), ) )

#plt.xlim(0,1)
#plt.ylim(0,1)

ani = animation.ArtistAnimation(fig, ims, interval=1, repeat=True)
plt.colorbar()
plt.show()
