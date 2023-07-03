import sys

_, f, x, y, z = sys.argv
bx = int(x).to_bytes(2, 'big')
by = int(y).to_bytes(2, 'big')
bz = int(z).to_bytes(2, 'big')

ff = open(f, "wb")
ff.write(bz+by+bx)
ff.close()
