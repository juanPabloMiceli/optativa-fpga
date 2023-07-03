import sys

_, f_name = sys.argv

f = open(f_name, "rb")
t = f.read(-1)
print(t)
print(int.from_bytes(t[0:2], 'big'))
print(int.from_bytes(t[2:4], 'big'))
print(int.from_bytes(t[4:6], 'big'))
