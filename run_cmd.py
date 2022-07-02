import os
import sys

for k in range(30):
    print('start #%d' % k)
    success_flag = os.system("python demo_real_zzh.py --idx %d" % k)
    if success_flag != 0:
        sys.exit()
    print('finish #%d' % k)
