import sys
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path

sys.path.insert(-1, '/Users/roberto/code/corelibs/src/tools-python')
sys.path.insert(-1, '/Users/roberto/code/corelibs/build/datatypes')
sys.path.insert(-1, '/Users/roberto/code/radar-utilities/build/radarutilities_datatypes_python')
sys.path.insert(-1, '/Users/roberto/code/pyro/build/lib')
sys.path.insert(-1, '/Users/roberto/code/corelibs/src/tools-python/mrg/adaptors')

from mrg.logging import IndexedMonolithic
from mrg.adaptors import image
from mrg.radar import utils

import protobuf.image.pbImageArray_pb2 as pbImageArray_pb2
import pyro

start_time = time.time()

scan_path = '/Users/roberto/data/RadarDataLogs/2017-08-18-11-21-04-oxford-10k-with-radar-1/logs/radar/cts350x/' \
            '2017-08-18-10-21-06/cts350x_raw_scan.monolithic'
radar_image_path = "/Users/roberto/Desktop/little-images/"
Path(radar_image_path).mkdir(parents=True, exist_ok=True)

indexed_mono = IndexedMonolithic(scan_path)
pb_polar_imgs = []
idx_start = 150
idx_end = idx_start + 4
num_bins_in_azi = 1000
num_azi = 400 - 1

for i in range(idx_start, idx_end):
    image_from_file = pbImageArray_pb2.pbImageArray()
    t = pyro.PyNavtechScanToImage().Convert(indexed_mono[i][0].SerializeToString(), num_azi, num_bins_in_azi)
    image_from_file.ParseFromString(t)
    pb_polar_imgs.append(image_from_file)

polar_imgs = []
for i in range(0, idx_end - idx_start):
    polar_img, timestamps = image.PbImageArrayToPython(pb_polar_imgs[i])
    polar_imgs.append(polar_img)
#     plt.figure(dpi=90)
#     imgplot = plt.imshow(polar_img[0]);
#     plt.savefig("%s%s%i%s" % (figpath, "frame-", i + idx_start, ".png"), bbox_inches='tight')

pixel_width = pixel_height = 800
resolution = 0.25  # 0.1728
azimuths = np.array(np.linspace(0, 2 * np.pi, num_azi))
ranges = np.array(np.linspace(0, num_bins_in_azi * resolution, num_bins_in_azi))

cart_imgs = []
for i in range(0, idx_end - idx_start):
    cart_img = utils.polar_img_to_cart(polar_imgs[i][0], azimuths, ranges, pixel_width, pixel_height, resolution,
                                       method='cv2')
    cart_imgs.append(cart_img)
    plt.figure(dpi=90)
    plt.imshow(cart_img)
    plt.savefig("%s%s%i%s" % (radar_image_path, "frame-", i + idx_start, ".png"), bbox_inches='tight')
    plt.close()

print("--- Radar image generation execution time: %s seconds ---" % (time.time() - start_time))
