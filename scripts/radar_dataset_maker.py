import sys
import matplotlib.pyplot as plt
import time
from pathlib import Path

sys.path.insert(-1, '/workspace/code/corelibs/src/tools-python')
sys.path.insert(-1, '/workspace/code/corelibs/build/datatypes')
sys.path.insert(-1, '/workspace/code/radar-utilities/build/radarutilities_datatypes_python')
sys.path.insert(-1, '/workspace/code/pyro/build/lib')
sys.path.insert(-1, '/workspace/code/corelibs/src/tools-python/mrg/adaptors')

from mrg.logging.indexed_monolithic import IndexedMonolithic
from mrg.adaptors.radar import pbNavtechRawConfigToPython, pbNavtechRawScanToPython


def export_radar_images():
    start_time = time.time()
    radar_path = '/workspace/data/RadarDataLogs/2017-08-18-11-21-04-oxford-10k-with-radar-1/logs/radar/cts350x/' \
                 '2017-08-18-10-21-06/cts350x_raw_scan.monolithic'
    radar_config_path = '/workspace/data/RadarDataLogs/2017-08-18-11-21-04-oxford-10k-with-radar-1/logs/radar/cts350x/' \
                        '2017-08-18-10-21-06/cts350x_config.monolithic'
    radar_image_path = "/workspace/Desktop/little-images/"
    Path(radar_image_path).mkdir(parents=True, exist_ok=True)

    radar_config_mono = IndexedMonolithic(radar_config_path)
    config_pb, name, timestamp = radar_config_mono[0]
    config = pbNavtechRawConfigToPython(config_pb)
    radar_mono = IndexedMonolithic(radar_path)
    scan_index = 123
    pbRawScan, name_scan, timestamp_scan = radar_mono[scan_index]
    radar_sweep = pbNavtechRawScanToPython(pbRawScan, config)
    image_dim = 256
    width, height, res = image_dim, image_dim, config.bin_size_or_resolution
    cart_img = radar_sweep.GetCartesian(pixel_width=width, pixel_height=height, resolution=res,
                                        method='slow', verbose=False)

    plt.figure(dpi=90) # do we need dpi?
    plt.imshow(cart_img)
    plt.savefig("%s%s%i%s" % (radar_image_path, "frame-", 123, ".png"), bbox_inches='tight')
    plt.close()
    print("--- Radar image generation execution time: %s seconds ---" % (time.time() - start_time))


def main():
    # Define a main loop to run and show some example data if this script is run as main
    print("Starting image exporting...")
    export_radar_images()


if __name__ == "__main__":
    main()
