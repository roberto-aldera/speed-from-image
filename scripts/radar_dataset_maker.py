import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
from pathlib import Path
import settings

sys.path.insert(-1, "/workspace/code/corelibs/src/tools-python")
sys.path.insert(-1, "/workspace/code/corelibs/build/datatypes")
sys.path.insert(-1, "/workspace/code/radar-utilities/build/radarutilities_datatypes_python")
sys.path.insert(-1, "/workspace/code/pyro/build/lib")
sys.path.insert(-1, "/workspace/code/corelibs/src/tools-python/mrg/adaptors")

from mrg.logging.indexed_monolithic import IndexedMonolithic
from mrg.adaptors.radar import pbNavtechRawConfigToPython, pbNavtechRawScanToPython


def export_radar_images():
    start_time = time.time()
    Path(settings.RADAR_IMAGE_DIR).mkdir(parents=True, exist_ok=True)

    radar_config_mono = IndexedMonolithic(settings.RADAR_CONFIG)
    config_pb, name, timestamp = radar_config_mono[0]
    config = pbNavtechRawConfigToPython(config_pb)
    radar_mono = IndexedMonolithic(settings.RAW_SCAN_MONOLITHIC)

    image_dim = 256
    width, height, res = image_dim, image_dim, config.bin_size_or_resolution
    start_index = 123
    export_total = 3

    for i in range(export_total):
        scan_index = start_index + i
        pb_raw_scan, name_scan, timestamp_scan = radar_mono[scan_index]
        radar_sweep = pbNavtechRawScanToPython(pb_raw_scan, config)
        cart_img = radar_sweep.GetCartesian(pixel_width=width, pixel_height=height, resolution=res,
                                            method='cv2', verbose=False)
        img = Image.fromarray(cart_img.astype(np.uint8), 'L')
        img.save("%s%s%i%s" % (settings.RADAR_IMAGE_DIR, "radar_frame-", scan_index, ".png"))
        img.close()
    print("--- Radar image generation execution time: %s seconds ---" % (time.time() - start_time))


def main():
    # Define a main loop to run and show some example data if this script is run as main
    print("Starting image exporting...")
    export_radar_images()


if __name__ == "__main__":
    main()
