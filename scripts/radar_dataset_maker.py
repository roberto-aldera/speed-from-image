import sys
import numpy as np
from PIL import Image
import time
from pathlib import Path
import settings

sys.path.insert(-1, "/workspace/code/corelibs/src/tools-python")
sys.path.insert(-1, "/workspace/code/corelibs/build/datatypes")
sys.path.insert(-1, "/workspace/code/radar-utilities/build/radarutilities_datatypes_python")
sys.path.insert(-1, "/workspace/code/corelibs/src/tools-python/mrg/adaptors")


def get_poses_from_file():
    print("Processing poses...")
    from mrg.logging import MonolithicDecoder
    from mrg.adaptors.transform import PbSerialisedTransformToPython

    dataset_path = settings.RADAR_DATASET_PATH
    ro_relative_poses_path = dataset_path + "/ro_relative_poses.monolithic"
    print("reading ro_relative_poses_path: " + ro_relative_poses_path)
    monolithic_decoder = MonolithicDecoder(
        ro_relative_poses_path)

    ro_se3s = []
    ro_timestamps = []
    for pb_serialised_transform, _, _ in monolithic_decoder:
        serialised_transform = PbSerialisedTransformToPython(
            pb_serialised_transform)
        ro_se3s.append(serialised_transform[0])
        ro_timestamps.append(serialised_transform[1])
    print("Finished reading", len(ro_timestamps), "poses.")
    return ro_se3s, ro_timestamps


def export_radar_images(ro_se3s, ro_timestamps, data_ratio, data_subset_type):
    from mrg.logging.indexed_monolithic import IndexedMonolithic
    from mrg.adaptors.radar import pbNavtechRawConfigToPython, pbNavtechRawScanToPython

    start_time = time.time()

    split_data_folder = "%s%s%s" % (settings.RADAR_IMAGE_DIR, data_subset_type, "/")
    Path(split_data_folder).mkdir(parents=True, exist_ok=True)

    radar_config_mono = IndexedMonolithic(settings.RADAR_CONFIG)
    config_pb, name, timestamp = radar_config_mono[0]
    config = pbNavtechRawConfigToPython(config_pb)
    radar_mono = IndexedMonolithic(settings.RAW_SCAN_MONOLITHIC)

    print("Length of ro_se3s:", len(ro_se3s))
    print("Length of ro_timestamps:", len(ro_timestamps))
    print("Length of scans monolithic:", len(radar_mono))

    width, height, res = settings.RADAR_IMAGE_DIMENSION, settings.RADAR_IMAGE_DIMENSION, config.bin_size_or_resolution
    start_index = 0
    num_samples = int(settings.TOTAL_SAMPLES * data_ratio)
    x_vals_labels = []

    for i in range(num_samples):
        scan_index = start_index + i
        pb_raw_scan, name_scan, _ = radar_mono[scan_index]
        radar_sweep = pbNavtechRawScanToPython(pb_raw_scan, config)

        cart_img = radar_sweep.GetCartesian(pixel_width=width, pixel_height=height, resolution=res,
                                            method='cv2', verbose=False)
        img = Image.fromarray(cart_img.astype(np.uint8), 'L')
        img.save("%s%s%s%i%s" % (split_data_folder, data_subset_type, "_", scan_index, ".png"))
        img.close()

        x_vals = np.zeros(settings.POSE_WINDOW_SIZE)
        for j in range(len(x_vals)):
            ro_idx = j + scan_index - 1  # -1 because there's no odometry for the first scan, so index is offset by 1
            x_vals[j] = ro_se3s[ro_idx][0, 3]
        x_vals_labels.append(x_vals)
    np.savetxt(("%s%s%s" % (split_data_folder, data_subset_type, "_x_vals_labels.csv")), x_vals_labels, delimiter=",",
               fmt="%10.5f")
    print("--- Radar image and pose generation execution time: %s seconds ---" % (time.time() - start_time))


def main():
    # Define a main loop to run and show some example data if this script is run as main
    print("Starting dataset generation...")
    ro_se3s, ro_timestamps = get_poses_from_file()
    export_radar_images(ro_se3s, ro_timestamps, settings.TRAIN_RATIO, settings.TRAIN_SUBSET)


if __name__ == "__main__":
    main()
