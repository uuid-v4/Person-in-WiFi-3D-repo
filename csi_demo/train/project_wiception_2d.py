#!/usr/bin/env python3
"""Project cached 3D keypoints into normalized orthographic front-view 2D."""
import argparse
import json
import os
import shutil

import numpy as np

FRONT_VIEW_AXES = [0, 2]  # Wiception3D: x = left-right, y = depth, z = height (increasing downward)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--src', required=True, help='Source ESP-like cache root')
    parser.add_argument('--dst', required=True, help='Destination 2D cache root')
    parser.add_argument('--link', action='store_true',
                        help='Symlink csi.npy instead of copying it')
    args = parser.parse_args()

    train_dir = os.path.join(args.src, 'train')
    train_keypoints = np.load(os.path.join(train_dir, 'keypoints.npy'), mmap_mode='r')
    train_counts = np.load(os.path.join(train_dir, 'num_person.npy'))
    valid_train = np.concatenate(
        [train_keypoints[i, :int(count)] for i, count in enumerate(train_counts)], axis=0)
    minimum = valid_train[..., FRONT_VIEW_AXES].min(axis=(0, 1))
    maximum = valid_train[..., FRONT_VIEW_AXES].max(axis=(0, 1))
    margin = (maximum - minimum) * 0.05
    lower, upper = minimum - margin, maximum + margin
    span = upper - lower

    projection = {
        'method': 'orthographic_front_view_xz',
        'source_axes': ['x', 'z'],
        'train_valid_min': minimum.tolist(),
        'train_valid_max': maximum.tolist(),
        'margin_fraction_per_side': 0.05,
        'bounds_min': lower.tolist(),
        'bounds_max': upper.tolist(),
    }

    os.makedirs(args.dst, exist_ok=True)
    for split in ('train', 'test'):
        source = os.path.join(args.src, split)
        destination = os.path.join(args.dst, split)
        os.makedirs(destination, exist_ok=True)
        keypoints = np.load(os.path.join(source, 'keypoints.npy'), mmap_mode='r')
        counts = np.load(os.path.join(source, 'num_person.npy'))
        output = np.lib.format.open_memmap(
            os.path.join(destination, 'keypoints.npy'), mode='w+', dtype=np.float32,
            shape=keypoints.shape[:-1] + (2,))
        for i, count in enumerate(counts):
            count = int(count)
            output[i] = 0
            if count:
                output[i, :count] = (keypoints[i, :count][..., FRONT_VIEW_AXES] - lower) / span
        output.flush()

        csi_source = os.path.join(source, 'csi.npy')
        csi_destination = os.path.join(destination, 'csi.npy')
        if args.link:
            os.symlink(os.path.abspath(csi_source), csi_destination)
        else:
            shutil.copyfile(csi_source, csi_destination)
        for name in ('num_person.npy', 'names.npy'):
            shutil.copyfile(os.path.join(source, name), os.path.join(destination, name))

        valid = np.concatenate(
            [output[i, :int(count)] for i, count in enumerate(counts)], axis=0)
        print(f'{split}: shape {output.shape}, normalized valid min '
              f'{valid.min(axis=(0, 1)).tolist()}, max {valid.max(axis=(0, 1)).tolist()}')

    with open(os.path.join(args.dst, 'projection.json'), 'w', encoding='utf-8') as file:
        json.dump(projection, file, indent=2)
        file.write('\n')


if __name__ == '__main__':
    main()
