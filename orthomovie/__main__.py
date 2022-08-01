#!/usr/bin/python
"""
Create individual MP4 movies of 2D orthslices through a given voxel coordinate
over time from a 4D MRI dataset. Useful for visualizing pre and post motion
correction image quality.

AUTHOR
----
Mike Tyszka

PLACE
----
Caltech Brain Imaging Center

LICENSE
----
Copyright (c) 2022 Mike Tyszka

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os.path as op
import imageio
import numpy as np
import nibabel as nb
import argparse
from skimage.exposure import rescale_intensity


def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create orthslice movies from a 4D MRI dataset')

    parser.add_argument('-i', '--infile', required=True,
                        help='4D Nifti-1 MRI dataset filename')
    parser.add_argument('-c', '--center', required=True, nargs=3,
                        help='x, y and z voxel coordinate of orthoslice intersection')

    args = parser.parse_args()
    img_fname = args.infile
    img_stub = op.basename(img_fname).replace('.nii.gz', '').replace('.nii', '')
    x0, y0, z0 = np.array(args.center).astype(int)

    print(f'Input 4D image       : {img_fname}')
    print(f'Orthoslice (x, y, z) : ({x0}, {y0}, {z0})')

    # Load 4D image
    print(f'Loading 4D image')
    img_nii = nb.load(img_fname)
    img = img_nii.get_fdata()
    print(f'Image size (nx, ny, nz, nt) : {img.shape}')

    # Get global 5th and 95th percentiles
    print(f'Determining robust rescaling limits')
    lims = tuple(np.percentile(img, (5, 99)))

    # Extract 3D (2D x time) orthoslices
    print(f'Extracting orthoslices')
    xy = img[:, :, z0, :].squeeze()
    xz = img[:, y0, :, :].squeeze()
    yz = img[x0, :, :, :].squeeze()

    # Put time axis first
    xy = np.flip(np.moveaxis(xy, [0, 1, 2], [2, 1, 0]), axis=1)
    xz = np.flip(np.moveaxis(xz, [0, 1, 2], [2, 1, 0]), axis=1)
    yz = np.flip(np.moveaxis(yz, [0, 1, 2], [2, 1, 0]), axis=(1,2))

    # Rescale globally to [0, 255]
    xy_uint8 = rescale_intensity(xy, in_range=lims, out_range=(0, 255)).astype(np.uint8)
    xz_uint8 = rescale_intensity(xz, in_range=lims, out_range=(0, 255)).astype(np.uint8)
    yz_uint8 = rescale_intensity(yz, in_range=lims, out_range=(0, 255)).astype(np.uint8)

    # Get upsampled dimensions for each slice orientation
    xy_w, xy_h = up_dims(xy_uint8)
    xz_w, xz_h = up_dims(xz_uint8)
    yz_w, yz_h = up_dims(yz_uint8)

    # Slice specific ffmpeg parameters
    xy_pars = ['-vf', f'scale={xy_w}:{xy_h}:flags=neighbor']
    xz_pars = ['-vf', f'scale={xz_w}:{xz_h}:flags=neighbor']
    yz_pars = ['-vf', f'scale={yz_w}:{yz_h}:flags=neighbor']

    # Save movies
    imageio.mimwrite(img_stub + '_xy.mp4', xy_uint8, fps=24, output_params=xy_pars)
    imageio.mimwrite(img_stub + '_xz.mp4', xz_uint8, fps=24, output_params=xz_pars)
    imageio.mimwrite(img_stub + '_yz.mp4', yz_uint8, fps=24, output_params=yz_pars)


def up_dims(im):

    max_dim = 1024

    nt, h, w = im.shape

    if w > h:
        w_up = max_dim
        h_up = int((h * max_dim/w)/16) * 16
    else:
        h_up = max_dim
        w_up = int((w * max_dim/h)/16) * 16

    print(f'Resampling from {w} x {h} to {w_up} x {h_up} ')

    return w_up, h_up


if '__main__' in __name__:

    main()