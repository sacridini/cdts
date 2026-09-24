#pragma once
#include <cstdint>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// SNIC - Simple Non-Iterative Clustering superpixels (Achanta & Susstrunk,
// "Superpixels and Polygons using Simple Non-Iterative Clustering", CVPR 2017),
// implemented in C++/Eigen/OpenMP for multiband images and satellite image
// time series: every (time, band) pair is one feature channel, as sits_snic()
// does through the R `snic` package.
//
// From given seeds it produces the same labels as the authors' reference
// implementation (tests/test_snic.py checks this against its outputs): each
// cluster keeps running sums, so the distance of pixel x to cluster k with n
// pixels is evaluated with a single division,
//     d = (sum_c (S_c - n*x_c)^2 + ((Sx - n*x)^2 + (Sy - n*y)^2) * M^2*K/N) / n^2,
// equal to ||c_k - x||^2 + (M/S)^2 ||p_k - p||^2 on the means with S^2 = N/K;
// the min-heap sifts with strict comparisons (left child on ties), which
// decides which cluster wins a pixel two clusters reach at the same cost; and
// the 4-neighbours are visited left, up, right, down.
//
// Pixels with a NaN in any channel are masked out (label -1) and N counts
// valid pixels only, as in the R `snic` package; a seed on a masked pixel
// yields an empty segment.

namespace cdts {
namespace snic {

// Segment a planar image data[F, H, W] (float32 or float64) from seeds[K, 2]
// = (row, col). The image is split into tiles of tile_height x tile_width
// (<= 0 means the full extent) that are segmented independently and in
// parallel, each seed belonging to the tile that contains it - the block-wise
// strategy of sits_segment(). With a single tile the result is exactly SNIC on
// the whole image.
//
// Returns (labels[H, W] int32 with the seed index or -1, means[K, F] float64,
// centroids[K, 2] float64 (row, col), sizes[K] int64). Segments whose seed
// fell on a masked pixel have size 0 and NaN mean/centroid.
pybind11::tuple snic_segment(
    pybind11::array data,
    pybind11::array_t<std::int32_t, pybind11::array::c_style | pybind11::array::forcecast> seeds,
    double compactness = 10.0,
    int tile_height = 0,
    int tile_width = 0,
    int n_jobs = -1);

} // namespace snic
} // namespace cdts
