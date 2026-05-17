from unittest.mock import patch

import numpy as np
from PIL import Image

from image_merger import ImageMerger


def _img(height, width=4):
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@patch.object(ImageMerger, "validate_overlap_robust", return_value=False)
@patch.object(ImageMerger, "compute_overlap_offset", return_value=(60, 140, 110, 24))
@patch.object(ImageMerger, "_calculate_static_bounds", return_value=(8, 6))
def test_failed_match_exposes_candidate_footprint_and_probe_band(
    _static_mock, _offset_mock, _validate_mock
):
    merged, meta = ImageMerger.merge_images_vertically(_img(400), _img(200), tolerance=20.0)

    assert merged.height == 400
    assert meta["match_status"] == "candidate"
    assert meta["matched_region_start"] == 60
    assert meta["matched_region_end"] == 260
    assert meta["overlap_visual_start"] == 110
    assert meta["overlap_visual_end"] == 134


@patch.object(ImageMerger, "validate_overlap_robust", return_value=True)
@patch.object(ImageMerger, "compute_overlap_offset", return_value=(60, 140, 110, 24))
@patch.object(ImageMerger, "_calculate_static_bounds", return_value=(8, 6))
def test_successful_merge_exposes_full_newest_screenshot_footprint_and_probe_band(
    _static_mock, _offset_mock, _validate_mock
):
    merged, meta = ImageMerger.merge_images_vertically(_img(400), _img(200), tolerance=20.0)

    assert merged.height == 260
    assert meta["match_status"] == "merged"
    assert meta["matched_region_start"] == 60
    assert meta["matched_region_end"] == 260
    assert meta["overlap_visual_start"] == 110
    assert meta["overlap_visual_end"] == 134
