from live_preview import (
    clamp_preview_zoom,
    compute_semantic_regions,
    describe_preview_region,
    next_preview_zoom,
)


def test_success_prefers_full_newest_footprint_not_only_appended_tail():
    regions = compute_semantic_regions(
        image_height=900,
        render_success=True,
        latest_slice=(700, 900),
        matched_region=(560, 900),
        probe_region=(610, 634),
        static_top=8,
        static_bottom=6,
    )

    assert regions["footprint"] == (560, 900)
    assert regions["probe"] == (610, 634)
    assert regions["show_previous_success_footprint"] is False


def test_failed_candidate_uses_candidate_footprint_and_not_previous_success_slice():
    regions = compute_semantic_regions(
        image_height=900,
        render_success=False,
        latest_slice=(700, 900),
        matched_region=(520, 820),
        probe_region=(610, 634),
        static_top=8,
        static_bottom=6,
    )

    assert regions["footprint"] == (520, 820)
    assert regions["probe"] == (610, 634)
    assert regions["show_latest_slice_color"] is False


def test_zoom_is_clamped_to_sane_bounds():
    assert clamp_preview_zoom(0.05) == 0.25
    assert clamp_preview_zoom(5.0) == 3.0


def test_ctrl_wheel_steps_zoom_in_and_out():
    assert next_preview_zoom(1.0, wheel_rotation=120) > 1.0
    assert next_preview_zoom(1.0, wheel_rotation=-120) < 1.0


def test_hover_returns_probe_label_inside_probe_band():
    label = describe_preview_region(
        pointer_y=620,
        footprint=(560, 900),
        probe=(610, 634),
        static_top_band=(560, 568),
        static_bottom_band=(894, 900),
    )
    assert label == "matched probe used for alignment"


def test_hover_returns_history_outside_all_semantic_regions():
    label = describe_preview_region(
        pointer_y=200,
        footprint=(560, 900),
        probe=(610, 634),
        static_top_band=(560, 568),
        static_bottom_band=(894, 900),
    )
    assert label == "stitched history"
