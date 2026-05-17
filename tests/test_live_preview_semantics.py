from live_preview import compute_semantic_regions


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
