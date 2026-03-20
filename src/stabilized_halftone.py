import cv2
import numpy as np

from halftone import (
    preprocess_gray,
    radius_from_intensity,
    sample_gray_at_points,
    generate_frame_dots,
    clone_dot_state,
    render_dots,
)
from optical_flow import compute_flow, sample_flow


def update_dot_state(
    prev_state,
    prev_gray,
    curr_gray,
    flow=None,
    grid_size=8,
    blur_ksize=5,
    max_radius_ratio=0.45,
    radius_alpha=0.6,
    drift_limit_cells=2.0,
    snap_strength=0.10,
    flow_params=None,
):
    """
    Advance the persistent dot field from prev_gray to curr_gray.

    Steps:
    1. compute flow if not provided
    2. move each dot by the flow vector at its current location
    3. resample current intensity at the moved location
    4. update the radius with temporal smoothing
    5. lightly pull back dots that drift too far from their home cell
    """
    if flow is None:
        prev_smooth = preprocess_gray(prev_gray, blur_ksize=blur_ksize)
        curr_smooth = preprocess_gray(curr_gray, blur_ksize=blur_ksize)
        flow = compute_flow(prev_smooth, curr_smooth, params=flow_params)

    state = clone_dot_state(prev_state)
    h, w = curr_gray.shape

    # 1. Advect positions using dense flow
    dx, dy = sample_flow(flow, state["x"], state["y"])
    state["x"] = state["x"] + dx
    state["y"] = state["y"] + dy

    state["x"] = np.clip(state["x"], 0, w - 1)
    state["y"] = np.clip(state["y"], 0, h - 1)

    # 2. Update radii using current local intensity
    curr_smooth = preprocess_gray(curr_gray, blur_ksize=blur_ksize)
    intensities = sample_gray_at_points(curr_smooth, state["x"], state["y"])
    target_r = radius_from_intensity(
        intensities,
        grid_size=grid_size,
        max_radius_ratio=max_radius_ratio
    )

    # Temporal smoothing on radius
    # radius_alpha = weight on the old radius
    state["r"] = radius_alpha * state["r"] + (1.0 - radius_alpha) * target_r

    # 3. Drift regularization
    # Dots can collapse or wander in low-texture or unreliable-flow regions.
    # If a dot moves too far from its original cell, pull it back a little.
    max_drift = drift_limit_cells * grid_size
    offset_x = state["x"] - state["home_x"]
    offset_y = state["y"] - state["home_y"]
    drift = np.sqrt(offset_x ** 2 + offset_y ** 2)

    mask = drift > max_drift
    state["x"][mask] = (1.0 - snap_strength) * state["x"][mask] + snap_strength * state["home_x"][mask]
    state["y"][mask] = (1.0 - snap_strength) * state["y"][mask] + snap_strength * state["home_y"][mask]

    return state, flow


def render_baseline_sequence(
    frames,
    grid_size=8,
    blur_ksize=5,
    max_radius_ratio=0.45,
):
    """
    Per-frame baseline: regenerate the dot field from scratch every frame.
    This is the correct comparison point for the stabilized pipeline.
    """
    outputs = []
    grays = []

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        state = generate_frame_dots(
            gray,
            grid_size=grid_size,
            blur_ksize=blur_ksize,
            max_radius_ratio=max_radius_ratio
        )
        rendered = render_dots(state, gray.shape)
        outputs.append(rendered)
        grays.append(gray)

    return outputs, grays


def render_stabilized_sequence(
    frames,
    grid_size=8,
    blur_ksize=5,
    max_radius_ratio=0.45,
    radius_alpha=0.6,
    drift_limit_cells=2.0,
    snap_strength=0.10,
    flow_params=None,
    blend_alpha=0.7,
):
    """
    Stabilized version with correction:
    1. warp the persistent dot field forward using optical flow
    2. render that warped result
    3. generate a fresh dot rendering from the current frame
    4. blend warped + fresh
    5. threshold back to a binary halftone

    blend_alpha controls the tradeoff:
    - higher = more temporal stability
    - lower = more faithful to the current frame
    """
    outputs = []
    grays = []

    if not frames:
        return outputs, grays

    first_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    state = generate_frame_dots(
        first_gray,
        grid_size=grid_size,
        blur_ksize=blur_ksize,
        max_radius_ratio=max_radius_ratio
    )

    first_render = render_dots(state, first_gray.shape)
    outputs.append(first_render)
    grays.append(first_gray)

    prev_gray = first_gray

    for frame in frames[1:]:
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Move the persistent dot state forward
        state, _ = update_dot_state(
            prev_state=state,
            prev_gray=prev_gray,
            curr_gray=curr_gray,
            flow=None,
            grid_size=grid_size,
            blur_ksize=blur_ksize,
            max_radius_ratio=max_radius_ratio,
            radius_alpha=radius_alpha,
            drift_limit_cells=drift_limit_cells,
            snap_strength=snap_strength,
            flow_params=flow_params,
        )

        # 2. Render the flow-advected dot field
        warped_render = render_dots(state, curr_gray.shape)

        # 3. Generate a fresh per-frame dot field from the current frame
        fresh_state = generate_frame_dots(
            curr_gray,
            grid_size=grid_size,
            blur_ksize=blur_ksize,
            max_radius_ratio=max_radius_ratio
        )
        fresh_render = render_dots(fresh_state, curr_gray.shape)

        # 4. Blend warped + fresh
        blended = cv2.addWeighted(warped_render, blend_alpha, fresh_render, 1.0 - blend_alpha, 0)

        # 5. Threshold back to binary
        _, stabilized = cv2.threshold(blended, 127, 255, cv2.THRESH_BINARY)

        outputs.append(stabilized)
        grays.append(curr_gray)
        prev_gray = curr_gray

    return outputs, grays