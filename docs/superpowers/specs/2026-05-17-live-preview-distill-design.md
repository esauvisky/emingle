# Live Preview Distillation Design

## Goal

Refocus Emingle's live preview so it answers the one question that matters during scrolling capture:

"Where is the relevant screenshot relative to the bottom of the stitched image?"

The preview should stop behaving like a debug surface and start behaving like a compact reading instrument.

## Scope

This design covers:

- preview rendering semantics for successful and failed captures
- overlay simplification
- preview-local zoom behavior
- hover feedback behavior
- supporting metadata flow from merge logic into the preview

This design does not cover:

- capture algorithm changes beyond the metadata needed for rendering
- new persistence/export features
- status panel redesign beyond the hover text integration needed for this feature

## User Outcome

During a capture session, the user should be able to tell at a glance:

- where the newest relevant screenshot sits inside the stitched image
- which smaller region the matcher actually used for alignment
- where ignored static borders sit inside that screenshot footprint
- how far the visible screenshot is from the bottom

## Core Model

The preview uses a `footprint + probe` model.

### 1. Screenshot footprint

The primary visual unit is the full vertical footprint of the relevant screenshot inside the stitched image.

- On a successful merge:
  - the entire newest merged screenshot footprint is shown in color
  - older content remains desaturated
- On a failed-but-matched capture:
  - the entire candidate screenshot footprint is shown in color
  - older content, including the previous successful footprint, remains desaturated
- On a complete miss with no valid match:
  - keep the previous meaningful context rather than clearing the preview

This footprint is the strongest cue in the window.

### 2. Probe band

Inside the footprint, draw a narrower band representing the exact crop/probe the matcher used to align the images.

- This band is narrower than the full footprint
- It must reflect the actual probe region used by template matching
- It must not extend to the full overlap span unless that truly was the probe

This band answers:

"What did the model actually use to align the images?"

### 3. Static border markers

Static ignored borders should be shown only as subtle top and bottom edge markers inside the footprint.

- No large tinted slabs
- No full-width heavy fills
- Just quiet markers that explain ignored header/footer areas

## Visual Hierarchy

The preview should use a restrained three-layer hierarchy:

1. Desaturated stitched image background
2. Color screenshot footprint
3. Thin probe band and subtle static-border edge markers

Anything outside these layers should be removed or visually demoted.

## Color Semantics

Use a stable semantic mapping:

- `Color footprint`: the full screenshot currently relevant to the user's decision
- `Warm narrow probe band`: the exact crop the matcher used
- `Subtle edge markers`: ignored static top/bottom regions inside that footprint

Rules:

- Do not use panel-level green/red success or failure backgrounds
- Do not use multiple equally loud overlay colors at once
- Do not reuse the same strong color for different meanings

## Interaction Model

### Zoom

Add preview-local zoom:

- `Ctrl + mouse wheel` over the preview image zooms in and out
- plain wheel should not be repurposed, so it does not interfere with capture workflow expectations
- zoom affects preview rendering only, not the underlying stitched image

Zoom should:

- preserve the current tail/full rendering model
- stay bounded to a sane range
- prefer keeping the current focus area visible

### Hover feedback

Add hover interpretation using the status line, not tooltips.

When hovering the preview image, the status text should explain the semantic region under the cursor, for example:

- `hover: candidate screenshot footprint`
- `hover: matched probe used for alignment`
- `hover: ignored static top border`
- `hover: stitched history`

Rules:

- use the existing status area rather than floating tooltip UI
- hover text should disappear or revert when the pointer leaves the image
- hover should describe meaning, not raw coordinates by default

## Data Flow

### Merge metadata

The merge pipeline must provide enough metadata for the preview to render:

- screenshot footprint start/end inside the stitched image
- probe band start/end inside the stitched image
- ignored static top height
- ignored static bottom height
- match state:
  - `merged`
  - `candidate`
  - `none`

### Preview state

The preview must retain the last meaningful semantic context so a failed capture does not collapse the user's mental model.

It should track:

- last successful footprint
- last candidate footprint
- active probe band
- active static-border markers
- current zoom level
- current hover region

## Rendering Rules

### Successful merge

- desaturate the full stitched image
- color the full newest merged screenshot footprint
- draw the narrow probe band inside that footprint
- draw static border markers inside that footprint

### Failed-but-matched capture

- desaturate the full stitched image
- color the full candidate screenshot footprint
- do not keep the previous successful footprint colored
- draw the narrow probe band inside the candidate footprint
- draw static border markers inside the candidate footprint when relevant

### No-match failure

- keep the previous meaningful preview context
- do not clear the preview to an uninformative neutral state

## Error Handling

- If probe metadata is missing but footprint metadata exists, render only the footprint
- If static-border metadata is missing, omit the border markers
- If hover region mapping fails, fall back to normal status text
- If zoom state becomes invalid, reset to default zoom

## Testing

Manual validation should cover:

1. Successful merge:
   - newest full screenshot footprint is colored
   - narrower probe band is visible inside it
2. Failed-but-matched capture:
   - candidate footprint is colored
   - previous successful footprint is no longer colored
   - probe band is narrower than the full footprint
3. No-match capture:
   - previous meaningful context remains visible
4. Static borders:
   - top/bottom ignored regions appear only as subtle internal edge markers
5. Zoom:
   - `Ctrl + wheel` zooms preview only
   - plain wheel behavior does not conflict with normal usage
6. Hover:
   - status text changes to describe hovered semantic region
   - leaving the image restores normal status output

## Implementation Notes

- Keep the preview visually restrained
- Prefer semantics over decorative feedback
- When in doubt, make the screenshot footprint more legible rather than adding another overlay
