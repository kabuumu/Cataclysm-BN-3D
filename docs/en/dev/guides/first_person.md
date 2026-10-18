# First-person renderer

Branch `feat/first-person` is based on this fork's 3D renderer main. It ports the
CDDA fork's first-person camera while leaving BN movement, visibility, targeting,
combat, turns, and saves in control of gameplay.

Enable **3D renderer**, then choose **3D camera: First person** in Graphics, or
press **Ctrl+F**. W/A/S/D move relative to facing; arrows turn by 45 degrees or
look up/down. Examine keeps its normal binding. Middle drag glances; releasing
returns to the saved heading. `/` selects the tile ahead in direction prompts
and the crosshair tile in look and targeting. **Ctrl+G** turns
the view toward a visible hostile without spending a turn. BN uses Ctrl+G because its input names do not encode Ctrl+Tab.

The port includes upright eye-level sprites, ceilings, sky, unexplored geometry
that occludes known rooms, compass and threat alerts, crosshair labels, sound
markers, optional step/turn easing, saved heading, and a circular heading-up
minimap. All information comes from BN's existing visibility and event state.
The ordinary orbit camera remains selectable. Graphics options control field of
view, eye height, easing, compass, edge alerts, minimap rotation, and occlusion.

Comparison sessions retain the player's camera mode and heading and temporarily
use the orbit camera for their standard eight-angle reference shots. Private
save copies, SQLite backups, diagnostics, and report packaging work as on main.

Build using the main renderer guide. Native macOS and Windows execution has not
been verified. CDDA's original feature notes are retained in its source branch;
this guide describes the BN implementation.
