#pragma once
#ifndef CATA_SRC_R3D_H
#define CATA_SRC_R3D_H

// Public entry points of the 3D renderer. No SDL types here so game code can include it in
// every build; without TILES everything is an inline no-op.

#include <optional>

#include  <string>
#include <vector>

#include "coordinates.h"



#if defined(TILES)
struct SDL_Renderer;
#endif

namespace r3d
{

#if defined(TILES)

/// Draws the map around center into the pixel rectangle (x, y, w, h) of the renderer's current
/// target, composited like any other 2D draw. Returns false when the 3D view cannot be drawn this
/// frame (option off, no GPU renderer, ...), in which case the caller draws the flat view instead.
/// render_epoch changes whenever the renderer or its textures were rebuilt.
struct draw_options {
    SDL_Renderer *renderer;
    float x;
    float y;
    float w;
    float h;
    const tripoint_bub_ms &center;
    unsigned int render_epoch;
};

auto draw( const draw_options &options ) -> bool;

/// Releases all GPU objects; call before the renderer goes away.
auto shutdown() -> void;

/// True when the option is on, the GPU renderer is available and the 3D scene is drawn.


/// A label of the first-person overlay, for the caller to draw after draw() in the game's own
/// font so it matches the sidebar. x, y in the pixels given to draw(); y is the top of the text;
/// color is a curses colour index of the font's palette.
struct hud_text {
    float x = 0.0f;
    float y = 0.0f;
    std::string text;
    unsigned char color = 15;
    bool centred = false; // x is the middle of the text rather than its left end
};
/// The size of a character cell of the font the overlay's labels are drawn in.
auto set_hud_font( int cell_width, int cell_height ) -> void ;
/// Labels of the last drawn first-person overlay; empty when draw() drew none.
auto hud_texts() -> const std::vector<hud_text> &; // *NOPAD*

/// True when the option is on, the GPU renderer is available and the 3D scene is drawn.
auto active() -> bool;

/// The map tile under the point (x, y) of the renderer, in the same logical pixels as the
/// rectangle given to draw(), as seen in the last drawn 3D frame. Like the flat view's picks, the
/// result is on the map's own level (get_abs_sub().z()): something seen through a hole on a lower
/// level yields the tile where the mouse ray crossed the viewed level. Nothing when the point is
/// outside that frame, points at the sky, or the 3D view is not active.
auto pick( float x, float y ) -> std::optional<tripoint_bub_ms>;

/// 1/8 turns clockwise to apply to map directions so that "north" walks away from the 3D camera
/// (see camera::forward_eighths); 0 when the 3D view is not shown.
auto view_eighths() -> int;
/// True when the main game takes WASD for walking and the arrow keys for the 3D camera.
auto wasd_controls() -> bool;

/// Debug capture for CATA_R3D_COMPARE (see r3d.cpp): saves the 3D view and the flat isometric
/// view of the same turn once. CATA_R3D_COMPARE_SESSION enables bounded periodic checkpoints
/// with eight 3D yaw poses and current saves. Sessions require gameplay_boundary=true;
/// the generic input hook can still perform the legacy one-shot without saving.
auto compare_capture( bool gameplay_boundary ) -> void;

/// Flips the RENDERER_3D option.
auto  fp_shots_capture() -> void ;

/// Flips the RENDERER_3D option.
auto toggle() -> void ;
/// Switches between the orbit and the first-person camera (the R3D_CAMERA_MODE option).
auto toggle_first_person() -> void ;
/// True when the first-person camera is what the last drawn 3D frame showed.
auto first_person() -> bool ;
/// Ends a mouse glance: the first-person view snaps back to the heading.
auto end_glance() -> void ;
/// First person: a step or turn is still being eased in; the game should redraw soon.
auto animating() -> bool ;
/// First person: the tile under the centre of the picture, nothing when it shows no tile.
auto crosshair() -> std::optional<tripoint_bub_ms> ;
/// First person: turns to face a tile dx east and dy south of the player (no game time).
auto face_towards( int dx, int dy ) -> void;
/// Orbit the camera by one step; dir < 0 is left, dir > 0 right.
auto rotate( int dir ) -> void;
/// Tilt the camera by one step; dir > 0 raises the view towards top-down.
auto pitch( int dir ) -> void;
/// Move the camera closer (dir > 0) or away (dir < 0).
auto zoom( int dir ) -> void;
/// Restore the default camera, centred on the view again.
auto reset() -> void;
/// True when (x, y), in the logical pixels given to draw(), lies on the last drawn 3D frame.
auto contains( float x, float y ) -> bool;
/// Free camera from a mouse drag of (dx, dy) pixels: orbit and tilt, or slide over the ground.
auto drag_orbit( float dx, float dy ) -> void;
auto drag_pan( float dx, float dy ) -> void;

#else

inline auto active() -> bool { return false; }
inline auto view_eighths() -> int { return 0; }
inline auto wasd_controls() -> bool { return false; }
inline auto compare_capture( bool /*gameplay_boundary*/ ) -> void {}
inline auto toggle() -> void {}
inline auto  toggle_first_person() -> void {}
inline auto first_person() -> bool
{ return false; }
inline auto end_glance() -> void {}
inline auto animating() -> bool
{ return false; }
inline auto crosshair() -> std::optional<tripoint_bub_ms>
{ return std::nullopt; }
inline auto face_towards( int /*dx*/, int /*dy*/ ) -> void {}
inline auto rotate( int /*direction*/ ) -> void {}
inline auto pitch( int /*direction*/ ) -> void {}
inline auto zoom( int /*direction*/ ) -> void {}
inline auto reset() -> void {}
inline auto contains( float /*x*/, float /*y*/ ) -> bool { return false; }
inline auto drag_orbit( float /*x*/, float /*y*/ ) -> void {}
inline auto drag_pan( float /*x*/, float /*y*/ ) -> void {}

#endif

} // namespace r3d

#endif // CATA_SRC_R3D_H
