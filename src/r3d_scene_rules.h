#pragma once
#ifndef CATA_SRC_R3D_SCENE_RULES_H
#define CATA_SRC_R3D_SCENE_RULES_H

#include <cstdint>
#include <optional>

#include "coordinates.h"
#include "type_id.h"

class avatar;
class map;

namespace r3d
{

/// Upper storeys must not cover the interior currently visible on the viewed level.
auto hide_overhead_tile( const map &here, const tripoint_bub_ms &p, int view_z ) -> bool;

struct field_render_data {
    field_type_id id;
    /// Same south/east/west/north connection bits used by the ordinary tile renderer.
    uint8_t connections = 0;
    bool display_field = false;
    bool display_items = true;
};

/// Dynamic fields are drawn only where the player currently sees, never from terrain memory.
auto visible_field_at( const map &here, const avatar &you,
                       const tripoint_bub_ms &p ) -> std::optional<field_render_data>;

} // namespace r3d

#endif // CATA_SRC_R3D_SCENE_RULES_H
