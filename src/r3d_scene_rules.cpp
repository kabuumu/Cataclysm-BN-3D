#include "r3d_scene_rules.h"

#include <array>

#include "avatar.h"
#include "map/field.h"
#include "map/map.h"
#include "map/mapdata.h"

namespace r3d
{

auto hide_overhead_tile( const map &here, const tripoint_bub_ms &p, int view_z ) -> bool
{
    if( p.z() <= view_z || !here.inbounds( p ) ) {
        return false;
    }
    const auto below = tripoint_bub_ms( p.xy(), view_z );
    if( !here.inbounds( below ) || here.is_outside( below ) ) {
        return false;
    }
    const auto &cache = here.access_cache( view_z );
    const auto vis = here.get_visibility( cache.visibility_cache[cache.idx( below.x(), below.y() )],
                                          here.get_visibility_variables_cache() );
    return vis == VIS_CLEAR || vis == VIS_LIT;
}

auto visible_field_at( const map &here, const avatar &you,
                       const tripoint_bub_ms &p ) -> std::optional<field_render_data>
{
    if( !here.inbounds( p ) ) {
        return std::nullopt;
    }
    const auto id = here.field_at( p ).displayed_field_type();
    if( id == fd_null || !you.sees( p ) ) {
        return std::nullopt;
    }
    auto out = field_render_data{ .id = id, .display_field = id->display_field,
                                  .display_items = id->display_items };
    const auto neighbors = std::array{ point_rel_ms::south(), point_rel_ms::east(),
                                       point_rel_ms::west(), point_rel_ms::north() };
    for( auto i = 0u; i < neighbors.size(); ++i ) {
        const auto n = p + neighbors[i];
        if( here.inbounds( n ) && here.field_at( n ).displayed_field_type() == id ) {
            out.connections |= 1u << i;
        }
    }
    return out;
}

} // namespace r3d
