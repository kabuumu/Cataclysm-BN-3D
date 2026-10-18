#if defined(TILES)

#include "r3d_entities.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "avatar.h"
#include "debug.h"
#include "cata_tiles.h"
#include "character.h"
#include "creature.h"
#include "game.h"
#include "item.h"
#include "map/map.h"
#include "monster.h"
#include "mtype.h"
#include "npc.h"
#include "r3d_shapes.h"
#include "r3d_scene_rules.h"
#include "r3d_decals.h"
#include "enums.h"
#include "units_utility.h"
#include "vehicle/vehicle.h"
#include "vehicle/veh_type.h"
#include "vehicle/vpart_position.h"

namespace r3d
{

namespace
{

constexpr auto creature_radius = static_cast<int>
                                 ( 45 ); // tiles around the view that are searched for creatures
constexpr auto item_radius = static_cast<int>
                             ( 20 );     // and for items (every tile costs a map lookup)
constexpr auto marker_box_height = static_cast<float>( 1.0f );
constexpr auto item_size = static_cast<float>( 0.6f );
constexpr auto vehicle_body_height = static_cast<float>( 0.25f );

// Light colour with a zero alpha: no depth pull (see r3d_block.vert), for everything in the
// entity mesh that is not the top of a billboard.
auto no_pull( uint32_t color ) -> uint32_t
{ return color & 0x00ffffffu; }

// Geometry and depth shared by the sprite layers of one billboard.
struct billboard_geometry {
    quad_sink &sink;
    const vec3 &right_world;
    const vec3 &up_world;
    float scale;
    float z_scale;
    float tx;
    float ty;
    float tz;
    float cx;
    float cy;
    float z_base;
    bool mirrored;
    int depth;
    const std::array<uint32_t, 4> &colors;

    auto emit_layer( const sprite_ref &s ) -> void {
        if( !s.valid ) {
            return;
        }
        const auto lw = static_cast<float>( std::clamp( s.size_x, 0.25f, 4.0f ) * scale );
        const auto lh = static_cast<float>( std::clamp( s.size_y, 0.25f, 6.0f ) * scale );
        const auto rx = static_cast<float>( right_world.x * lw * 0.5f );
        const auto ry = static_cast<float>( -right_world.y * lw * 0.5f );
        const auto ux = static_cast<float>( up_world.x * lh );
        const auto uy = static_cast<float>( -up_world.y * lh );
        const auto uz = static_cast<float>( up_world.z * lh / z_scale );
        // The flat view draws a sprite offset_y pixels below its tile's top edge, so its bottom
        // row lands (offset_y + height - tile) below the tile's bottom: the same shift here,
        // along the screen's up, keeps a 48-row body and 36-row clothes on the same feet.
        const auto drop = static_cast<float>( ( s.offset_y + s.size_y - 1.0f ) * scale );
        const auto side = static_cast<float>( s.offset_x * scale );
        const auto lift = static_cast<float>( 0.01f * static_cast<float>( depth++ ) );
        const auto ox = static_cast<float>( cx + tx * lift - up_world.x * drop + right_world.x * side );
        const auto oy = static_cast<float>( cy + ty * lift + up_world.y * drop - right_world.y * side );
        const auto oz = static_cast<float>( z_base + tz * lift - up_world.z * drop / z_scale );
        const auto ul = static_cast<float>( mirrored ? s.u1 : s.u0 );
        const auto ur = static_cast<float>( mirrored ? s.u0 : s.u1 );
        const auto uv = uvs_t{{ { ul, s.v0 }, { ur, s.v0 }, { ur, s.v1 }, { ul, s.v1 } }};
        const auto c = corners_t{{ { ox - rx + ux, oy - ry + uy, oz + uz }, { ox + rx + ux, oy + ry + uy, oz + uz },
                { ox + rx, oy + ry, oz }, { ox - rx, oy - ry, oz }
            }
        };
        sink.add_colored( { .sprite = s, .corners = c, .uvs = uv, .ref = light_ref{}, .colors = colors } );
    }
};

// Camera-facing upright quads standing on the tile centre, sized like the first layer's sprite
// is in the flat view (big trees are several tiles tall). Later layers (clothing over a body) sit
// slightly closer to the camera. Corners are in scene-frame tile space.
struct emit_billboard_options {
    quad_sink &sink;
    const std::vector<const tile_info *> &layers;
    const vec3 &right_world;
    const vec3 &up_world;
    const vec3 &toward_world;
    float z_scale;
    float cx;
    float cy;
    float z_base;
    uint32_t color;
    bool mirrored = false;
    float scale = 1.0f;
};

auto emit_billboard( const emit_billboard_options &options ) -> void
{
    auto &sink = options.sink;
    const auto &layers = options.layers;
    const auto &right_world = options.right_world;
    const auto &up_world = options.up_world;
    const auto &toward_world = options.toward_world;
    auto z_scale = options.z_scale;
    auto cx = options.cx;
    auto cy = options.cy;
    auto z_base = options.z_base;
    auto color = options.color;
    auto mirrored = options.mirrored;
    auto scale = options.scale;

    if( layers.empty() || !layers.front() ) {
        return;
    }
    const auto &main = layers.front()->main();
    const auto height = static_cast<float>( std::clamp( main.size_y, 0.25f, 6.0f ) * scale );
    // The camera vectors are in the right-handed world frame; tile space has y flipped.
    const auto tx = static_cast<float>( toward_world.x );
    const auto ty = static_cast<float>( -toward_world.y );
    // The camera vectors are in stretched space, where sprites keep their flat-view proportions;
    // their vertical parts shrink back into world space, where a z-level is 1 high.
    const auto tz = static_cast<float>( toward_world.z / z_scale );
    // Facing the camera squarely leans the top away from it, into whatever stands behind the
    // tile (a wall). Sliding the quad towards the camera along the line of sight keeps it where
    // it is on screen but brings the top back inside the tile's own footprint.
    const auto lean = static_cast<float>( height * std::sqrt( up_world.x * up_world.x + up_world.y *
                                          up_world.y ) );
    const auto toward_ground = static_cast<float>( std::sqrt( toward_world.x * toward_world.x +
                               toward_world.y * toward_world.y ) );
    constexpr auto half_tile = static_cast<float>( 0.45f );
    const auto pull = static_cast<float>( std::clamp( ( lean - half_tile ) / std::max( toward_ground,
                                          0.05f ), 0.0f,
                                          max_billboard_pull ) );
    // The pull only moves depth (see r3d_block.vert): the top's light alpha carries it, the base
    // keeps 0 so feet stay where they stand.
    const auto top_alpha = static_cast<uint32_t>( static_cast<uint32_t>( pull / max_billboard_pull *
                           255.0f + 0.5f ) << 24 );
    const auto colors = std::array<uint32_t, 4> {{
            ( color & 0x00ffffffu ) | top_alpha, ( color & 0x00ffffffu ) | top_alpha,
            color & 0x00ffffffu, color & 0x00ffffffu
        }
    };
    // The quad faces the camera squarely (leaning back with its up vector), so sprites keep the
    // full size and detail they have in the flat view instead of being squashed by the pitch.
    // Each layer keeps its own sprite's size, standing on the same feet: a 48-row body under
    // 36-row clothing, as the flat view draws them.
    auto geometry = billboard_geometry{ .sink = sink, .right_world = right_world, .up_world = up_world, .scale = scale, .z_scale = z_scale, .tx = tx, .ty = ty, .tz = tz,
                                        .cx = cx, .cy = cy, .z_base = z_base, .mirrored = mirrored, .depth = 0, .colors = colors };
    for( const tile_info *t : layers ) {
        if( t ) {
            if( !t->sh.ground_backdrop ) {
                geometry.emit_layer( t->backdrop );
            }
            geometry.emit_layer( t->sprite );
        }
    }
}

// Sprite lying flat at height z, size tiles wide.
struct emit_flat_options {
    quad_sink &sink;
    const tile_info &sprite;
    float cx;
    float cy;
    float z;
    uint32_t color;
    float size = item_size;
};

auto emit_flat( const emit_flat_options &options ) -> void
{
    auto &sink = options.sink;
    const auto &sprite = options.sprite;
    auto cx = options.cx;
    auto cy = options.cy;
    auto z = options.z;
    auto color = options.color;
    auto size = options.size;

    const auto h = static_cast<float>( size * 0.5f );
    const auto c = corners_t{{ { cx - h, cy - h, z }, { cx + h, cy - h, z }, { cx + h, cy + h, z },
            { cx - h, cy + h, z }
        }
    };
    for( const sprite_ref *s : {
             &sprite.backdrop, &sprite.sprite
         } ) {
        if( !s->valid ) {
            continue;
        }
        const auto uv = uvs_t{{ { s->u0, s->v0 }, { s->u1, s->v0 }, { s->u1, s->v1 }, { s->u0, s->v1 } }};
        sink.add_colored( { .sprite = *s, .corners = c, .uvs = uv, .ref = light_ref{}, .color = no_pull( color ) } );
    }
}

// Box standing on a tile with the sprite on its four sides, so a cursor reads as a 3D outline.
struct emit_marker_box_options {
    quad_sink &sink;
    const sprite_ref &s;
    float x0;
    float y0;
    float z0;
    float z1;
    uint32_t color;
    float width = 1.0f;
};

auto emit_marker_box( const emit_marker_box_options &options ) -> void
{
    auto &sink = options.sink;
    const auto &s = options.s;
    auto x0 = options.x0;
    auto y0 = options.y0;
    auto z0 = options.z0;
    auto z1 = options.z1;
    auto color = options.color;
    auto width = options.width;

    const auto x1 = static_cast<float>( x0 + width );
    const auto y1 = static_cast<float>( y0 + width );
    const auto uv = uvs_t{{ { s.u0, s.v0 }, { s.u1, s.v0 }, { s.u1, s.v1 }, { s.u0, s.v1 } }};
    const auto sides = std::array<std::array<float, 4>, 4> {{
            { x0, y1, x1, y1 }, { x1, y0, x0, y0 }, { x1, y1, x1, y0 }, { x0, y0, x0, y1 }
        }
    };
    for( const std::array<float, 4> &e : sides ) {
        const auto c = corners_t{{ { e[0], e[1], z1 }, { e[2], e[3], z1 }, { e[2], e[3], z0 },
                { e[0], e[1], z0 }
            }
        };
        sink.add_colored( { .sprite = s, .corners = c, .uvs = uv, .ref = light_ref{}, .color = no_pull( color ) } );
    }
}


// Vehicle art is a plan view. Keep it in the map plane, raised on a shallow body,
// rather than turning every part towards the camera independently.
struct emit_vehicle_part_options {
    quad_sink &sink;
    const tile_info &info;
    float x;
    float y;
    float z;
    uint32_t color;
};

auto emit_vehicle_part( const emit_vehicle_part_options &options ) -> void
{
    auto &sink = options.sink;
    const auto &info = options.info;
    auto x = options.x;
    auto y = options.y;
    auto z = options.z;
    auto color = options.color;

    auto lift = static_cast<float>( 0.0f );
    for( const sprite_ref *sprite : {
             &info.backdrop, &info.sprite
         } ) {
        if( !sprite->valid ) {
            continue;
        }
        const auto &s = *sprite;
        // Vehicle tiles can overhang their cell (doors and windscreens often use
        // 32x48/64 artwork). Preserve the regular renderer's size and offsets.
        const auto x0 = static_cast<float>( x + s.offset_x );
        const auto y0 = static_cast<float>( y + s.offset_y );
        auto top = corners_t{{ { x0, y0, z + vehicle_body_height + lift },
                { x0 + s.size_x, y0, z + vehicle_body_height + lift },
                { x0 + s.size_x, y0 + s.size_y, z + vehicle_body_height + lift },
                { x0, y0 + s.size_y, z + vehicle_body_height + lift }
            }
        };
        for( auto &corner : top ) {
            for( auto turn = static_cast<int>( 0 ); turn < ( s.quarter_turns & 3 ); ++turn ) {
                const auto dx = static_cast<float>( corner[0] - x - 0.5f );
                const auto dy = static_cast<float>( corner[1] - y - 0.5f );
                corner[0] = x + 0.5f - dy;
                corner[1] = y + 0.5f + dx;
            }
        }
        const auto left = static_cast<float>( s.flip ? s.u1 : s.u0 );
        const auto right = static_cast<float>( s.flip ? s.u0 : s.u1 );
        const auto uv = uvs_t{{ { left, s.v0 }, { right, s.v0 }, { right, s.v1 }, { left, s.v1 } }};
        sink.add_colored( { .sprite = s, .corners = top, .uvs = uv, .ref = light_ref{}, .color = no_pull( color ) } );
        emit_marker_box( { .sink = sink, .s = s, .x0 = x - lift, .y0 = y - lift, .z0 = z, .z1 = z + vehicle_body_height + lift, .color = color,
                           .width = 1.0f + 2.0f * lift } );
        // The foreground must pass the depth test over the background on every face.
        lift += 0.004f;
    }
}

struct add_vehicles_options {
    quad_sink &sink;
    const entity_view &view;
    map &here;
    const avatar &you;
};

auto add_vehicles( const add_vehicles_options &options ) -> void
{
    auto &sink = options.sink;
    const auto &view = options.view;
    auto &here = options.here;
    const auto &you = options.you;

    for( const wrapped_vehicle &wrapped : here.get_vehicles() ) {
        auto &veh = *wrapped.v;

        for( const tripoint_abs_ms &absolute : veh.get_points() ) {
            const auto p = tripoint_bub_ms( abs_to_bub( absolute ) );
            if( !here.inbounds( p ) || p.z() < view.z_lo || p.z() > view.z_hi ||
                !you.sees( p ) ) {
                continue;
            }
            const auto light = static_cast<uint32_t>( view.light( p ) );
            if( light == pack_light( 0, 0, 0 ) ) {
                continue;
            }
            const auto part = optional_vpart_position( here.veh_at( p ) );
            if( !part || &part->vehicle() != &veh ) {
                continue;
            }
            auto modifier = char{};
            auto part_index = part->part_index();
            const auto &id = veh.part_id_string( part_index, false, modifier );
            if( id.is_null() ) { continue; }
            const auto subtile = modifier == 1 ? open_ : modifier == 2 ? broken : 0;
            const auto facing = tileray( veh.part_display_direction( part_index, false ) );
            const auto rotation = 3 - facing.dir4();
            const auto &sprite = view.cache.named( { .category = C_VEHICLE_PART, .id = "vp_" + id.str(), .tiles = view.tiles, .subtile = subtile,
                                                   .rota = rotation } );
            const auto x = static_cast<float>( p.x() - view.center.x() );
            const auto y = static_cast<float>( p.y() - view.center.y() );
            const auto z = static_cast<float>( static_cast<float>( p.z() - view.center.z() ) + view.surface(
                                                   p ) + 0.02f );
            emit_vehicle_part( { .sink = sink, .info = sprite, .x = x, .y = y, .z = z, .color = light } );
            const auto cargo = part.part_with_feature( "CARGO", true );
            if( cargo && !veh.get_items( cargo->part_index() ).empty() ) {
                const auto &highlight = view.cache.named( { .category = C_NONE, .id = "highlight", .tiles = view.tiles } );
                emit_flat( { .sink = sink, .sprite = highlight, .cx = x + 0.5f, .cy = y + 0.5f, .z = z + vehicle_body_height + 0.02f,
                             .color = pack_light( 255, 255, 255 ), .size = 1.0f } );
            }
        }
    }
}

} // namespace

auto build_entity_mesh( const entity_view &view ) -> chunk_mesh
{
    auto &here = get_map();
    const auto &you = get_avatar();
    auto sink = quad_sink{};
    const auto right = vec3( view.cam.ground_right() );
    // Camera vectors in the stretched space the picture is in (see camera::z_scale).
    const auto toward = vec3( normalize( view.cam.eye() - view.cam.stretched_target() ) );
    const auto up = vec3( cross( right, vec3{ .x = 0.0f, .y = 0.0f, .z = 0.0f } - toward ) );
    const auto white = static_cast<uint32_t>( pack_light( 255, 255, 255 ) );

    auto in_range = [ &view ]( const tripoint_bub_ms & p, int radius ) -> bool {
        return p.z() >= view.z_lo && p.z() <= view.z_hi &&
        std::abs( p.x() - view.center.x() ) <= radius &&
        std::abs( p.y() - view.center.y() ) <= radius;
    };
    auto frame_x = [ &view ]( const tripoint_bub_ms & p ) -> float {
        return static_cast<float>( p.x() - view.center.x() ) + 0.5f;
    };
    auto frame_y = [ &view ]( const tripoint_bub_ms & p ) -> float {
        return static_cast<float>( p.y() - view.center.y() ) + 0.5f;
    };
    auto frame_z = [ &view ]( const tripoint_bub_ms & p ) -> float {
        return static_cast<float>( p.z() - view.center.z() );
    };

    add_vehicles( { .sink = sink, .view = view, .here = here, .you = you } );

    // Fields and items change independently of the terrain mesh, including between turns.
    for( auto z = static_cast<int>( view.z_lo ); z <= view.z_hi; ++z ) {
        for( auto dy = -creature_radius; dy <= creature_radius; ++dy ) {
            for( auto dx = -creature_radius; dx <= creature_radius; ++dx ) {
                const auto p = tripoint_bub_ms( view.center.x() + dx, view.center.y() + dy, z );
                if( !here.inbounds( p ) ) {
                    continue;
                }
                const auto field = visible_field_at( here, you, p );
                if( field && field->display_field ) {
                    auto subtile = 0;
                    auto rotation = 0;
                    view.tiles.get_orientation( { .pos = p, .furniture = false, .subtile = subtile,
                                                  .rotation = rotation, .field = true } );
                    const auto &sprite = view.cache.named( { .category = C_FIELD, .id = field->id.id().str(),
                                                           .tiles = view.tiles, .subtile = subtile, .rota = rotation } );
                    add_ground_decal( { .sink = sink, .tile = sprite, .x = frame_x( p ) - 0.5f,
                                        .y = frame_y( p ) - 0.5f, .z = frame_z( p ) + floor_height + 0.015f,
                                        .color = view.light( p ) } );
                }
                // Nothing on tiles the player cannot see into, and nothing inside closed
                // furniture (lockers, cupboards): the flat view shows those once the player
                // stands next to them, but here the furniture is drawn closed, and its
                // contents would sit on its top (or float there when the cutaway removes it).
                if( std::abs( dx ) > item_radius || std::abs( dy ) > item_radius ||
                    ( field && !field->display_items ) || !here.sees_some_items( p, you ) ||
                    here.has_flag( "CONTAINER", p ) ) {
                    continue;
                }
                auto stack = map_stack( here.i_at( p ) );
                if( stack.empty() ) {
                    continue;
                }
                const auto light = static_cast<uint32_t>( view.light( p ) );
                if( light == pack_light( 0, 0, 0 ) ) {
                    continue; // not visible
                }
                const auto id = std::string( ( *stack.begin() )->typeId().str() );
                const auto &sprite = view.cache.named( { .category = C_ITEM, .id = id, .tiles = view.tiles } );
                // Keep the item's illustrated profile visible at every camera angle, like
                // creature sprites, rather than projecting it onto the ground.
                emit_billboard( { .sink = sink, .layers = { &sprite }, .right_world = right, .up_world = up, .toward_world = toward, .z_scale = view.cam.z_scale,
                                  .cx = frame_x( p ), .cy = frame_y( p ), .z_base = frame_z( p ) + view.surface( p ) + 0.02f,
                                  .color = light, .mirrored = false, .scale = item_size } );
            }
        }
    }

    // Trees and tall furniture of the visible chunks; unseen ones are drawn like their tile.
    for( const entity_view::standing &u : view.uprights ) {
        if( !u.info ) {
            continue;
        }
        emit_billboard( { .sink = sink, .layers = { u.info }, .right_world = right, .up_world = up, .toward_world = toward, .z_scale = view.cam.z_scale, .cx = frame_x( u.pos ),
                          .cy = frame_y( u.pos ),
                          .z_base = frame_z( u.pos ) + u.z_base, .color = view.light( u.pos ) } );
    }

    // Creatures: the player, NPCs and monsters that the player can see.
    for( Creature &critter : g->all_creatures() ) {
        const auto p = tripoint_bub_ms( critter.bub_pos() );
        if( !in_range( p, creature_radius ) ) {
            continue;
        }
        if( !critter.is_avatar() && !you.sees( critter ) ) {
            continue;
        }
        // Characters are their body plus worn and wielded overlays, like the flat view.
        std::vector<const tile_info *> layers;
        auto mirrored = static_cast<bool>( false );
        if( const monster *m = critter.as_monster() ) {
            layers.push_back( &view.cache.named( { .category = C_MONSTER, .id = m->type->id.str(), .tiles = view.tiles } ) );
        } else if( const Character *ch = critter.as_character() ) {
            const auto looks = cata_tiles::character_layers( view.tiles.get_character_layers( *ch ) );
            mirrored = looks.mirrored;
            for( const std::pair<std::string, TILE_CATEGORY> &id : looks.ids ) {
                layers.push_back( &view.cache.named( { .category = id.second, .id = id.first, .tiles = view.tiles } ) );
            }
        } else {
            continue;
        }
        const auto color = static_cast<uint32_t>( critter.is_avatar() ? white : view.light( p ) );
        // Debug aid: CATA_R3D_DEBUG_LAYERS=1 logs the sprite layers of the player once per run.
        static const auto debug_layers = static_cast<bool>( std::getenv( "CATA_R3D_DEBUG_LAYERS" ) !=
                                         nullptr );
        static auto logged = static_cast<bool>( false );
        if( debug_layers && critter.is_avatar() && !logged ) {
            logged = true;
            const auto looks = cata_tiles::character_layers( view.tiles.get_character_layers(
                                   *critter.as_character() ) );
            for( auto i = size_t( 0 ); i < looks.ids.size() && i < layers.size(); ++i ) {
                const auto &m = layers[i]->main();
                DebugLog( DL::Info, DC::Main ) << "r3d layer " << i << " " << looks.ids[i].first << " valid=" <<
                                               m.valid
                                               << " size=" << m.size_x << "x" << m.size_y << " uv=" << m.u0 << "," << m.v0
                                               << " backdrop=" << layers[i]->backdrop.valid;
            }
        }
        emit_billboard( { .sink = sink, .layers = layers, .right_world = right, .up_world = up, .toward_world = toward, .z_scale = view.cam.z_scale, .cx = frame_x( p ), .cy = frame_y( p ),
                          .z_base = frame_z( p ) + view.surface( p ), .color = color == pack_light( 0, 0, 0 ) ? white : color,
                          .mirrored = mirrored } );
    }
    // Markers last; they sit just above the surface so the depth test keeps them in front of it.
    for( const entity_view::marker &m : view.markers ) {
        if( !m.sprite || m.pos.z() < view.z_lo || m.pos.z() > view.z_hi ) {
            continue;
        }
        const auto x0 = static_cast<float>( m.pos.x() - view.center.x() );
        const auto y0 = static_cast<float>( m.pos.y() - view.center.y() );
        const auto z = static_cast<float>( frame_z( m.pos ) + m.top + 0.03f );
        emit_flat( { .sink = sink, .sprite = *m.sprite, .cx = x0 + 0.5f, .cy = y0 + 0.5f, .z = z, .color = white, .size = 1.0f } );
        if( m.box ) {
            const auto &s = m.sprite->sprite.valid ? m.sprite->sprite : m.sprite->backdrop;
            if( s.valid ) {
                emit_marker_box( { .sink = sink, .s = s, .x0 = x0, .y0 = y0, .z0 = z, .z1 = z + marker_box_height, .color = white } );
            }
        }
    }
    return sink.finish();
}



auto build_particle_mesh( const entity_view &view ) -> chunk_mesh
{
    auto sink = quad_sink{};
    const auto right = vec3( view.cam.ground_right() );
    const auto toward = vec3( normalize( view.cam.eye() - view.cam.stretched_target() ) );
    const auto up = vec3( cross( right, vec3{ 0.0f, 0.0f, 0.0f } - toward ) );
    // Weather particles: plain quads facing the camera, centred on their point.
    for( const entity_view::particle &pt : view.particles ) {
        if( !pt.sprite || !pt.sprite->valid ) {
            continue;
        }
        const sprite_ref &s = *pt.sprite;
        // Only the opaque part of the sprite (a raindrop is a few pixels of a whole tile), at
        // its own proportions, pt.size tall.
        const auto u0 = static_cast<float>( s.u0 + ( s.u1 - s.u0 ) * s.opaque_x0 );
        const auto u1 = static_cast<float>( s.u0 + ( s.u1 - s.u0 ) * s.opaque_x1 );
        const auto v0 = static_cast<float>( s.v0 + ( s.v1 - s.v0 ) * s.opaque_y0 );
        const auto v1 = static_cast<float>( s.v0 + ( s.v1 - s.v0 ) * s.opaque_y1 );
        const auto opaque_h = static_cast<float>( std::max( ( s.opaque_y1 - s.opaque_y0 ) * s.size_y,
                              0.01f ) );
        const auto aspect = static_cast<float>( std::clamp( ( s.opaque_x1 - s.opaque_x0 ) * s.size_x /
                                                opaque_h, 0.1f,
                                                4.0f ) );
        const auto half_h = static_cast<float>( pt.size * 0.5f );
        const auto half_w = static_cast<float>( half_h * aspect );
        const auto rx = static_cast<float>( right.x * half_w );
        const auto ry = static_cast<float>( -right.y * half_w );
        const auto ux = static_cast<float>( up.x * half_h );
        const auto uy = static_cast<float>( -up.y * half_h );
        const auto uz = static_cast<float>( up.z * half_h / view.cam.z_scale );
        const auto uv = uvs_t{{ { u0, v0 }, { u1, v0 }, { u1, v1 }, { u0, v1 } }};
        const auto c = corners_t{{
                { pt.x - rx + ux, pt.y - ry + uy, pt.z + uz }, { pt.x + rx + ux, pt.y + ry + uy, pt.z + uz },
                { pt.x + rx - ux, pt.y + ry - uy, pt.z - uz }, { pt.x - rx - ux, pt.y - ry - uy, pt.z - uz }
            }
        };
        sink.add_colored( { .sprite = s, .corners = c, .uvs = uv, .ref = light_ref{}, .color = no_pull( pt.color ) } );
    }
    return sink.finish();
}

} // namespace r3d

#endif // defined(TILES)
