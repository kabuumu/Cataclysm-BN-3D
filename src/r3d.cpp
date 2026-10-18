#if defined(TILES)

#include "r3d.h"
#include "r3d_capture_messages.h"
#include "r3d_capture_store.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "avatar.h"
#include "calendar.h"
#include "world.h"
#include "cata_utility.h"
#include "cata_tiles.h"
#include "coordinates.h"
#include "creature.h"
#include "creature_tracker.h"
#include "debug.h"

#include "game.h"
#include "game_info.h"
#include "game_constants.h"

#include "json.h"
#include "map/map.h"
#include "game_constants.h"
#include "map/mapbuffer.h"
#include "map/mapdata.h"
#include "options.h"
#include "path_info.h"
#include "r3d_camera.h"
#include "r3d_dda.h"
#include "r3d_entities.h"
#include "r3d_scene_rules.h"
#include "r3d_gpu.h"
#include  "r3d_hud.h"
#include "r3d_mesh.h"
#include "rng.h"
#include "sdl_wrappers.h"
#include "sdltiles.h"
#include "map/submap.h"
#include "sounds.h"
#include "string_formatter.h"
#include "translations.h"
#include "ui_manager.h"


#include "uistate.h"

namespace r3d
{

namespace
{

constexpr auto max_chunk_builds_per_frame = static_cast<int>( 8 );
constexpr auto chunk_idle_frames = static_cast<uint64_t>( 600 );
// Tallest geometry in a chunk: upright sprites (big trees) can stand several tiles tall.
constexpr auto chunk_max_height = static_cast<float>( 4.0f );

struct chunk_key {
    int x = 0;
    int y = 0;
    int z = 0;
    auto operator==( const chunk_key &o ) const -> bool {
        return x == o.x && y == o.y && z == o.z;
    }
};

struct chunk_key_hash {
    auto operator()( const chunk_key &k ) const -> size_t {
        return ( static_cast<size_t>( k.x ) * 73856093u ) ^ ( static_cast<size_t>( k.y ) * 19349663u ) ^
               ( static_cast<size_t>( k.z ) * 83492791u );
    }
};

struct chunk_gpu {
    bool built = false;
    uint64_t stamp = 0;
    uint64_t last_used = 0;
    SDL_GPUBuffer *positions = nullptr;
    SDL_GPUBuffer *lights = nullptr;
    SDL_GPUBuffer *indices = nullptr;
    std::vector<mesh_batch> batches;
    std::vector<upright_sprite> uprights; // drawn as billboards every frame
    std::vector<light_ref> refs; // where each vertex samples its light
    std::vector<uint32_t> light; // current RGBA8 light per vertex
    uint64_t light_key = 0;      // light_key() the stream was last computed for
};

// What the last drawn frame showed, so mouse picks match the picture on screen.
struct shown_view {
    bool valid = false;
    float x = 0.0f;
    float y = 0.0f;
    float w = 0.0f;
    float h = 0.0f;
    camera cam;
    tripoint_bub_ms center;
    int z_lo = 0;
    int z_hi = 0;
};

struct state {
    gpu device;
    tile_info_cache cache;
    std::unordered_map<chunk_key, chunk_gpu, chunk_key_hash> chunks;
    camera cam;
    unsigned epoch = 0;
    bool epoch_valid = false;
    bool init_failed = false;
    unsigned failed_epoch = 0;
    uint64_t frame = 0;
    bool drawn = false;
    // The tileset load the cached sprites and meshes were built from.
    uint64_t tileset_loads = 0;
    bool env_camera_applied = false;
    bool screenshot_taken = false;
    // Per-frame buffers for creatures and items, grown as needed and re-uploaded every frame.
    SDL_GPUBuffer *dyn_positions = nullptr;
    SDL_GPUBuffer *dyn_lights = nullptr;
    SDL_GPUBuffer *dyn_indices = nullptr;
    uint32_t dyn_position_bytes = 0;
    uint32_t dyn_light_bytes = 0;
    uint32_t dyn_index_bytes = 0;
    shown_view shown;
    bool map_drawn = false; // the game's map view has been on screen (a game is running)
    vec3 pan; // free camera: offset of the camera target from the viewed tile, world space
    bool walls_on = false; // whether the cache was given the wall side rules and atlas
    bool thin_walls = true;
    bool plaster_atlas = false; // whether painted walls take the atlas plaster (see the option)
    // whether painted walls take the atlas plaster (see the option)
    bool chunks_first_person = false; // the mode the chunk meshes were built for
    // First-person easing: the player and the heading jump, the picture catches up. The offsets
    // are what is left to show, measured from the logical camera back to where it was.
    bool fp_prev_valid = false;
    tripoint_abs_ms fp_prev_pos;
    int fp_prev_heading = 0;
    float fp_ease_dx = 0.0f; // tiles east and south at the start of the ease
    float fp_ease_dy = 0.0f;
    float fp_ease_dz = 0.0f; // levels up at the start of the ease (stairs, climbing on things)
    float fp_prev_surface = 0.0f;
    float fp_ease_yaw = 0.0f; // radians at the start of the ease
    uint32_t fp_ease_start = 0;
    uint32_t fp_ease_ms = 0;
    bool fp_easing = false; // the last frame was drawn mid-ease, so another is needed
    int fp_heading_saved = 0; // the heading last written to the save's UI state
    std::vector<hud_text> hud; // labels of the first-person overlay, see hud_texts()
    // Sounds heard this turn whose post cannot be seen (out of the picture or behind something):
    // the compass strip shows a tick for each.
    std::vector<tripoint_bub_ms> heard_ticks;
    int hud_cell_w = 8;        // the size of a character of the font they are drawn in
    int hud_cell_h = 16;
};

auto st() -> state &
{
    // Intentionally leaked: the GPU objects must not be destroyed after SDL has shut down.
    static auto *s = new state();
    return *s;
}

auto release_dynamic( state &s ) -> void
{
    s.device.release_buffer( s.dyn_positions );
    s.device.release_buffer( s.dyn_lights );
    s.device.release_buffer( s.dyn_indices );
    s.dyn_positions = nullptr;
    s.dyn_lights = nullptr;
    s.dyn_indices = nullptr;
    s.dyn_position_bytes = 0;
    s.dyn_light_bytes = 0;
    s.dyn_index_bytes = 0;
}

// Makes sure a dynamic buffer holds at least bytes, then queues the data for upload.
struct upload_dynamic_options {
    state &s;
    SDL_GPUBuffer *&buffer;
    uint32_t &capacity;
    SDL_GPUBufferUsageFlags usage;
    const void *data;
    uint32_t bytes;
};

auto upload_dynamic( const upload_dynamic_options &options ) -> bool
{
    auto &s = options.s;
    auto &buffer = options.buffer;
    auto &capacity = options.capacity;
    auto usage = options.usage;
    const auto data = options.data;
    auto bytes = options.bytes;

    if( !buffer || capacity < bytes ) {
        s.device.release_buffer( buffer );
        capacity = std::max( bytes, capacity * 2 );
        buffer = s.device.create_buffer( usage, capacity );
        if( !buffer ) {
            capacity = 0;
            return false;
        }
    }
    return s.device.upload( { .dst = buffer, .data = data, .size = bytes, .cycle = true } );
}

auto release_chunk( state &s, chunk_gpu &c ) -> void
{
    s.device.release_buffer( c.positions );
    s.device.release_buffer( c.lights );
    s.device.release_buffer( c.indices );
    c.positions = nullptr;
    c.lights = nullptr;
    c.indices = nullptr;
    c.batches.clear();
    c.uprights.clear();
    c.refs.clear();
    c.light.clear();
    c.light_key = 0;
    c.built = false;
}

auto release_all_chunks( state &s ) -> void
{
    for( auto &entry : s.chunks ) {
        release_chunk( s, entry.second );
    }
    s.chunks.clear();
}

struct rebuild_chunk_options {
    state &s;
    chunk_gpu &c;
    const tripoint_abs_sm &pos;
    uint64_t stamp;
    const submap_lookup &lookup;
};

auto rebuild_chunk( const rebuild_chunk_options &options ) -> bool
{
    auto &s = options.s;
    auto &c = options.c;
    const auto &pos = options.pos;
    auto stamp = options.stamp;
    const auto &lookup = options.lookup;

    const auto config = mesh_options{ .first_person = s.chunks_first_person, .ceilings = s.chunks_first_person };
    auto mesh = build_chunk_mesh( { .pos = pos, .lookup = lookup, .cache = s.cache,
                                    .tiles = *tilecontext, .config = config } );
    release_chunk( s, c );
    c.stamp = stamp;
    c.built = true;
    c.uprights = std::move( mesh.uprights );
    if( mesh.indices.empty() ) {
        return true;
    }
    const auto vertex_bytes = static_cast<uint32_t>( mesh.vertices.size() * sizeof( vertex ) );
    const auto light_bytes = static_cast<uint32_t>( mesh.light.size() * sizeof( uint32_t ) );
    const auto index_bytes = static_cast<uint32_t>( mesh.indices.size() * sizeof( uint32_t ) );
    c.positions = s.device.create_buffer( SDL_GPU_BUFFERUSAGE_VERTEX, vertex_bytes );
    c.lights = s.device.create_buffer( SDL_GPU_BUFFERUSAGE_VERTEX, light_bytes );
    c.indices = s.device.create_buffer( SDL_GPU_BUFFERUSAGE_INDEX, index_bytes );
    if( !c.positions || !c.lights || !c.indices ||
        !s.device.upload( { .dst = c.positions, .data = mesh.vertices.data(), .size = vertex_bytes } ) ||
        !s.device.upload( { .dst = c.indices, .data = mesh.indices.data(), .size = index_bytes } ) ) {
        release_chunk( s, c );
        return false;
    }
    c.batches = std::move( mesh.batches );
    c.refs = std::move( mesh.light_refs );
    c.light = std::move( mesh.light );
    return true;
}

// Loads data/raw/r3d/wall_sides.{json,png} (see make_wall_textures.py) onto the renderer.
// Nothing (the tileset sprites stay on wall sides) if either is missing or broken.
auto load_wall_sides( gpu &device ) -> std::shared_ptr<const wall_sides>
{
    try {
        const auto dir = std::filesystem::path( PATH_INFO::datadir() ) / "raw" / "r3d";
        auto input = std::ifstream( dir / "wall_sides.json" );
        if( !input ) { return nullptr; }
        auto json = JsonIn( input );
        auto jo = json.get_object();
        auto sides = std::shared_ptr<wall_sides>( std::make_shared<wall_sides>() );
        sides->cell = jo.get_int( "cell" );
        for( const std::string name : jo.get_array( "textures" ) ) {
            sides->names.push_back( name );
        }
        for( const JsonMember m : jo.get_object( "by_id" ) ) {
            sides->rules.by_id[m.name()] = m.get_string();
        }
        for( const JsonMember m : jo.get_object( "by_prefix" ) ) {
            sides->rules.by_prefix[m.name()] = m.get_string();
        }
        sides->rules.fallback = jo.get_string( "fallback", "" );
        if( jo.has_member( "colors" ) ) {
            const auto colors = JsonObject( jo.get_object( "colors" ) );
            for( const JsonMember m : colors ) {
                const auto rgb = std::vector<int>( colors.get_int_array( m.name() ) );
                if( rgb.size() == 3 ) {
                    sides->colors[m.name()] = static_cast<uint32_t>( rgb[0] ) << 16 |
                                              static_cast<uint32_t>( rgb[1] ) << 8 | static_cast<uint32_t>( rgb[2] );
                }
            }
        }

        const auto surface = SDL_Surface_Ptr( load_image( ( dir / "wall_sides.png" ).string().c_str() ) );
        // Uploaded by the 3D renderer itself: a texture made through the SDL_Renderer would
        // only receive its pixels at the renderer's next present.
        sides->atlas = device.create_sheet( surface.get() );
        if( !sides->atlas ) {
            return nullptr;
        }
        sides->width = surface->w;
        sides->height = surface->h;
        return sides;
    } catch( const std::exception &e ) {
        DebugLog( DL::Warn, DC::Main ) << "r3d: wall side textures not loaded: " << e.what();
        return nullptr;
    }
}

// Height of the solid part of a tile as drawn: 0 for open air, the floor slab, furniture or a
// wall up to 1. Used for picking and for placing markers on top of things.
auto column_height( state &s, const map &here, const tripoint_bub_ms &p ) -> float
{
    if( !here.inbounds( p ) ) {
        return 0.0f;
    }
    const auto t = shape( s.cache.terrain( { .id = here.ter( p ), .tiles = *tilecontext } ).sh );
    auto h = static_cast<float>( t.height );
    if( t.kind == shape_kind::floor ) {
        h = std::max( h, s.cache.furniture( { .id = here.furn( p ), .tiles = *tilecontext } ).sh.height );
    }
    return std::min( h, 1.0f );
}

// Tint of tiles that are remembered but not currently seen: a light sepia, close to how the
// flat view draws its memory (color_pixel_sepia_light), so known rooms stay readable.
auto remembered_light() -> uint32_t
{ return pack_light( 205, 188, 160 ); }

// Marker light of a never-seen tile under a roof (see light_sampler and update_chunk_light).
auto unseen_indoor_light() -> uint32_t
{ return pack_light( 1, 1, 1 ); }

// Where things on a tile stand: the floor, or the top of furniture that can be climbed onto
// (benches, counters, tables are boxes; anything that is not a full tile can be stood on).
auto surface_height( state &s, const map &here, const tripoint_bub_ms &p ) -> float
{
    if( !here.inbounds( p ) ) {
        return floor_height;
    }
    const auto t = shape( s.cache.terrain( { .id = here.ter( p ), .tiles = *tilecontext } ).sh );
    if( t.kind != shape_kind::floor ) {
        return floor_height;
    }
    const auto f = shape( s.cache.furniture( { .id = here.furn( p ), .tiles = *tilecontext } ).sh );
    return f.kind == shape_kind::prism || f.kind == shape_kind::block ?
           std::max( t.height, f.height ) : t.height;
}

// Turns the game's own visibility knowledge into a light colour per tile: lit tiles are shown
// normally, unseen but remembered ones dim and bluish, everything else black.
struct light_sampler {
    const map &here;
    const visibility_variables &vvc;
    const avatar &you;
    // Lowest level whose visibility cache the game keeps up to date for this view (see
    // map::update_visibility_cache); anything below can only show what is remembered.
    int vision_floor = -OVERMAP_DEPTH;
    // Levels above this have no visibility cache of their own (the game only computes up to
    // the viewed level); they take the light of the ground below them.
    int vision_top = OVERMAP_HEIGHT;
    int view_z = 0;

    // Whether any of the eight tiles around p is seen or remembered.
    auto borders_known( const tripoint_bub_ms &p ) const -> bool {
        for( auto dy = static_cast<int>( -1 ); dy <= 1; ++dy ) {
            for( auto dx = static_cast<int>( -1 ); dx <= 1; ++dx ) {
                if( dx == 0 && dy == 0 ) {
                    continue;
                }
                const auto n = tripoint_bub_ms( p + tripoint_rel_ms( dx, dy, 0 ) );
                if( !here.inbounds( n ) ) {
                    continue;
                }
                const auto ll = lit_level( here.access_cache( n.z() ).visibility_cache[here.access_cache(
                                               n.z() ).idx(
                                               n.x(), n.y() )] );
                const auto vis = visibility_type( here.get_visibility( ll, vvc ) );
                if( vis == VIS_CLEAR || vis == VIS_LIT ||
                    !you.get_terrain_tile( bub_to_abs( n ) ).tile.empty() ) {
                    return true;
                }
            }
        }
        return false;
    }

    auto color_at( const tripoint_bub_ms &p ) const -> uint32_t {
        if( !here.inbounds( p ) ) {
            return pack_light( 0, 0, 0 );
        }
        if( hide_overhead_tile( here, p, view_z ) ) {
            return pack_light( 0, 0, 0 );
        }
        // Debug aid: CATA_R3D_FULLBRIGHT=1 shows all loaded geometry regardless of visibility.
        static const auto fullbright = static_cast<bool>( std::getenv( "CATA_R3D_FULLBRIGHT" ) != nullptr );
        if( fullbright ) {
            return pack_light( 255, 255, 255 );
        }
        if( p.z() > vision_top ) {
            const auto below = static_cast<uint32_t>( color_at( tripoint_bub_ms( p.xy(), vision_top ) ) );
            if( below != pack_light( 0, 0, 0 ) ) {
                return below;
            }
            return !you.get_terrain_tile( bub_to_abs( p ) ).tile.empty() ? remembered_light() : pack_light( 0,
                    0, 0 );
        }
        if( p.z() < vision_floor ) {
            return !you.get_terrain_tile( bub_to_abs( p ) ).tile.empty() ? remembered_light() : pack_light( 0,
                    0, 0 );
        }
        const auto ll = lit_level( here.access_cache( p.z() ).visibility_cache[here.access_cache(
                                       p.z() ).idx(
                                       p.x(), p.y() )] );
        const auto vis = visibility_type( here.get_visibility( ll, vvc ) );
        auto r = static_cast<float>( 0.0f );
        auto g = static_cast<float>( 0.0f );
        auto b = static_cast<float>( 0.0f );
        if( vis == VIS_CLEAR || vis == VIS_LIT ) {
            switch( ll ) {
                case lit_level::LIT:
                case lit_level::BRIGHT:
                    r = g = b = 1.0f;
                    break;
                // Close to the flat view, which barely dims tiles that are seen at all.
                case lit_level::BRIGHT_ONLY:
                    r = g = b = 0.95f;
                    break;
                case lit_level::LOW:
                    r = g = b = 0.88f;
                    break;
                case lit_level::DARK:
                    r = g = b = 0.8f;
                    break;
                default:
                    break;
            }
        } else if( !you.get_terrain_tile( bub_to_abs( p ) ).tile.empty() ) {
            return remembered_light();
        } else if( !here.is_outside( p ) || borders_known( p ) ) {
            // Never seen, but under a roof or next to known ground: marked so that the floor
            // and any wall there can be drawn dark (an unexplored room reads as a dark room,
            // not a hole, and the walls at the edge of the known do not end in mid-air);
            // nothing else on it is.
            return unseen_indoor_light();
        }
        const auto to_byte = []( float v ) -> uint8_t {
            return static_cast<uint8_t>( std::clamp( v, 0.0f, 1.0f ) * 255.0f + 0.5f );
        };
        return pack_light( to_byte( r ), to_byte( g ), to_byte( b ) );
    }
};

// Changes whenever the light of any drawn tile may have: a new turn, a moved view, or a fresh
// visibility cache on one of the drawn levels (doors, light sources, ... within a turn).
struct light_key_options {
    const map &here;
    const tripoint_abs_ms &origin;
    int z_lo;
    int z_hi;
};

auto light_key( const light_key_options &options ) -> uint64_t
{
    const auto &here = options.here;
    const auto &origin = options.origin;
    auto z_lo = options.z_lo;
    auto z_hi = options.z_hi;

    auto key = static_cast<uint64_t>( to_turn<int>( calendar::turn ) );
    for( auto z = static_cast<int>( z_lo ); z <= z_hi; ++z ) {
        key = key * 1099511628211ull ^ here.get_cache_ref( z ).visibility_generation;
    }
    key = key * 1000003u + static_cast<uint64_t>( origin.x() );
    key = key * 1000003u + static_cast<uint64_t>( origin.y() );
    key = key * 1000003u + static_cast<uint64_t>( origin.z() + OVERMAP_DEPTH );
    return key | 1u; // never 0, which means "not computed"
}

struct update_chunk_light_options {
    state &s;
    chunk_gpu &c;
    const tripoint_abs_sm &pos;
    const tripoint_abs_ms &origin;
    const tripoint_bub_ms &center;
    const light_sampler &sampler;
    uint64_t key;
};

auto update_chunk_light( const update_chunk_light_options &options ) -> void
{
    auto &s = options.s;
    auto &c = options.c;
    const auto &pos = options.pos;
    const auto &origin = options.origin;
    const auto &center = options.center;
    const auto &sampler = options.sampler;
    auto key = options.key;

    c.light_key = key;
    if( c.refs.empty() ) {
        return;
    }
    constexpr auto border = static_cast<int>( SEEX + 2 );
    const auto chunk_origin = tripoint_abs_ms( project_to<coords::ms>( pos ) );
    const auto base = tripoint_bub_ms( center.x() + ( chunk_origin.x() - origin.x() ),
                                       center.y() + ( chunk_origin.y() - origin.y() ),
                                       center.z() + ( pos.z() - origin.z() ) );
    // Light of the tiles in and around the chunk, on its level and the one above.
    std::array<std::array<uint32_t, border *border>, 2> grid;
    for( auto dz = static_cast<int>( 0 ); dz < 2; ++dz ) {
        for( auto ly = static_cast<int>( -1 ); ly <= SEEY; ++ly ) {
            for( auto lx = static_cast<int>( -1 ); lx <= SEEX; ++lx ) {
                grid[dz][static_cast<size_t>( ( ly + 1 ) * border + ( lx + 1 ) )] =
                    sampler.color_at( tripoint_bub_ms( base.x() + lx, base.y() + ly, base.z() + dz ) );
            }
        }
    }
    for( auto i = size_t( 0 ); i < c.refs.size(); ++i ) {
        const auto &r = c.refs[i];
        // Decide from the geometry's own tile, before sampling a neighbouring tile's light.
        // Otherwise a wall side could survive inside an otherwise removed upper storey.
        if( hide_overhead_tile( sampler.here,
                                tripoint_bub_ms( base.x() + r.lx, base.y() + r.ly, base.z() ),
                                sampler.view_z ) ) {
            c.light[i] = pack_light( 0, 0, 0 );
            continue;
        }
        const auto gx = static_cast<int>( std::clamp( r.lx + r.dx + 1, 0, border - 1 ) );
        const auto gy = static_cast<int>( std::clamp( r.ly + r.dy + 1, 0, border - 1 ) );
        auto tile = static_cast<uint32_t>( grid[std::clamp( static_cast<int>( r.dz ), 0,
                                                1 )][static_cast<size_t>
                                                        ( gy * border + gx )] );
        // A side lit by the tile it faces would go black when that tile is out of sight (behind
        // lockers, the far side of a wall) although the thing itself is seen: then it takes its
        // own tile's light.
        if( tile == pack_light( 0, 0, 0 ) && ( r.dx != 0 || r.dy != 0 ) ) {
            tile = grid[0][static_cast<size_t>( ( r.ly + 1 ) * border + ( r.lx + 1 ) )];
        }
        // A never-seen floor or wall is drawn dark (the shader leaves black out), so an
        // unexplored room reads as a dark room, not a hole, and a wall does not stop short of
        // a tile nobody has looked at; nothing else standing there shows.
        if( tile == unseen_indoor_light() ) {
            tile = r.floor || r.wall ? pack_light( 70, 68, 74 ) : pack_light( 0, 0, 0 );
        }
        const auto shade = static_cast<uint32_t>( r.shade );
        c.light[i] = pack_light(
                         static_cast<uint8_t>( ( ( tile >> 0 ) & 255u ) * shade / 255u * r.tint_r / 255u ),
                         static_cast<uint8_t>( ( ( tile >> 8 ) & 255u ) * shade / 255u * r.tint_g / 255u ),
                         static_cast<uint8_t>( ( ( tile >> 16 ) & 255u ) * shade / 255u * r.tint_b / 255u ) );
    }
    s.device.upload( { .dst = c.lights, .data = c.light.data(),
                       .size = static_cast<uint32_t>( c.light.size() * sizeof( uint32_t ) ), .cycle = true } );
}

// Cuts a hole around the player through whatever stands in front of them (see r3d_block.frag).
struct set_cutaway_options {
    block_uniforms &u;
    const mat4 &view_proj;
    float aspect;
    const camera &cam;
    const tripoint_bub_ms &center;
};

auto set_cutaway( const set_cutaway_options &options ) -> void
{
    auto &u = options.u;
    const auto &view_proj = options.view_proj;
    auto aspect = options.aspect;
    const auto &cam = options.cam;
    const auto &center = options.center;

    const auto &you = get_avatar();
    const auto p = tripoint_bub_ms( you.bub_pos() );
    if( cam.first_person() || !get_option<bool>( "R3D_CUTAWAY" ) || p.z() != center.z() ) {
        return;
    }
    // Radius in tiles around the player's middle, and the height (above their floor) below which
    // nothing is cut, so the floor and low furniture around them stay.
    constexpr auto radius = static_cast<float>( 1.3f );
    constexpr auto keep_below = static_cast<float>( 0.3f );
    const auto base = static_cast<float>( floor_height );
    const auto middle = vec3( tile_to_world( { .x = static_cast<float>( p.x() - center.x() ) + 0.5f,
                              .y = static_cast<float>( p.y() - center.y() ) + 0.5f,
                              .z = base + cutaway_stub_height } ) );
    const auto at = vec3( transform_point( view_proj, middle ) );
    const auto side = vec3( transform_point( view_proj, middle + cam.ground_right() * radius ) );
    u.cut[0] = at.x;
    u.cut[1] = at.y;
    u.cut[2] = at.z;
    u.cut[3] = std::abs( side.x - at.x );
    u.cut2[0] = 1.0f / std::max( aspect, 0.01f );
    u.cut2[1] = base + keep_below;
}

auto  sync_camera_mode( camera &cam ) -> void
{
    constexpr auto eighth = static_cast<float>( 0.78539816f );
    const auto want_first_person = static_cast<bool>( get_option<std::string>( "R3D_CAMERA_MODE" ) ==
                                   "first_person" );
    if( want_first_person == cam.first_person() ) {
        return;
    }
    if( want_first_person ) {
        cam.heading = cam.forward_eighths();
        cam.mode = camera_mode::first_person;
        cam.look_pitch = 0.0f;
        cam.glance_yaw = 0.0f;
        cam.glance_pitch = 0.0f;
    } else {
        cam.mode = camera_mode::orbit;
        // The orbit yaw turns the other way round (see camera::forward_eighths).
        cam.yaw = -static_cast<float>( cam.heading ) * eighth;
    }
}

// The first-person heading is kept in the save's UI state (uistate), so a loaded game faces the
// way it was saved. A value there that the renderer did not write itself came from loading a
// game: the view takes it over at once, without easing a turn in.
auto sync_saved_heading( state &s ) -> void
{
    const auto saved = static_cast<int>( ( uistate.r3d_fp_heading % 8 + 8 ) % 8 );
    if( saved != s.fp_heading_saved ) {
        s.cam.heading = saved;
        s.fp_prev_heading = saved;
    }
    if( s.cam.first_person() ) {
        uistate.r3d_fp_heading = s.cam.heading;
    }
    s.fp_heading_saved = ( uistate.r3d_fp_heading % 8 + 8 ) % 8;
}

// Writes a heading the player just chose to the save's UI state, before any frame is drawn.
auto remember_heading( state &s ) -> void
{
    if( s.cam.first_person() ) {
        uistate.r3d_fp_heading = s.cam.heading;
        s.fp_heading_saved = s.cam.heading;
    }
}

// First person: eases the picture from where the camera was to where the player now is. The
// game state (position, heading, picking) always has the end values; only what is drawn lags, for
// at most R3D_FP_TWEEN_MS. Sets the offsets to add to the camera this frame and the yaw to show.
// surface is the height the player stands at within their level (floor, counter top): taking
// stairs, or climbing onto a counter, eases the eye up or down as well (offset_z, in levels).
struct update_fp_ease_options {
    state &s;
    const tripoint_abs_ms &pos;
    float surface;
    float &offset_x;
    float &offset_y;
    float &offset_z;
};

auto update_fp_ease( const update_fp_ease_options &options ) -> void
{
    auto &s = options.s;
    const auto &pos = options.pos;
    auto surface = options.surface;
    auto &offset_x = options.offset_x;
    auto &offset_y = options.offset_y;
    auto &offset_z = options.offset_z;

    constexpr auto eighth = static_cast<float>( 0.78539816f );
    auto &you = get_avatar();
    const auto now = static_cast<uint32_t>( static_cast<uint32_t>( SDL_GetTicks() ) );
    auto length = static_cast<uint32_t>( static_cast<uint32_t>( std::max( 0,
                                         get_option<int>( "R3D_FP_TWEEN_MS" ) ) ) );
    // Travelling and activities make many moves in a row; showing each would only slow them.
    if( you.is_auto_moving() || you.activity || !get_option<bool>( "ANIMATIONS" ) ) {
        length = 0;
    }
    // What is still to be shown of the ease in progress: it fades out cubically.
    auto remaining = [ &s, now ]() {
        if( s.fp_ease_ms == 0 || now - s.fp_ease_start >= s.fp_ease_ms ) {
            return 0.0f;
        }
        const auto left = static_cast<float>( 1.0f - static_cast<float>( now - s.fp_ease_start ) /
                                              static_cast<float>( s.fp_ease_ms ) );
        return left * left * left;
    };
    if( s.fp_prev_valid && length > 0 ) {
        const auto step = tripoint_rel_ms( pos - s.fp_prev_pos );
        // How far the eye rises: a storey for stairs, plus the change in what is stood on.
        const auto rise = static_cast<float>( static_cast<float>( step.z() ) + surface -
                                              s.fp_prev_surface );
        // A step to a neighbouring tile, up or down stairs (which may land on a tile next to
        // the one above or below), or onto furniture; teleports and falls are not eased.
        const auto stepped = static_cast<bool>( ( step.x() != 0 || step.y() != 0 || rise != 0.0f ) &&
                                                std::abs( step.x() ) <= 1 && std::abs( step.y() ) <= 1 &&
                                                std::abs( step.z() ) <= 1 );
        const auto turned = static_cast<int>( ( ( s.cam.heading - s.fp_prev_heading + 12 ) % 8 ) - 4 );
        if( stepped || turned != 0 ) {
            // Carry over what is left, so a held key flows from one step into the next.
            const auto left = static_cast<float>( remaining() );
            s.fp_ease_dx = s.fp_ease_dx * left - ( stepped ? static_cast<float>( step.x() ) : 0.0f );
            s.fp_ease_dy = s.fp_ease_dy * left - ( stepped ? static_cast<float>( step.y() ) : 0.0f );
            s.fp_ease_dz = s.fp_ease_dz * left - ( stepped ? rise : 0.0f );
            s.fp_ease_yaw = s.fp_ease_yaw * left - static_cast<float>( turned ) * eighth;
            s.fp_ease_start = now;
            s.fp_ease_ms = length;
        }
    } else if( length == 0 ) {
        s.fp_ease_ms = 0;
    }
    s.fp_prev_valid = true;
    s.fp_prev_pos = pos;
    s.fp_prev_surface = surface;
    s.fp_prev_heading = s.cam.heading;
    const auto left = static_cast<float>( remaining() );
    s.fp_easing = left > 0.0f;
    offset_x = s.fp_ease_dx * left;
    offset_y = s.fp_ease_dy * left;
    offset_z = s.fp_ease_dz * left;
    s.cam.tween_yaw = s.fp_ease_yaw * left;
}

auto apply_env_camera( camera &cam ) -> void
{
    // CATA_R3D_CAMERA="yaw_deg,pitch_deg,distance[,ortho]" gives repeatable views for
    // screenshots; ortho 1 switches to an orthographic projection. "-45,30,20,1" matches the
    // flat isometric view: looking north-east at 30 degrees, which draws tiles as 2:1 diamonds.
    const char *spec = std::getenv( "CATA_R3D_CAMERA" );
    if( !spec ) {
        return;
    }
    auto yaw = static_cast<float>( 0.0f );
    auto pitch = static_cast<float>( 0.0f );
    auto dist = static_cast<float>( 0.0f );
    auto ortho = static_cast<int>( 0 );

    auto fp_heading = static_cast<int>( 0 );
    auto fp_pitch = static_cast<float>( 0.0f );
    const auto n = static_cast<int>( std::sscanf( spec, "%f,%f,%f,%d,%d,%f", &yaw, &pitch, &dist,
                                     &ortho, &fp_heading,
                                     &fp_pitch ) );
    if( n >= 3 ) {
        constexpr auto deg = static_cast<float>( 0.017453293f );
        cam.yaw = yaw * deg;
        cam.pitch = std::clamp( pitch * deg, camera::min_pitch, camera::max_pitch );
        cam.distance = std::clamp( dist, camera::min_distance, camera::max_distance );
        if( n >= 4 ) {
            cam.orthographic = ortho != 0;
        }
        if( n >= 5 ) {
            cam.heading = ( fp_heading % 8 + 8 ) % 8;
        }
        if( n >= 6 ) {
            cam.look_pitch = std::clamp( fp_pitch * deg, -camera::max_look_pitch,
                                         camera::max_look_pitch );
        }
    }
}

auto maybe_take_screenshot( state &s ) -> void
{
    // CATA_R3D_SCREENSHOT=<path> saves the finished previous frame once things have settled;
    // with CATA_R3D_QUIT_AFTER_SHOT set the game then exits, for scripted visual checks.
    const char *path = std::getenv( "CATA_R3D_SCREENSHOT" );
    if( !path || s.screenshot_taken || s.frame < 40 ) {
        return;
    }
    s.screenshot_taken = true;
    save_screenshot( path );
    if( std::getenv( "CATA_R3D_QUIT_AFTER_SHOT" ) ) {
        std::_Exit( 0 );
    }
}

struct picking_column_height_options {
    state &s;
    const shown_view &v;
    const map &here;
    const avatar &you;
    int dx;
    int dy;
    int dz;
};

auto picking_column_height( const picking_column_height_options &options ) -> float
{
    auto &s = options.s;
    const auto &v = options.v;
    const auto &here = options.here;
    const auto &you = options.you;
    auto dx = options.dx;
    auto dy = options.dy;
    auto dz = options.dz;

    const auto p = tripoint_bub_ms( v.center.x() + dx, v.center.y() + dy, v.center.z() + dz );
    if( p.z() < v.z_lo || p.z() > v.z_hi || !here.inbounds( p ) ) {
        return 0.0f;
    }
    const auto h = static_cast<float>( column_height( s, here, p ) );
    const Creature *critter = g->critter_at( p, true );
    if( critter && ( critter->is_avatar() ? !v.cam.first_person() : you.sees( *critter ) ) ) {
        return 1.0f;
    }
    return h;
}

struct loaded_submap_at_options {
    const tripoint_abs_sm &base;
    int view_z;
    bool multi_z;
    const tripoint_abs_sm &p;
};

auto loaded_submap_at( const loaded_submap_at_options &options ) -> const submap *
{
    const auto &base = options.base;
    auto view_z = options.view_z;
    auto multi_z = options.multi_z;
    const auto &p = options.p;

    const auto dx = static_cast<int>( p.x() - base.x() );
    const auto dy = static_cast<int>( p.y() - base.y() );
    if( dx < 0 || dy < 0 || dx >= get_map().getmapsize() || dy >= get_map().getmapsize() ||
        p.z() < -OVERMAP_DEPTH || p.z() > OVERMAP_HEIGHT ) {
        return nullptr;
    }
    if( p.z() != view_z && !multi_z ) {
        return nullptr;
    }
    const auto view = get_map().active_submap_view( p );
    return view ? &view->get_submap() : nullptr;
}

struct append_frame_markers_options {
    state &s;
    const map &here;
    entity_view &entities_view;
    const std::vector<tripoint_bub_ms> &where;
    const std::string &id;
    bool box;
};

auto append_frame_markers( const append_frame_markers_options &options, bool large ) -> void
{
    auto &s = options.s;
    const auto &here = options.here;
    auto &entities_view = options.entities_view;
    const auto &where = options.where;
    const auto &id = options.id;
    auto box = options.box;

    const auto &sprite = s.cache.named( { .category = C_NONE, .id = id, .tiles = *tilecontext } );
    for( const tripoint_bub_ms &p : where ) {
        entities_view.markers.push_back( { .pos = p, .top = column_height( s, here, p ), .sprite = &sprite, .box = box, .large = large, .heard = false } );
    }

}

struct occluding_column_height_options {
    state &s;
    const map &here;
    const light_sampler &sampler;
    const tripoint_bub_ms &center;
    int z_lo;
    int z_hi;
    bool occlude_unseen;
    int dx;
    int dy;
    int dz;
};

auto occluding_column_height( const occluding_column_height_options &options ) -> float
{
    auto &s = options.s;
    const auto &here = options.here;
    const auto &sampler = options.sampler;
    const auto &center = options.center;
    auto z_lo = options.z_lo;
    auto z_hi = options.z_hi;
    auto occlude_unseen = options.occlude_unseen;
    auto dx = options.dx;
    auto dy = options.dy;
    auto dz = options.dz;

    const auto c = tripoint_bub_ms( center.x() + dx, center.y() + dy, center.z() + dz );
    if( c.z() < z_lo || c.z() > z_hi || !here.inbounds( c ) ) {
        return 0.0f;
    }
    if( !occlude_unseen && sampler.color_at( c ) == pack_light( 0, 0, 0 ) ) {
        return 0.0f;
    }
    return column_height( s, here, c );
}

} // namespace

namespace
{
// Reloads the map tileset after its option changed, as the options menu does.
auto reload_map_tileset() -> bool
{
    try {
        tilecontext->reinit();
        tilecontext->load_tileset( get_option<std::string>( "TILES" ),
                                   g->get_active_world()->info->active_mod_order, false, true, false );
        g->reset_zoom();
        g->mark_main_ui_adaptor_resize();
        return true;
    } catch( const std::exception &e ) {
        DebugLog( DL::Error, DC::Main ) << "r3d compare: loading tileset " <<
                                        get_option<std::string>( "TILES" ) << " failed: " << e.what();
        return false;
    }
}

// Draws the whole screen into the renderer and saves it before it is presented.
// Writes what the comparison shots show as text: the player and camera, and the tiles around
// the view (furniture symbol over terrain symbol, and whether each is seen, remembered or
// unknown), so a difference between the two pictures can be traced to a map tile.
auto  capture( const std::string &path ) -> bool
{
    g->invalidate_main_ui_adaptor();
    ui_manager::redraw();
    const auto ok = static_cast<bool>( save_screenshot( path ) );
    DebugLog( ok ? DL::Info : DL::Error,
              DC::Main ) << "r3d compare: " << ( ok ? "saved " : "failed " ) <<
                         path;
    return ok;
}
// Writes what the shots show as text: the player and the camera (orbit and first person), and
// the tiles around the view (furniture symbol over terrain symbol, and whether each is seen,
// remembered or unknown), so something odd in a picture can be traced to a map tile.
auto dump_view( const std::string &path, const tripoint_bub_ms &center ) -> void
{
    auto out = std::ofstream( path );
    if( !out ) {
        return;
    }

    constexpr auto deg = static_cast<float>( 57.29578f );
    const auto &here = get_map();
    const auto &you = get_avatar();
    const auto &s = st();
    const auto &cam = s.cam;
    out << "turn " << to_turn<int>( calendar::turn ) << " moves " << you.get_moves() << "\n";
    out << "player bub " << you.bub_pos().to_string() << " abs " << you.abs_pos().to_string()
        << "\nview center bub " << center.to_string()
        << "\ncamera mode " << ( cam.first_person() ? "first_person" : "orbit" )
        << "\norbit yaw " << cam.yaw << " pitch " << cam.pitch << " distance " << cam.distance
        << " ortho " << cam.orthographic << " pan " << s.pan.x << "," << s.pan.y << "," << s.pan.z
        << " z_scale " << cam.z_scale << "\n";

    if( cam.first_person() ) {
        static const char *const names[8] = { "N", "NE", "E", "SE", "S", "SW", "W", "NW" };
        const auto eye = vec3( cam.eye() );
        const auto aspect = static_cast<float>( s.shown.valid &&
                                                s.shown.h > 0.0f ? s.shown.w / s.shown.h : 1.0f );
        out << "first person heading " << cam.heading << " (" << names[( cam.heading % 8 + 8 ) % 8]
            << ")\n  eye (stretched world, scene frame: x east, y north, z up) " << eye.x << ","
            << eye.y << "," << eye.z << "\n  eye height " << cam.eye_height << " yaw "
            << cam.fp_yaw() * deg << " deg, look pitch " << cam.look_pitch *deg << " + tilt "
            << cam.tilt_pitch *deg << " + glance " << cam.glance_pitch *deg << " deg\n  fov x "
            << cam.fp_fovx( aspect ) * deg << " deg, fov y " << cam.fp_fovy( aspect ) * deg
            << " deg at aspect "
            << aspect << "\n";
    }
    constexpr auto half_w = static_cast<int>( 16 );
    constexpr auto half_h = static_cast<int>( 12 );
    const auto &vvc = here.get_visibility_variables_cache();
    auto legend = std::map<std::string, std::string> {};
    out << "\nmap (x " << center.x() - half_w << ".." << center.x() + half_w << ", y "
        << center.y() - half_h << ".." << center.y() + half_h << ", north up, @ = player):\n";
    for( auto dy = static_cast<int>( -half_h ); dy <= half_h; ++dy ) {
        for( auto dx = static_cast<int>( -half_w ); dx <= half_w; ++dx ) {
            const auto p = tripoint_bub_ms( center + tripoint_rel_ms( dx, dy, 0 ) );
            if( !here.inbounds( p ) ) {
                out << ' ';
                continue;
            }
            if( p == you.bub_pos() ) {
                out << '@';
                continue;
            }
            const auto f = furn_id( here.furn( p ) );
            const auto t = ter_id( here.ter( p ) );
            const auto sym = static_cast<int>( f != furn_str_id::NULL_ID() ? f->symbol() : t->symbol() );
            const auto id = std::string( f != furn_str_id::NULL_ID() ? f.id().str() : t.id().str() );
            legend[std::string( 1, static_cast<char>( sym ) )] = id;
            out << static_cast<char>( sym );
        }
        out << '\n';
    }
    out << "\nvisibility (S seen, M remembered, . unknown):\n";
    for( auto dy = static_cast<int>( -half_h ); dy <= half_h; ++dy ) {
        for( auto dx = static_cast<int>( -half_w ); dx <= half_w; ++dx ) {
            const auto p = tripoint_bub_ms( center + tripoint_rel_ms( dx, dy, 0 ) );
            if( !here.inbounds( p ) ) {
                out << ' ';
                continue;
            }
            const auto ll = lit_level( here.access_cache( p.z() ).visibility_cache[here.access_cache(
                                           p.z() ).idx(
                                           p.x(), p.y() )] );
            const auto vis = visibility_type( here.get_visibility( ll, vvc ) );
            const auto seen = static_cast<bool>( vis == VIS_CLEAR || vis == VIS_LIT );
            out << ( seen ? 'S' : !you.get_terrain_tile( bub_to_abs( p ) ).tile.empty() ? 'M' : '.' );
        }
        out << '\n';
    }
    out << "\nitems (tiles whose items the player sees; top item and count):\n";
    for( auto dy = static_cast<int>( -half_h ); dy <= half_h; ++dy ) {
        for( auto dx = static_cast<int>( -half_w ); dx <= half_w; ++dx ) {
            const auto p = tripoint_bub_ms( center + tripoint_rel_ms( dx, dy, 0 ) );
            if( !here.inbounds( p ) || !here.sees_some_items( p, you ) ) {
                continue;
            }
            const auto items = map_stack( get_map().i_at( p ) );
            if( items.empty() ) {
                continue;
            }
            out << "  (" << p.x() << ',' << p.y() << ") " << ( *items.begin() )->tname() << " x"
                << items.size() << '\n';
        }
    }
    out << "\nlegend (last id drawn with each symbol):\n";
    for( const auto &e : legend ) {
        out << "  " << e.first << "  " << e.second << '\n';
    }
}

auto settle_frames( ) -> void
{
    constexpr auto frames = static_cast<int>( 40 );
    const auto &r = get_sdl_renderer();
    for( auto i = static_cast<int>( 0 ); i < frames; ++i ) {
        g->invalidate_main_ui_adaptor();
        ui_manager::redraw();
        if( r ) {
            SDL_RenderPresent( r.get() );
        }
    }
}

// Restores the complete player view even when capture or tileset loading throws.
struct comparison_view_guard {
    options_manager::cOpt &tiles = get_options().get_option( "TILES" );
    options_manager::cOpt &view_3d = get_options().get_option( "RENDERER_3D" );
    std::string old_tiles = tiles.getValue();
    std::string old_view_3d = view_3d.getValue();
    options_manager::cOpt &mode = get_options().get_option( "R3D_CAMERA_MODE" );
    std::string old_mode = mode.getValue();
    int old_heading = uistate.r3d_fp_heading;
    camera old_camera;
    vec3 old_pan;
    bool old_env_camera;
    int old_zoom;
    bool attempted = false;

    comparison_view_guard( const camera &cam, const vec3 &pan, bool env_camera, int zoom ) :
        old_camera( cam ), old_pan( pan ), old_env_camera( env_camera ), old_zoom( zoom ) {}

    auto restore() -> bool {
        mode.setValue( old_mode );
        uistate.r3d_fp_heading = old_heading;
        tiles.setValue( old_tiles );
        view_3d.setValue( old_view_3d );
        const auto restored = static_cast<bool>( reload_map_tileset() );
        g->set_zoom( old_zoom );
        st().cam = old_camera;
        st().pan = old_pan;
        st().env_camera_applied = old_env_camera;
        // Flush the restored atlas before resuming the player's world view.
        if( const auto &r = get_sdl_renderer() ) {
            SDL_RenderPresent( r.get() );
        }
        for( auto i = static_cast<int>( 0 ); i < 40; ++i ) {
            g->invalidate_main_ui_adaptor();
            ui_manager::redraw();
        }
        attempted = true;
        return restored;
    }

    ~comparison_view_guard() {
        if( !attempted ) {
            try {
                restore();
            } catch( const std::exception &e ) {
                DebugLog( DL::Error, DC::Main ) << "r3d compare: view restoration failed: " << e.what();
            }
        }
    }
};
} // namespace

auto compare_capture( bool gameplay_boundary ) -> void
{
    // CATA_R3D_COMPARE=<prefix> saves, once, as soon as a game is on screen:
    //  - <prefix>_2d.png: the flat view with the isometric tileset CATA_R3D_COMPARE_TILESET
    //    (default Ultica_iso);
    //  - <prefix>_3d.png: the same turn and place in the 3D view with tileset
    //    CATA_R3D_COMPARE_3D_TILESET (default UltimateCataclysm) through the CATA_R3D_CAMERA
    //    camera (e.g. "-45,30,20,1"); an orthographic camera gets the isometric shot's scale.
    // No input is handled in between, so no turn passes. Afterwards the previous tileset and
    // view come back (the options file is never written), or with CATA_R3D_QUIT_AFTER_SHOT the
    // game exits (status 0 when both shots were saved).
    static const char *const prefix = std::getenv( "CATA_R3D_COMPARE" );
    static const char *const session = std::getenv( "CATA_R3D_COMPARE_SESSION" );
    static auto done = static_cast<bool>( false );
    static auto busy = static_cast<bool>( false );
    static auto sequence = static_cast<int>( 0 );
    static auto last = std::chrono::steady_clock::time_point::min();
    const auto number = []( const char *name, int fallback, int maximum ) -> int {
        const char *v = std::getenv( name );
        return v ? std::clamp( std::atoi( v ), 1, maximum ) : fallback;
    };
    const auto now = std::chrono::steady_clock::now();
    const auto status = []( const std::string & message ) -> void {
        if( session )
        {
            std::ofstream( std::filesystem::path( session ) / "session-status.txt" ) << message << '\n';
        }
    };
    static auto started = false;
    static auto reached_gameplay = false;
    if( session && !started ) {
        status( "Capture support initialized; waiting for a character to reach gameplay input." );
        started = true;
    }
    if( session && gameplay_boundary && !busy && !reached_gameplay ) {
        status( "Gameplay input reached; waiting for a tiled map frame. USE_TILES=" +
                get_options().get_option( "USE_TILES" ).getValue() );
        reached_gameplay = true;
    }
    if( ( session && !gameplay_boundary ) || busy || ( !prefix && !session ) || ( !session && done ) ||
        !g ||
        !tilecontext || !st().map_drawn ||
        ( session && ( sequence >= number( "CATA_R3D_COMPARE_COUNT", 5, 100 ) ||
                       ( sequence > 0 && now - last < std::chrono::seconds(
                             number( "CATA_R3D_COMPARE_INTERVAL", 30, 86400 ) ) ) ) ) ) {
        return;
    }
    done = true;
    busy = true;
    const auto message_read_guard = Messages::capture_read_guard{};
    auto clear_busy = on_out_of_scope( []() -> void {
        busy = false;
    } );
    last = now;
    ++sequence;
    auto base = std::string( prefix ? prefix : "" );
    auto checkpoint = std::filesystem::path{};
    // Saving and rendering must not perturb the next gameplay random draw.
    const auto old_rng = rng_get_engine();
    auto restore_rng = on_out_of_scope( [&old_rng]() -> void {
        rng_get_engine() = old_rng;
    } );
    const auto old_camera = camera( st().cam );
    const auto old_pan = vec3( st().pan );
    const auto old_env_camera = static_cast<bool>( st().env_camera_applied );
    const auto old_zoom = static_cast<int>( g->get_zoom() );
    if( session ) {
        char name[32];
        std::snprintf( name, sizeof( name ), "capture-%04d", sequence );
        const auto directory = std::filesystem::path( session ) / name;
        checkpoint = directory;
        status( "Capturing " + std::string( name ) );
        try {
            std::filesystem::create_directories( directory );
            // A capture save omits the gameplay save event. The launcher isolates disk writes
            // from the user's original world. Never copy an older disk save on failure.
            if( !g->save_for_capture() ) {
                throw std::runtime_error( "current-state save failed; check disk space and permissions" );
            }
            auto world = std::filesystem::path( g->get_active_world()->info->folder_path() );
            if( world.filename().empty() ) {
                world = world.parent_path();
            }
            const auto destination = directory / "save" / world.filename();
            std::filesystem::create_directories( destination );
            for( const auto &entry : std::filesystem::recursive_directory_iterator( world ) ) {
                if( entry.is_symlink() ) {
                    throw std::runtime_error( "symlink in world save" );
                }
                const auto target = destination / entry.path().lexically_relative( world );
                if( entry.is_directory() ) {
                    std::filesystem::create_directories( target );
                } else if( entry.is_regular_file() ) {
                    const auto name = entry.path().filename().string();
                    if( entry.path().extension() == ".sqlite3" ) {
                        const auto saved = snapshot_database( entry.path(), target );
                        if( !saved ) { throw std::runtime_error( saved.error() ); }
                    } else if( !name.ends_with( "-wal" ) && !name.ends_with( "-shm" ) &&
                               !name.ends_with( "-journal" ) ) {
                        std::filesystem::copy_file( entry.path(), target );
                    }
                }
            }
            std::filesystem::create_directories( directory / "config" );
            auto options_file = std::ofstream( directory / "config" / "options.json" );
            auto options_json = JsonOut( options_file, true );
            get_options().serialize( options_json );
            options_file.close();
            if( !options_file ) {
                throw std::runtime_error( "checkpoint options write failed" );
            }
            std::ofstream( directory / "state.txt" ) << "turn " << to_turn<int>( calendar::turn )
                    << " moves " << get_avatar().get_moves() << "\n";
            base = ( directory / "view" ).string();
            std::ofstream( directory / "game-report.txt" ) << game_info::game_report();
            std::ofstream( directory / "rng.txt" ) << old_rng;
        } catch( const std::exception &e ) {
            DebugLog( DL::Error, DC::Main ) << "r3d compare: checkpoint failed: " << e.what();
            std::ofstream( directory / "FAILED.txt" ) << e.what();
            status( std::string( name ) + " failed: " + e.what() );
            return;
        }
    }
    auto original_view = comparison_view_guard( old_camera, old_pan, old_env_camera, old_zoom );
    auto &tiles = original_view.tiles;
    auto &view_3d = original_view.view_3d;
    const auto env_or = []( const char *name, const char *fallback ) -> std::string {
        const char *const value = std::getenv( name );
        return std::string( value ? value : fallback );
    };

    try {
        const auto ok_before = static_cast<bool>( !session || capture( base + "_before.png" ) );
        original_view.mode.setValue( "orbit" );
        // The flat isometric view first: its tile size sets the scale of the 3D shot.
        tiles.setValue( env_or( "CATA_R3D_COMPARE_TILESET", "HitButton_iso" ) );
        view_3d.setValue( "false" );
        const auto ok_2d = static_cast<bool>( reload_map_tileset() && capture( base + "_2d.png" ) );
        const auto iso_tile_width = static_cast<int>( tilecontext->get_tile_width() );

        if( session ) {
            st().pan = vec3{};
            st().cam = camera();
            st().cam.z_scale = old_camera.z_scale;
            st().cam.orthographic = true;
            apply_env_camera( st().cam );
            st().env_camera_applied = true;
        }
        tiles.setValue( env_or( "CATA_R3D_COMPARE_3D_TILESET", "UltimateCataclysm" ) );
        view_3d.setValue( "true" );
        auto ok_3d = static_cast<bool>( reload_map_tileset() );
        if( ok_3d ) {
            // The game is waiting for input and draws nothing by itself; draw until the chunks
            // around the view are built (a few per frame).
            const auto redraw = []() -> void {
                g->invalidate_main_ui_adaptor();
                ui_manager::redraw();
            };
            redraw();
            // The renderer only uploads the new tileset's textures when it presents, and the
            // capture reads the 3D pass before any present: show one frame so the sheets the
            // meshes sample from are on the GPU (SDL versions differ in when a flush uploads).
            if( const auto &r = get_sdl_renderer() ) {
                SDL_RenderPresent( r.get() );
            }
            auto &s = st();
            if( s.cam.orthographic && iso_tile_width > 0 && s.shown.valid ) {
                // Same scale as the flat view: a tile (whose diagonal spans sqrt(2) across the
                // screen at the isometric yaw) as wide as an isometric diamond.
                const auto half_height = static_cast<float>( 1.41421356f * s.shown.h / ( 2.0f * iso_tile_width ) );
                s.cam.distance = std::clamp( half_height / std::tan( s.cam.fovy * 0.5f ),
                                             camera::min_distance, camera::max_distance );
            }
            constexpr auto settle_frames = static_cast<int>( 40 );
            for( auto i = static_cast<int>( 0 ); i < settle_frames; ++i ) {
                redraw();
            }
            ok_3d = active() && capture( base + "_3d.png" );
            dump_view( base + "_map.txt", g->ter_view_p );
            if( session ) {
                // Isometric 2D has a fixed orientation: its one reference is shared by all poses.
                constexpr auto deg = static_cast<float>( 0.017453293f );
                for( auto angle = static_cast<int>( 0 ); angle < 8; ++angle ) {
                    st().cam.yaw = ( -45.0f + angle * 45.0f ) * deg;
                    st().cam.pitch = 30.0f * deg;
                    for( auto i = static_cast<int>( 0 ); i < settle_frames; ++i ) {
                        redraw();
                    }
                    ok_3d = capture( base + "_3d_yaw_" + std::to_string( angle ) + ".png" ) && ok_3d;
                }
            }
        }

        if( std::getenv( "CATA_R3D_QUIT_AFTER_SHOT" ) ) {
            std::_Exit( ok_3d && ok_2d ? 0 : 1 );
        }
        const auto restored = static_cast<bool>( original_view.restore() );
        const auto resumed = static_cast<bool>( !session || ( restored &&
                                                capture( base + "_resumed.png" ) ) );
        rng_get_engine() = old_rng;
        if( session ) {
            const auto complete = static_cast<bool>( ok_before && ok_2d && ok_3d && restored && resumed );
            auto result =
                std::string( "Complete: 2D isometric reference, eight 3D yaw poses, same saved state." );
            if( !complete ) {
                result = "Failed stages:";
                if( !ok_before ) {
                    result += " player-view screenshot;";
                }
                if( !ok_2d ) {
                    result += " 2D tileset or screenshot;";
                }
                if( !ok_3d ) {
                    result += " 3D renderer or screenshot (requires SDL gpu renderer and compiled shaders);";
                }
                if( !restored || !resumed ) {
                    result += " player-view restoration;";
                }
            }
            std::ofstream( checkpoint / ( complete ? "COMPLETE.txt" : "FAILED.txt" ) ) << result << '\n';
            DebugLog( complete ? DL::Info : DL::Error, DC::Main ) << "r3d compare: " << result;
            status( checkpoint.filename().string() + ": " + result );
        }
    } catch( const std::exception &e ) {
        DebugLog( DL::Error, DC::Main ) << "r3d compare: capture failed: " << e.what();
        if( session ) {
            std::ofstream( checkpoint / "FAILED.txt" ) << e.what();
            status( checkpoint.filename().string() + " failed: " + e.what() );
        }
    }
}

auto  set_hud_font( int cell_width, int cell_height ) -> void
{
    auto &s = st();
    s.hud_cell_w = std::max( 1, cell_width );
    s.hud_cell_h = std::max( 1, cell_height );
}

auto hud_texts() -> const std::vector<hud_text> & // *NOPAD*
{ return st().hud; }

auto fp_shots_capture() -> void
{
    // CATA_R3D_FP_SHOTS=<prefix> saves, once, as soon as a game is on screen, with no input
    // handled in between (so no turn passes):
    //  - <prefix>_orbit.png: the 3D view through the orbit camera, for reference;
    //  - <prefix>_fp_<n|ne|e|se|s|sw|w|nw>.png: the whole window in first person facing each of
    //    the eight headings, HUD included;
    //  - <prefix>_map.txt: the player, both cameras and the tiles around them (see dump_view).
    // CATA_R3D_FP_TILESET picks the tileset (default: the current one), CATA_R3D_FP_LOOK=up,
    // down or level (default) tilts the first-person shots by 45 degrees. Afterwards the
    // tileset, the 3D view, the camera mode and the heading come back (the options file is
    // never written), or with CATA_R3D_QUIT_AFTER_SHOT the game exits (0 when every shot saved).
    static const char *const prefix = std::getenv( "CATA_R3D_FP_SHOTS" );
    static auto done = static_cast<bool>( false );
    if( !prefix || done || !g || !tilecontext || !st().map_drawn ) {
        return;
    }
    done = true;
    const auto base = std::string( prefix );
    auto &tiles = get_options().get_option( "TILES" );
    auto &view_3d = get_options().get_option( "RENDERER_3D" );
    auto &mode = get_options().get_option( "R3D_CAMERA_MODE" );
    const auto old_tiles = std::string( tiles.getValue() );
    const auto old_view_3d = std::string( view_3d.getValue() );
    const auto old_mode = std::string( mode.getValue() );
    auto &s = st();
    // The whole camera: switching modes turns the orbit camera to the heading and back.
    const auto old_cam = camera( s.cam );
    const auto old_saved_heading = static_cast<int>( uistate.r3d_fp_heading );

    auto ok = static_cast<bool>( true );
    const char *const tileset = std::getenv( "CATA_R3D_FP_TILESET" );
    const auto swap_tileset = static_cast<bool>( tileset && old_tiles != tileset );
    view_3d.setValue( "true" );
    mode.setValue( "orbit" );
    if( swap_tileset ) {
        tiles.setValue( tileset );
        ok = reload_map_tileset();
    }
    // The settling frames are presented, which also puts a new tileset's textures on the GPU
    // (the renderer only uploads them when it presents) before the shot reads the 3D pass.
    settle_frames();
    // Every shot is attempted even after one failed, so the files show what went wrong.
    ok = active() && capture( base + "_orbit.png" ) && ok;

    mode.setValue( "first_person" );
    // The first frame switches the camera (which levels the view); the tilt is set after it.
    settle_frames();
    const auto look = std::string( std::getenv( "CATA_R3D_FP_LOOK" ) ? std::getenv( "CATA_R3D_FP_LOOK" )
                                   :
                                   "level" );
    constexpr auto level_tilt = static_cast<float>( 0.7854f );
    s.cam.look_pitch = look == "up" ? level_tilt : look == "down" ? -level_tilt : 0.0f;
    static const char *const names[8] = { "n", "ne", "e", "se", "s", "sw", "w", "nw" };
    for( auto h = static_cast<int>( 0 ); h < 8; ++h ) {
        s.cam.heading = h;
        end_glance();
        // No easing between the shots, whatever R3D_FP_TWEEN_MS says.
        s.fp_prev_valid = false;
        s.fp_ease_ms = 0;
        settle_frames();
        ok = capture( base + "_fp_" + names[h] + ".png" ) && first_person() && ok;
        if( h == 0 ) {
            dump_view( base + "_map.txt", g->ter_view_p );
        }
    }
    DebugLog( ok ? DL::Info : DL::Error, DC::Main ) << "r3d fp shots: " << ( ok ? "all saved" :
            "some shots failed" );

    if( std::getenv( "CATA_R3D_QUIT_AFTER_SHOT" ) ) {
        std::_Exit( ok ? 0 : 1 );
    }
    s.cam = old_cam;
    uistate.r3d_fp_heading = old_saved_heading;
    s.fp_heading_saved = ( old_saved_heading % 8 + 8 ) % 8;
    s.fp_prev_valid = false;
    mode.setValue( old_mode );
    view_3d.setValue( old_view_3d );
    if( swap_tileset ) {
        tiles.setValue( old_tiles );
        reload_map_tileset();
    }
    g->invalidate_main_ui_adaptor();
}

auto active() -> bool
{ return st().drawn && get_option<bool>( "RENDERER_3D" ); }

auto view_eighths() -> int
{
    const auto &s = st();
    if( !active() || !s.shown.valid ) {
        return 0;
    }
    // The first-person heading changes before the next frame is drawn; the orbit camera is read
    // as last shown.
    return s.shown.cam.first_person() ? s.cam.heading : s.shown.cam.forward_eighths();
}

auto animating() -> bool
{
    const auto &s = st();
    return s.fp_easing && active() && s.shown.valid && s.shown.cam.first_person();
}

auto first_person() -> bool
{
    const auto &s = st();
    return active() && s.shown.valid && s.shown.cam.first_person() ;
}

auto wasd_controls() -> bool
{ return active() && get_option<bool>( "R3D_WASD" ); }

auto toggle() -> void
{
    get_options().get_option( "RENDERER_3D" ).setNext();
    get_options().save();
}

auto  toggle_first_person() -> void
{
    get_options().get_option( "R3D_CAMERA_MODE" ).setNext();
    get_options().save();
}

auto end_glance() -> void
{
    auto &cam = st().cam;
    cam.glance_yaw = 0.0f;
    cam.glance_pitch = 0.0f;
}

auto crosshair() -> std::optional<tripoint_bub_ms>
{
    const auto &v = st().shown;
    if( !first_person() ) {
        return std::nullopt;
    }
    return pick( v.x + v.w * 0.5f, v.y + v.h * 0.5f );
}

auto face_towards( int dx, int dy ) -> void
{
    auto &s = st();
    if( s.cam.first_person() && ( dx != 0 || dy != 0 ) ) {
        s.cam.heading = eighths_towards( dx, dy );
        remember_heading( s );
        end_glance();
    }
}

auto rotate( int dir ) -> void
{

    auto &s = st();
    s.cam.rotate_steps( dir );

    if( s.cam.first_person() ) {
        remember_heading( s );
        end_glance();
    }
}

auto pitch( int dir ) -> void
{
    st().cam.pitch_steps( dir );
}

auto zoom( int dir ) -> void
{

    auto &cam = st().cam;
    if( cam.first_person() ) {
        // There is no distance to change: the same keys look one level up or down, which is a
        // 45 degree tilt of the view and back.
        constexpr auto level_tilt = static_cast<float>( 0.7854f );
        const auto now = static_cast<int>( cam.look_pitch > 0.4f ? 1 : cam.look_pitch < -0.4f ? -1 : 0 );
        cam.look_pitch = static_cast<float>( std::clamp( now + ( dir > 0 ? 1 : -1 ), -1, 1 ) ) *
                         level_tilt;
        return;
    }
    cam.zoom_steps( dir );
}

auto reset() -> void
{
    auto &s = st();

    // First person keeps its mode and heading: resetting only straightens the view.
    const auto mode = camera_mode( s.cam.mode );
    const auto heading = static_cast<int>( s.cam.heading );
    s.cam = camera();
    s.cam.mode = mode;
    s.cam.heading = heading;
    s.pan = vec3{};
    s.env_camera_applied = false;
}

auto contains( float x, float y ) -> bool
{
    const auto &v = st().shown;
    return active() && v.valid && x >= v.x && y >= v.y && x < v.x + v.w && y < v.y + v.h;
}

auto drag_orbit( float dx, float dy ) -> void
{
    auto &cam = st().cam;
    constexpr auto yaw_per_pixel = static_cast<float>( 0.008f );
    constexpr auto pitch_per_pixel = static_cast<float>( 0.006f );
    constexpr auto two_pi = static_cast<float>( 6.2831853f );

    if( cam.first_person() ) {
        // A glance: looks around for as long as the button is held, then snaps back to the
        // heading. The mouse follows the view (right looks right, down looks down).
        constexpr auto max_glance_yaw = static_cast<float>( 2.4f );
        cam.glance_yaw = std::clamp( cam.glance_yaw + dx * yaw_per_pixel, -max_glance_yaw,
                                     max_glance_yaw );
        cam.glance_pitch = std::clamp( cam.glance_pitch - dy * pitch_per_pixel,
                                       -camera::max_look_pitch - cam.look_pitch,
                                       camera::max_look_pitch - cam.look_pitch );
        return;
    }
    cam.yaw = std::fmod( cam.yaw - dx * yaw_per_pixel, two_pi );
    cam.pitch = std::clamp( cam.pitch + dy * pitch_per_pixel, camera::min_pitch, camera::max_pitch );
}

auto drag_pan( float dx, float dy ) -> void
{
    auto &s = st();
    // The ground under the cursor follows the mouse: farther cameras move faster.

    if( s.cam.first_person() ) {
        return; // the eye stays on the player
    }
    // The ground under the cursor follows the mouse: farther cameras move faster.
    const auto units_per_pixel = static_cast<float>( s.cam.distance * 0.0016f );
    const auto right = vec3( s.cam.ground_right() );
    const auto ahead = vec3{ .x = -right.y, .y = right.x, .z = 0.0f }; // away from the camera, along the ground
    s.pan = s.pan + right * ( -dx * units_per_pixel ) + ahead * ( dy * units_per_pixel );
    // Stay over the loaded map around the view.
    constexpr auto max_pan = static_cast<float>( 48.0f );
    s.pan.x = std::clamp( s.pan.x, -max_pan, max_pan );
    s.pan.y = std::clamp( s.pan.y, -max_pan, max_pan );
}

auto pick( float x, float y ) -> std::optional<tripoint_bub_ms>
{
    auto &s = st();
    const auto &v = s.shown;
    if( !active() || !v.valid || v.w <= 0.0f || v.h <= 0.0f || !tilecontext ) {
        return std::nullopt;
    }
    if( x < v.x || y < v.y || x >= v.x + v.w || y >= v.y + v.h ) {
        return std::nullopt;
    }
    auto &here = get_map();
    const auto &you = get_avatar();

    // Visible creatures count as a full-height column so clicking their sprite picks their tile.
    const auto height = column_height_fn( [ &v, &here, &s, &you ]( int dx, int dy, int dz ) -> float {
        return picking_column_height( { .s = s, .v = v, .here = here, .you = you, .dx = dx, .dy = dy, .dz = dz } );
    } );
    const auto ndc_x = static_cast<float>( ( x - v.x ) / v.w * 2.0f - 1.0f );
    const auto ndc_y = static_cast<float>( 1.0f - ( y - v.y ) / v.h * 2.0f );
    constexpr auto max_pick_distance = static_cast<float>( 400.0f );
    const auto hit = voxel_hit( pick_cell( { .cam = v.cam, .ndc_x = ndc_x, .ndc_y = ndc_y, .aspect = v.w / v.h, .max_t = max_pick_distance, .height = height,
                                           .floor_z = floor_height } ) );
    if( !hit.hit ) {
        return std::nullopt;
    }
    // Callers expect the level of the 2D view, so a hit seen through a hole on a lower level
    // becomes the tile where the ray crossed that level (the hole), not the floor below it.
    auto px = static_cast<int>( hit.x );
    auto py = static_cast<int>( hit.y );
    if( hit.z < 0 ) {
        const auto world = ray( v.cam.screen_ray( ndc_x, ndc_y, v.w / v.h ) );
        const auto origin = vec3( world_to_tile( world.origin ) );
        const auto dir = vec3( world_to_tile( world.dir ) );
        if( dir.z < 0.0f ) {
            const auto cross = vec3( origin + dir * ( ( floor_height - origin.z ) / dir.z ) );
            px = static_cast<int>( std::floor( cross.x ) );
            py = static_cast<int>( std::floor( cross.y ) );
        }
    }
    const auto p = tripoint_bub_ms( v.center.x() + px, v.center.y() + py, get_avatar().bub_pos().z() );
    if( !here.inbounds( p ) ) {
        return std::nullopt;
    }
    return p;
}

auto shutdown() -> void
{
    auto &s = st();
    release_all_chunks( s );
    release_dynamic( s );
    s.cache.set_wall_sides( nullptr );
    s.walls_on = false;
    s.tileset_loads = 0;
    s.device.shutdown();
    s.epoch_valid = false;
    s.drawn = false;
    s.shown.valid = false;
}



namespace
{
// First-person overlay, drawn with the 2D renderer over the finished picture: a compass strip
// (the game keeps full-circle vision, so this shows what is behind you), an alert on the edge
// nearest an adjacent hostile that is out of the picture, and a crosshair with its label.
// Bearings follow r3d_hud.h: radians clockwise from north, relative to the view's yaw.
struct draw_fp_hud_options {
    state &s;
    SDL_Renderer *renderer;
    float x;
    float y;
    float w;
    float h;
    const camera &cam;
};

auto draw_fp_hud( const draw_fp_hud_options &options ) -> void
{
    auto &s = options.s;
    auto renderer = options.renderer;
    auto x = options.x;
    auto y = options.y;
    auto w = options.w;
    auto h = options.h;
    const auto &cam = options.cam;

    constexpr auto pi = static_cast<float>( 3.14159265f );
    // The compass strip shows half a turn: the view in the middle, what is beside you at the ends.
    constexpr auto compass_span = static_cast<float>( pi );
    const auto compass = std::string( get_option<std::string>( "R3D_FP_COMPASS" ) );
    const auto edge_alert = static_cast<bool>( get_option<bool>( "R3D_FP_EDGE_ALERT" ) );
    const auto &you = get_avatar();
    auto &here = get_map();
    const auto view_yaw = static_cast<float>( cam.fp_yaw() );
    auto bearing_to = [ &you, view_yaw ]( const tripoint_bub_ms & at ) {
        const auto d = tripoint_rel_ms( at - you.bub_pos() );
        return relative_bearing( static_cast<float>( d.x() ), static_cast<float>( d.y() ), view_yaw );
    };
    auto fill = [ renderer ]( float rx, float ry, float rw, float rh, uint8_t r, uint8_t g, uint8_t b,
    uint8_t a ) {
        SDL_SetRenderDrawBlendMode( renderer, SDL_BLENDMODE_BLEND );
        SDL_SetRenderDrawColor( renderer, r, g, b, a );
        const auto rect = SDL_FRect{ rx, ry, rw, rh };
        SDL_RenderFillRect( renderer, &rect );
    };
    // Labels are drawn by the caller in the game's font (see hud_texts), centred on tx.
    auto text = [ &s ]( float tx, float ty, const std::string & str, unsigned char color ) {
        s.hud.push_back( { tx, ty, str, color, true } );
    };
    constexpr auto white = static_cast<unsigned char>( 15 );
    constexpr auto light_gray = static_cast<unsigned char>( 7 );
    const auto cell_h = static_cast<float>( static_cast<float>( s.hud_cell_h ) );

    if( compass != "off" ) {
        const auto strip_w = static_cast<float>( std::max( 120.0f, w * 0.4f ) );
        // Room for one line of text above the ticks' footing.
        const auto strip_h = static_cast<float>( std::max( 22.0f, cell_h + 8.0f ) );
        const auto left = static_cast<float>( x + ( w - strip_w ) * 0.5f );
        const auto top = static_cast<float>( y + 6.0f );
        const auto mid = static_cast<float>( left + strip_w * 0.5f );
        auto strip_x = [ mid, strip_w, compass_span ]( float rel ) {
            return mid + compass_offset( rel, strip_w, compass_span );
        };
        fill( left, top, strip_w, strip_h, 0, 0, 0, 110 );
        static const char *const names[8] = { "N", "NE", "E", "SE", "S", "SW", "W", "NW" };
        for( auto i = static_cast<int>( 0 ); i < 8; ++i ) {
            const auto rel = static_cast<float>( wrap_angle( heading_bearing( i ) - view_yaw ) );
            if( !on_compass( rel, compass_span ) ) {
                continue;
            }
            const auto label = std::string( names[i] );
            const auto cardinal = static_cast<bool>( label.size() == 1 );
            text( strip_x( rel ), top + 1.0f, label, cardinal ? white : light_gray );
        }
        fill( mid - 1.0f, top + strip_h - 5.0f, 2.0f, 5.0f, 255, 255, 255, 255 );
        // Sounds heard this turn whose post cannot be seen: a short yellow tick each, dimmer
        // when they lie behind the strip's ends. Those in sight are posts on their tile.
        for( const tripoint_bub_ms &p : s.heard_ticks ) {
            const auto rel = static_cast<float>( bearing_to( p ) );
            fill( strip_x( rel ) - 2.0f, top + strip_h - 9.0f, 4.0f, 9.0f, 240, 210, 60,
                  on_compass( rel, compass_span ) ? 255 : 150 );
        }
        if( compass == "threats" ) {
            for( Creature *critter : you.get_hostile_creatures( MAX_VIEW_DISTANCE ) ) {
                if( critter->bub_pos().z() != you.bub_pos().z() ) {
                    continue;
                }
                const auto rel = static_cast<float>( bearing_to( critter->bub_pos() ) );
                const auto distance = static_cast<int>( std::max( 1, rl_dist( you.bub_pos(),
                                                        critter->bub_pos() ) ) );
                // Behind the player the tick sits at the end of the strip it is nearer to.
                const auto tick_h = static_cast<float>( std::clamp( 24.0f / static_cast<float>( distance ), 5.0f,
                                                        strip_h ) );
                const auto tick_w = static_cast<float>( distance == 1 ? 5.0f : 3.0f );
                fill( strip_x( rel ) - tick_w * 0.5f, top + strip_h - tick_h, tick_w, tick_h, 230, 40, 40,
                      on_compass( rel, compass_span ) ? 255 : 150 );
            }
        }
    }

    if( edge_alert ) {
        constexpr auto bar = static_cast<float>( 6.0f );
        for( Creature *critter : you.get_hostile_creatures( 1 ) ) {
            if( critter->bub_pos().z() != you.bub_pos().z() ) {
                continue;
            }
            switch( edge_alert_side( bearing_to( critter->bub_pos() ), cam.fp_fovx( w / h ) ) ) {
                case edge_side::none:
                    break;
                case edge_side::behind:
                    fill( x, y + h - bar, w, bar, 230, 40, 40, 200 );
                    break;
                case edge_side::left:
                    fill( x, y, bar, h, 230, 40, 40, 200 );
                    break;
                case edge_side::right:
                    fill( x + w - bar, y, bar, h, 230, 40, 40, 200 );
                    break;
            }
        }
    }

    // Look mode on another level: say which, under the compass strip.
    const auto looking = static_cast<int>( you.view_offset.z() );
    if( looking != 0 ) {
        const auto levels = static_cast<int>( std::abs( looking ) );
        const auto line = std::string( looking > 0 ?
                                       string_format( ngettext( "Looking: %d level up", "Looking: %d levels up",
                                               levels ), levels ) :
                                       string_format( ngettext( "Looking: %d level down", "Looking: %d levels down",
                                               levels ), levels ) );
        constexpr auto light_cyan = static_cast<unsigned char>( 14 );
        text( x + w * 0.5f, y + std::max( 22.0f, cell_h + 8.0f ) + 10.0f, line, light_cyan );
    }

    if( get_option<bool>( "R3D_FP_CROSSHAIR" ) ) {
        const auto cx = static_cast<float>( x + w * 0.5f );
        const auto cy = static_cast<float>( y + h * 0.5f );
        fill( cx - 2.0f, cy - 2.0f, 4.0f, 4.0f, 255, 255, 255, 200 );
        auto label = std::string{};
        const auto at = std::optional<tripoint_bub_ms>( crosshair() );
        // Only what the player sees or remembers is named: a black, unexplored wall stays unknown.
        if( at && ( you.sees( *at ) || !you.get_terrain_tile( bub_to_abs( *at ) ).tile.empty() ) ) {
            const Creature *critter = g->critter_at( *at, true );
            if( critter && you.sees( *critter ) ) {
                label = critter->disp_name();
            } else if( here.furn( *at ) != furn_str_id::NULL_ID() ) {
                label = here.furnname( *at );
            } else {
                label = here.tername( *at );
            }
        }
        if( !label.empty() ) {
            text( cx, cy + 8.0f, label, white );
        }
    }
}

} // namespace

auto draw( const draw_options &options ) -> bool
{
    auto renderer = options.renderer;
    auto x = options.x;
    auto y = options.y;
    auto w = options.w;
    auto h = options.h;
    const auto &center = options.center;
    auto render_epoch = options.render_epoch;

    auto &s = st();
    s.drawn = false;
    s.map_drawn = true;
    s.hud.clear();
    if( !get_option<bool>( "RENDERER_3D" ) ) {
        if( s.device.ready() ) {
            shutdown();
        }
        return false;
    }
    if( !tilecontext || tilecontext->current_tileset() == nullptr ) {
        return false;
    }

    // Anything the renderer rebuilt (device loss, software fallback, texture reupload) invalidates
    // every GPU object we hold, so start over against the new renderer.
    if( !s.device.ready() || s.device.renderer() != renderer || !s.epoch_valid ||
        s.epoch != render_epoch ) {
        if( s.init_failed && s.failed_epoch == render_epoch ) {
            return false;
        }
        shutdown();
        s.init_failed = false;
        if( !s.device.init( renderer ) ) {
            s.init_failed = true;
            s.failed_epoch = render_epoch;
            return false;
        }
        s.epoch = render_epoch;
        s.epoch_valid = true;
    }

    // The wall-side rules and atlas are always loaded (they name each wall's material); the
    // option only decides whether painted walls take the atlas plaster or their own sprite.
    // Switching it rebuilds every chunk.
    if( !s.walls_on ) {
        s.cache.set_wall_sides( load_wall_sides( s.device ) );
        s.walls_on = true;
    }
    const auto want_thin_walls = static_cast<bool>( get_option<bool>( "R3D_THIN_WALLS" ) );
    if( want_thin_walls != s.thin_walls ) {
        release_all_chunks( s );
        s.cache.set_thin_walls( want_thin_walls );
        s.thin_walls = want_thin_walls;
    }
    const auto want_atlas = static_cast<bool>( get_option<bool>( "R3D_WALL_TEXTURES" ) );
    if( want_atlas != s.plaster_atlas ) {
        release_all_chunks( s );
        s.cache.set_plaster_atlas( want_atlas );
        s.plaster_atlas = want_atlas;
    }

    // A tileset (re)load swaps the sheet textures and may lay the atlas out differently
    // without bumping the epoch: everything cached from the old sheets is dropped.
    if( tilecontext->tileset_loads() != s.tileset_loads ) {
        release_all_chunks( s );
        s.cache.clear();
        s.tileset_loads = tilecontext->tileset_loads();
    }

    ++s.frame;
    maybe_take_screenshot( s );

    sync_camera_mode( s.cam );
    sync_saved_heading( s );
    // Meshes differ between the modes (see mesh_options), so a switch rebuilds every chunk.
    if( s.cam.first_person() != s.chunks_first_person ) {
        release_all_chunks( s );
        s.chunks_first_person = s.cam.first_person();
    }
    if( !s.env_camera_applied ) {
        apply_env_camera( s.cam );
        s.env_camera_applied = true;
    }

    // The scene is rendered at a multiple of the window size and averaged down when composited,
    // which keeps distant textures from turning to grainy noise. CATA_R3D_SUPERSAMPLE=1 turns
    // it off (it costs fill rate), larger values go up to 3.
    static const auto supersample = static_cast<int>( []() -> int {
        const char *env = std::getenv( "CATA_R3D_SUPERSAMPLE" );
        return env ? std::clamp( std::atoi( env ), 1, 3 ) : 2;
    }
    () );
    const auto target_w = static_cast<int>( std::max( 1, static_cast<int>( w ) ) * supersample );
    const auto target_h = static_cast<int>( std::max( 1, static_cast<int>( h ) ) * supersample );
    if( !s.device.ensure_target( target_w, target_h ) ) {
        return false;
    }

    auto &here = get_map();
    if( std::getenv( "CATA_R3D_DEBUG_MAP" ) && s.frame < 4 ) {
        const auto &ch = here.access_cache( center.z() );
        std::cerr << "r3d probe center=" << center.to_string() << " ter=" << here.ter( center ).id().str()
                  << " ll=" << int( ch.visibility_cache[ch.idx( center.x(), center.y() )] )
                  << " generation=" << ch.visibility_generation
                  << " move=" << here.ter( center )->movecost
                  << " wall=" << here.ter( center )->has_flag( TFLAG_WALL )
                  << " cached_kind=" << int( s.cache.terrain( { .id = here.ter( center ), .tiles = *tilecontext } ).sh.kind )
                  << "\n";
    }
    const auto origin = tripoint_abs_ms( bub_to_abs( center ) );
    const auto view_z = static_cast<int>( origin.z() );
    const auto multi_z = static_cast<bool>( true );
    const auto depth = static_cast<int>( multi_z ? get_option<int>( "R3D_VIEW_DEPTH" ) : 0 );
    // Outdoors the storeys above are drawn too, so buildings stand at full height; under a roof
    // they are cut away to show the room.
    constexpr auto max_levels_above = static_cast<int>( 5 );
    const auto overhead = tripoint_bub_ms( center + tripoint_rel_ms( 0, 0, 1 ) );
    const auto open_sky = static_cast<bool>( !multi_z || !here.inbounds( overhead ) ||
                          s.cache.terrain( { .id = here.ter( overhead ), .tiles = *tilecontext } ).sh.kind ==
                          shape_kind::empty );
    const auto fp = s.cam.first_person();
    const auto player_pos = tripoint_bub_ms( get_avatar().bub_pos() );
    // In first person the eye stays on the player even when look mode moves the viewed tile, so
    // the levels cover both, and the storeys above are always drawn (they are the ceilings).
    const auto low_z = fp ? std::min( view_z, player_pos.z() ) : view_z;
    const auto high_z = fp ? std::max( view_z, player_pos.z() ) : view_z;
    const auto z_hi = multi_z && ( open_sky ||
                                   fp ) ? std::min( high_z + max_levels_above, OVERMAP_HEIGHT ) : high_z;
    const auto z_lo = static_cast<int>( std::max( low_z - depth, -OVERMAP_DEPTH ) );
    const auto base = tripoint_abs_sm( here.get_abs_sub(), view_z );

    // Only submaps inside the reality bubble: looking anything else up could load it from disk.
    const auto lookup = submap_lookup( [ &base,
    view_z ]( const tripoint_abs_sm & p ) -> const submap * {
        return loaded_submap_at( { .base = base, .view_z = view_z, .multi_z = multi_z, .p = p } );
    } );

    const auto sampler = light_sampler{ .here = here, .vvc = here.get_visibility_variables_cache(), .you = get_avatar(),
                                        .vision_floor = -OVERMAP_DEPTH, .vision_top = OVERMAP_HEIGHT, .view_z = view_z };
    const auto frame_light_key = static_cast<uint64_t>( light_key( { .here = here, .origin = origin, .z_lo = z_lo, .z_hi = z_hi } ) );

    // The scene is drawn in a frame whose origin is the viewed tile, which keeps floats small.
    s.cam.z_scale = static_cast<float>( get_option<int>( "R3D_LEVEL_HEIGHT" ) ) / 100.0f;
    if( fp ) {
        auto ease_x = static_cast<float>( 0.0f );
        auto ease_y = static_cast<float>( 0.0f );
        auto ease_z = static_cast<float>( 0.0f );
        const auto stand = static_cast<float>( surface_height( s, here, player_pos ) );
        update_fp_ease( { .s = s, .pos = bub_to_abs( player_pos ), .surface = stand, .offset_x = ease_x, .offset_y = ease_y, .offset_z = ease_z } );
        // The camera stands on the player's tile, at the height of what they stand on.
        s.cam.target = tile_to_world( {
            static_cast<float>( player_pos.x() - center.x() ) + 0.5f + ease_x,
            static_cast<float>( player_pos.y() - center.y() ) + 0.5f + ease_y,
            static_cast<float>( player_pos.z() - center.z() ) + stand + ease_z
        } );
        // Looking at another level (look mode's < and >) tilts the view towards it, 45 degrees
        // per level at most one level's worth; the eye stays where the player stands.
        constexpr auto level_tilt = static_cast<float>( 0.7854f );
        s.cam.tilt_pitch = static_cast<float>( std::clamp( view_z - player_pos.z(), -1, 1 ) ) *
                           level_tilt;
        s.cam.fovx = static_cast<float>( get_option<int>( "R3D_FP_FOV" ) ) * 0.017453293f;
        s.cam.eye_height = static_cast<float>( get_option<int>( "R3D_FP_EYE_HEIGHT" ) ) / 100.0f;
    } else {
        s.fp_prev_valid = false;
        s.fp_easing = false;
        s.cam.tween_yaw = 0.0f;
        s.cam.tilt_pitch = 0.0f;
        s.cam.target = tile_to_world( { .x = 0.5f, .y = 0.5f, .z = 0.5f } ) + s.pan;
    }
    const auto aspect = static_cast<float>( static_cast<float>( target_w ) / static_cast<float>
                                            ( target_h ) );
    const auto view_proj = mat4( s.cam.view_projection( aspect ) );
    const auto fr = frustum( s.cam.view_frustum( aspect ) );

    if( !s.device.begin_frame() ) {
        return false;
    }

    struct visible_chunk {
        const chunk_gpu *chunk;
        float offset[3];
    };
    auto visible = std::vector<visible_chunk> {};
    auto upright_list = std::vector<entity_view::standing> {};
    // A newly opened view must be complete before waiting for the player's next key.
    // Later edits retain the normal bounded rebuild budget.
    auto builds_left = s.chunks.empty() ? std::numeric_limits<int>::max() : max_chunk_builds_per_frame;
    for( auto z = static_cast<int>( z_lo ); z <= z_hi; ++z ) {
        for( auto gy = static_cast<int>( 0 ); gy < here.getmapsize(); ++gy ) {
            for( auto gx = static_cast<int>( 0 ); gx < here.getmapsize(); ++gx ) {
                const auto pos = tripoint_abs_sm( base.x() + gx, base.y() + gy, z );
                const auto chunk_origin = tripoint_abs_ms( project_to<coords::ms>( pos ) );
                const auto ox = static_cast<float>( chunk_origin.x() - origin.x() );
                const auto oy = static_cast<float>( -static_cast<float>( chunk_origin.y() - origin.y() ) );
                const auto oz = static_cast<float>( z - view_z );
                const auto box = aabb{ .min = { .x = ox, .y = oy - SEEY, .z = oz }, .max = { .x = ox + SEEX, .y = oy, .z = oz + chunk_max_height } };
                if( !fr.intersects( box ) ) {
                    continue;
                }
                const auto stamp = static_cast<uint64_t>( chunk_stamp( pos, lookup ) );
                if( stamp == 0 ) {
                    continue; // not loaded
                }
                auto &c = s.chunks[chunk_key{ .x = pos.x(), .y = pos.y(), .z = pos.z() }];
                c.last_used = s.frame;
                if( ( !c.built || c.stamp != stamp ) && builds_left > 0 ) {
                    --builds_left;
                    rebuild_chunk( { .s = s, .c = c, .pos = pos, .stamp = stamp, .lookup = lookup } );
                }
                if( c.built && !c.batches.empty() ) {
                    if( c.light_key != frame_light_key ) {
                        update_chunk_light( { .s = s, .c = c, .pos = pos, .origin = origin, .center = center, .sampler = sampler, .key = frame_light_key } );
                    }
                    visible.push_back( { .chunk = &c, .offset = { ox, oy, oz } } );
                }
                if( c.built ) {
                    const auto chunk_base = tripoint_bub_ms( center.x() + ( chunk_origin.x() - origin.x() ),
                                            center.y() + ( chunk_origin.y() - origin.y() ), z );
                    for( const upright_sprite &u : c.uprights ) {
                        upright_list.push_back( { .pos = chunk_base + tripoint_rel_ms( static_cast<int>( u.lx ), static_cast<int>( u.ly ), 0 ), .z_base = u.z_base,
                                                  .info = u.info } );
                    }
                }
            }
        }
    }

    // Creatures and items change every frame, so their geometry is rebuilt and re-uploaded.
    auto entities_view = entity_view{ .tiles = *tilecontext, .cache = s.cache, .cam = s.cam, .center = center, .z_lo = z_lo, .z_hi = z_hi,
                                      .light = [&sampler]( const tripoint_bub_ms & p )
    -> uint32_t {
        return sampler.color_at( p );
    }, .markers = {}, .uprights = std::move( upright_list ), .surface = [&s, &here]( const tripoint_bub_ms & p )
    -> float {
        return surface_height( s, here, p );
    } };
    // The game queues its cursors and highlights for cata_tiles::draw, which is skipped here.

    entities_view.upright_billboards = fp;
    // The game queues its cursors and highlights for cata_tiles::draw, which is skipped here.
    {
        const auto queued = cata_tiles::frame_markers( tilecontext->get_frame_markers() );
        const auto add_markers = [ &s, &here, &entities_view ]( const std::vector<tripoint_bub_ms> &where,
                                 const std::string & id,
        bool box, bool large ) -> void {
            append_frame_markers( { .s = s, .here = here, .entities_view = entities_view, .where = where, .id = id, .box = box }, large );
        };
        add_markers( queued.line, "animation_line", false, false );
        if( queued.line_end && !queued.line_end_id.empty() ) {
            add_markers( { *queued.line_end }, queued.line_end_id, false, true );
        }
        add_markers( queued.highlights, "highlight", false, true );
        add_markers( queued.cursors, "cursor", true, true );

        // Sounds heard this turn, which cata_tiles::draw_footsteps_frame would draw: posts on
        // their tile with normal depth, so walls and creatures in front hide them. The 2D view
        // draws them over everything; here the compass strip points at the hidden ones instead
        // (out of the picture, or behind something solid).
        s.heard_ticks.clear();
        if( fp ) {
            const auto player_z = static_cast<int>( player_pos.z() );
            const auto occlude_unseen = static_cast<bool>( get_option<bool>( "R3D_FP_OCCLUDE_UNSEEN" ) );
            // What hides a post: terrain and furniture as drawn. Unseen places only when they
            // are drawn solid (R3D_FP_OCCLUDE_UNSEEN); creatures never fill a whole tile.
            const auto solid = column_height_fn( [ &center, z_lo, z_hi, &here, occlude_unseen, &sampler,
                                                            &s ]( int dx,
            int dy, int dz ) {
                return occluding_column_height( { .s = s, .here = here, .sampler = sampler, .center = center, .z_lo = z_lo, .z_hi = z_hi, .occlude_unseen = occlude_unseen, .dx = dx, .dy = dy, .dz = dz } );
            } );
            const auto eye_world = vec3( s.cam.eye() );
            const auto eye_tile = vec3( world_to_tile( { eye_world.x, eye_world.y,
                                        eye_world.z / s.cam.z_scale } ) );
            const auto view_yaw = static_cast<float>( s.cam.fp_yaw() );
            for( const tripoint_bub_ms &p : sounds::get_footstep_markers() ) {
                const auto d = tripoint_rel_ms( p - player_pos );
                const auto rel = static_cast<float>( relative_bearing( static_cast<float>( d.x() ),
                                                     static_cast<float>( d.y() ), view_yaw ) );
                // The middle of the post (see r3d_entities.cpp), in the scene's tile space.
                const auto post = vec3{ static_cast<float>( p.x() - center.x() ) + 0.5f,
                                        static_cast<float>( p.y() - center.y() ) + 0.5f,
                                        static_cast<float>( p.z() - center.z() ) + column_height( s, here, p ) + 0.4f };
                if( std::abs( rel ) > s.cam.fp_fovx( aspect ) * 0.5f ||
                    !clear_line( eye_tile, post, solid ) ) {
                    s.heard_ticks.push_back( p );
                }
                const char *const id = p.z() > player_z ? "footstep_above" :
                                       p.z() < player_z ? "footstep_below" : "footstep";
                const tile_info *sprite = &s.cache.named( { .category = C_NONE, .id = id, .tiles = *tilecontext } );
                if( !sprite->main().valid ) {
                    sprite = &s.cache.named( { .category = C_NONE, .id = "footstep", .tiles = *tilecontext } );
                }
                if( !sprite->main().valid ) {
                    sprite = &s.cache.named( { .category = C_NONE, .id = "highlight", .tiles = *tilecontext } );
                }
                entities_view.markers.push_back( { p, column_height( s, here, p ), sprite, false, false,
                                                   true } );
            }
        }
    }
    const auto entities = chunk_mesh( build_entity_mesh( entities_view ) );
    const auto draw_entities = static_cast<bool>( !entities.empty() &&
                               upload_dynamic( { .s = s, .buffer = s.dyn_positions, .capacity = s.dyn_position_bytes, .usage = SDL_GPU_BUFFERUSAGE_VERTEX,
                                       .data = entities.vertices.data(),
                                       .bytes = static_cast<uint32_t>( entities.vertices.size() * sizeof( vertex ) ) } ) &&
                               upload_dynamic( { .s = s, .buffer = s.dyn_lights, .capacity = s.dyn_light_bytes, .usage = SDL_GPU_BUFFERUSAGE_VERTEX,
                                       .data = entities.light.data(),
                                       .bytes = static_cast<uint32_t>( entities.light.size() * sizeof( uint32_t ) ) } ) &&
                               upload_dynamic( { .s = s, .buffer = s.dyn_indices, .capacity = s.dyn_index_bytes, .usage = SDL_GPU_BUFFERUSAGE_INDEX,
                                       .data = entities.indices.data(),
                                       .bytes = static_cast<uint32_t>( entities.indices.size() * sizeof( uint32_t ) ) } ) );

    float sky[3] = { 0.05f, 0.05f, 0.07f  };
    if( fp ) {
        const auto day = static_cast<float>( std::clamp( sunlight( calendar::turn ) / static_cast<float>
                                             ( default_daylight_level() ),
                                             0.0f, 1.0f ) );
        constexpr float night_sky[3] = { 0.02f, 0.025f, 0.05f };
        constexpr float day_sky[3] = { 0.50f, 0.65f, 0.85f };
        for( auto i = static_cast<int>( 0 ); i < 3; ++i ) {
            sky[i] = night_sky[i] + ( day_sky[i] - night_sky[i] ) * day;
        }
    }
    if( !s.device.begin_pass( sky[0], sky[1], sky[2] ) ) {
        s.device.abort_frame();
        // This frame's uploads may not have happened, so no chunk's buffers can be trusted.
        release_all_chunks( s );
        return false;
    }
    s.device.bind_block_pipeline();
    auto uniforms = block_uniforms{};
    uniforms.view_proj = view_proj;
    set_cutaway( { .u = uniforms, .view_proj = view_proj, .aspect = aspect, .cam = s.cam, .center = center } );
    // First person: never-seen places hide what lies behind them (R3D_FP_OCCLUDE_UNSEEN).
    uniforms.flags[0] = fp && get_option<bool>( "R3D_FP_OCCLUDE_UNSEEN" ) ? 1.0f : 0.0f;
    for( const visible_chunk &vc : visible ) {
        uniforms.offset[0] = vc.offset[0];
        uniforms.offset[1] = vc.offset[1];
        uniforms.offset[2] = vc.offset[2];
        for( const mesh_batch &b : vc.chunk->batches ) {
            s.device.draw_block( { .positions = vc.chunk->positions, .lights = vc.chunk->lights, .indices = vc.chunk->indices,
                                   .first_index = b.first_index, .index_count = b.index_count, .sheet = b.sheet, .uniforms = uniforms } );
        }
    }
    if( draw_entities ) {
        s.device.bind_entity_pipeline();
        uniforms.cut[3] = 0.0f; // creatures in front of the player stay
        uniforms.flags[0] = 0.0f;
        uniforms.offset[0] = 0.0f;
        uniforms.offset[1] = 0.0f;
        uniforms.offset[2] = 0.0f;
        // Billboard tops get their depth pulled towards the camera (world direction: the
        // stretched view direction with its height shrunk back).
        const auto toward = vec3( normalize( s.cam.eye() - s.cam.stretched_target() ) );
        uniforms.pull[0] = toward.x;
        uniforms.pull[1] = toward.y;
        uniforms.pull[2] = toward.z / s.cam.z_scale;
        uniforms.pull[3] =  fp ? 0.0f : max_billboard_pull;
        for( const mesh_batch &b : entities.batches ) {
            s.device.draw_block( { .positions = s.dyn_positions, .lights = s.dyn_lights, .indices = s.dyn_indices, .first_index = b.first_index, .index_count = b.index_count,
                                   .sheet = b.sheet, .uniforms = uniforms } );
        }
    }
    SDL_Texture *const result = s.device.end_frame();
    if( !result ) {
        return false;
    }
    const auto dst = SDL_FRect{ .x = x, .y = y, .w = w, .h = h };
    SDL_RenderTexture( renderer, result, nullptr, &dst );

    // Drop chunks that have not been needed for a while (the player moved on).
    for( auto it = s.chunks.begin(); it != s.chunks.end(); ) {
        if( s.frame - it->second.last_used > chunk_idle_frames ) {
            release_chunk( s, it->second );
            it = s.chunks.erase( it );
        } else {
            ++it;
        }
    }
    // Only now that the frame is on screen: on failure the flat view draws the markers instead.
    tilecontext->drop_frame_layers();
    s.shown = { .valid = true, .x = x, .y = y, .w = w, .h = h, .cam = s.cam, .center = center, .z_lo = z_lo, .z_hi = z_hi };
    s.drawn = true;

    if( fp ) {
        draw_fp_hud( { .s = s, .renderer = renderer, .x = x, .y = y, .w = w, .h = h, .cam = s.cam } );
    }
    return true;
}

} // namespace r3d

#endif // defined(TILES)
