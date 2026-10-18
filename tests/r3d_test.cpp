#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "catch/catch.hpp"
#include "r3d_camera.h"
#include "r3d_dda.h"

#include "r3d_hud_math.h"
#include "r3d_math.h"
#include "r3d_shapes.h"
#include "r3d_wall_rules.h"

namespace
{

bool approx( float a, float b, float eps = 1e-3f )
{
    return std::fabs( a - b ) <= eps;
}

} // namespace

TEST_CASE( "r3d_matrix_inverse_round_trips", "[r3d]" )
{
    auto cam = r3d::camera{};
    cam.target = { 3.0f, -7.0f, 2.0f };
    cam.yaw = 0.7f;
    const auto vp = cam.view_projection( 16.0f / 9.0f );
    auto inv = r3d::mat4{};
    REQUIRE( r3d::invert( vp, inv ) );
    const auto id = vp * inv;
    for( auto r = 0; r < 4; ++r ) {
        for( auto c = 0; c < 4; ++c ) {
            CHECK( approx( id.at( r, c ), r == c ? 1.0f : 0.0f, 1e-2f ) );
        }
    }
}

TEST_CASE( "r3d_camera_looks_at_target", "[r3d]" )
{
    for( bool ortho : {
             false, true
         } ) {
        auto cam = r3d::camera{};
        cam.orthographic = ortho;
        cam.target = { 10.5f, -4.5f, 1.5f };
        cam.yaw = 1.1f;
        cam.pitch = 0.8f;
        cam.distance = 30.0f;
        const auto aspect = 1.6f;

        // Target projects to the middle of the screen and in front of the camera.
        const auto ndc = r3d::transform_point( cam.view_projection( aspect ), cam.target );
        CHECK( approx( ndc.x, 0.0f ) );
        CHECK( approx( ndc.y, 0.0f ) );
        CHECK( ndc.z > 0.0f );
        CHECK( ndc.z < 1.0f );

        // The centre ray passes through the target.
        const auto centre = cam.screen_ray( 0.0f, 0.0f, aspect );
        const auto to_target = cam.target - centre.origin;
        const auto along = r3d::dot( to_target, centre.dir );
        const auto closest = centre.origin + centre.dir * along;
        CHECK( r3d::length( closest - cam.target ) < 1e-2f );
    }
}

TEST_CASE( "r3d_camera_screen_ray_round_trips_projection", "[r3d]" )
{
    auto cam = r3d::camera{};
    cam.yaw = -0.4f;
    const auto aspect = 1.78f;
    const auto vp = cam.view_projection( aspect );
    for( float nx : {
             -0.8f, 0.0f, 0.5f
             } ) {
        for( float ny : {
                 -0.6f, 0.2f, 0.9f
                 } ) {
            const auto r = cam.screen_ray( nx, ny, aspect );
            // Any point along the ray must project back to the same screen position.
            const auto p = r.origin + r.dir * 20.0f;
            const auto ndc = r3d::transform_point( vp, p );
            CHECK( approx( ndc.x, nx, 1e-2f ) );
            CHECK( approx( ndc.y, ny, 1e-2f ) );
        }
    }
}

TEST_CASE( "r3d_camera_orientation_matches_tile_space", "[r3d]" )
{
    // yaw 0: camera south of the target looking north, so north (tile -y) is up on screen and east
    // (tile +x) is to the right.
    auto cam = r3d::camera{};
    cam.yaw = 0.0f;
    cam.target = r3d::tile_to_world( { 0.5f, 0.5f, 0.0f } );
    const auto vp = cam.view_projection( 1.0f );
    const auto centre = r3d::transform_point( vp, cam.target );
    const auto east = r3d::transform_point( vp, r3d::tile_to_world( { 3.5f, 0.5f, 0.0f } ) );
    const auto north = r3d::transform_point( vp, r3d::tile_to_world( { 0.5f, -2.5f, 0.0f } ) );
    CHECK( east.x > centre.x );
    CHECK( north.y > centre.y );

    // A quarter turn clockwise (seen from above) puts the camera east of the target looking west:
    // west is now straight ahead (up on screen) and north is to the right.
    cam.yaw = 1.5707963f;
    const auto vp2 = cam.view_projection( 1.0f );
    const auto c2 = r3d::transform_point( vp2, cam.target );
    const auto west = r3d::transform_point( vp2, r3d::tile_to_world( { -2.5f, 0.5f, 0.0f } ) );
    const auto north2 = r3d::transform_point( vp2, r3d::tile_to_world( { 0.5f, -2.5f, 0.0f } ) );
    CHECK( west.y > c2.y );
    CHECK( north2.x > c2.x );
}

TEST_CASE( "r3d_camera_forward_eighths", "[r3d]" )
{
    // yaw 0 puts the camera south looking north; yaw grows clockwise seen from above, which
    // swings the camera east of the target, looking west.
    auto cam = r3d::camera{};
    cam.yaw = 0.0f;
    CHECK( cam.forward_eighths() == 0 );
    cam.yaw = 1.5707963f;
    CHECK( cam.forward_eighths() == 6 );
    cam.yaw = 3.1415927f;
    CHECK( cam.forward_eighths() == 4 );
    cam.yaw = -1.5707963f;
    CHECK( cam.forward_eighths() == 2 );
    cam.yaw = 0.7853982f;
    CHECK( cam.forward_eighths() == 7 );
    // Taller storeys or pitch do not change the heading.
    cam.z_scale = 1.6f;
    cam.pitch = 1.4f;
    CHECK( cam.forward_eighths() == 7 );
}

TEST_CASE( "r3d_thin_walls_join_at_a_uniform_thickness", "[r3d]" )
{
    // Check every combination, including corners, T junctions and crossings. Only the
    // centre strip reaches a connected neighbour; diagonal cells never fatten a corner.
    for( auto mask = 0; mask < 16; ++mask ) {
        const auto connections = static_cast<uint8_t>( mask << 1 );
        const auto cells = r3d::wall_footprint( connections );
        CHECK( cells[4] );
        for( int corner : {
                 0, 2, 6, 8
             } ) {
            CHECK_FALSE( cells[corner] );
        }
        const auto vertical_end = connections == r3d::face_north || connections == r3d::face_south;
        const auto horizontal_end = connections == 0 || connections == r3d::face_west ||
                                    connections == r3d::face_east;
        CHECK( cells[1] == ( vertical_end || ( connections & r3d::face_north ) != 0 ) );
        CHECK( cells[7] == ( vertical_end || ( connections & r3d::face_south ) != 0 ) );
        CHECK( cells[3] == ( horizontal_end || ( connections & r3d::face_west ) != 0 ) );
        CHECK( cells[5] == ( horizontal_end || ( connections & r3d::face_east ) != 0 ) );
    }
}

TEST_CASE( "r3d_thin_walls_trim_only_indoor_corner_spurs", "[r3d]" )
{
    using namespace r3d;
    const auto all = face_north | face_south | face_west | face_east;
    // Inward spur joins the west side of a north/east corner, as in shelter_2.
    CHECK( trim_wall_spur( face_east, all & ~face_east, face_west | face_north | face_east ) );
    CHECK( trim_wall_spur( face_west, all & ~face_west, face_east | face_south | face_west ) );
    CHECK( trim_wall_spur( face_north, all & ~face_north, face_south | face_north | face_east ) );
    CHECK( trim_wall_spur( face_south, all & ~face_south, face_north | face_south | face_west ) );
    // Preserve outdoor ends, freestanding partitions, straight walls and true T junctions.
    CHECK_FALSE( trim_wall_spur( face_east, 0, face_west | face_north | face_east ) );
    CHECK_FALSE( trim_wall_spur( face_east, all & ~face_east, face_west | face_east ) );
    CHECK_FALSE( trim_wall_spur( face_east, all & ~face_east, face_west | face_north | face_south ) );
    CHECK_FALSE( trim_wall_spur( face_north | face_south, face_east | face_west, all ) );
    CHECK_FALSE( trim_wall_spur( 0, all, all ) );
}

TEST_CASE( "r3d_corner_locker_faces_into_the_room", "[r3d]" )
{
    // At the south-west corner the solitary locker backs onto the south wall,
    // a quarter turn anticlockwise from its previous west-wall orientation.
    CHECK( r3d::furniture_wall_facing( false, false, r3d::face_south | r3d::face_west ) == 2 );
    CHECK( r3d::furniture_wall_facing( false, true, r3d::face_south | r3d::face_west ) == 3 );
    CHECK( r3d::furniture_wall_facing( true, false, r3d::face_north | r3d::face_east ) == 0 );
}

TEST_CASE( "r3d_notice_board_is_a_panel", "[r3d]" )
{
    auto traits = r3d::tile_traits{};
    traits.move_cost = 0;
    traits.panel = true;
    const auto board = r3d::classify_furniture( traits );
    CHECK( board.kind == r3d::shape_kind::pane );
    CHECK( board.height > r3d::cutaway_stub_height );
}

TEST_CASE( "r3d_camera_rotation_matches_movement_directions", "[r3d]" )
{
    for( int direction : {
             -1, 1
             } ) {
        auto cam = r3d::camera{};
        for( auto step = 1; step <= 16; ++step ) {
            cam.rotate_steps( direction );
            const auto heading = ( ( -direction * step ) % 8 + 8 ) % 8;
            CHECK( cam.forward_eighths() == heading );
            const auto angle = -heading * 0.78539816f;
            const auto right = cam.ground_right();
            CHECK( approx( right.x, std::cos( angle ) ) );
            CHECK( approx( right.y, std::sin( angle ) ) );
        }
    }
}

TEST_CASE( "r3d_camera_rotation_aligns_after_free_orbit", "[r3d]" )
{
    auto cam = r3d::camera{};
    cam.yaw = 0.3f;
    cam.rotate_steps( 1 );
    CHECK( approx( cam.yaw, 0.78539816f ) );
    cam.yaw = -0.3f;
    cam.rotate_steps( -1 );
    CHECK( approx( cam.yaw, -0.78539816f ) );
}

TEST_CASE( "r3d_camera_steps_clamp", "[r3d]" )
{
    auto cam = r3d::camera{};
    cam.pitch_steps( 1000 );
    CHECK( cam.pitch <= r3d::camera::max_pitch );
    cam.pitch_steps( -1000 );
    CHECK( cam.pitch >= r3d::camera::min_pitch );
    cam.zoom_steps( 1000 );
    CHECK( cam.distance >= r3d::camera::min_distance );
    cam.zoom_steps( -1000 );
    CHECK( cam.distance <= r3d::camera::max_distance );

    cam.yaw = 0.0f;
    cam.rotate_steps( 8 ); // 8 * 45 degrees = a full turn
    CHECK( ( approx( cam.yaw, 0.0f, 0.05f ) || approx( std::fabs( cam.yaw ), 6.2831853f, 0.05f ) ) );
}

TEST_CASE( "r3d_frustum_culls_boxes", "[r3d]" )
{
    auto cam = r3d::camera{};
    cam.target = { 0.0f, 0.0f, 0.0f };
    const auto f = cam.view_frustum( 1.5f );
    CHECK( f.intersects( { { -0.5f, -0.5f, -0.5f }, { 0.5f, 0.5f, 0.5f } } ) );
    CHECK_FALSE( f.intersects( { { 500.0f, 0.0f, 0.0f }, { 501.0f, 1.0f, 1.0f } } ) );
    CHECK_FALSE( f.intersects( { { 0.0f, 0.0f, 900.0f }, { 1.0f, 1.0f, 901.0f } } ) );
}

TEST_CASE( "r3d_raycast_hits_floor_tile_under_ray", "[r3d]" )
{
    // A flat floor of height 0.1 at z = 0 over x, y in [0, 20).
    const auto floor_world = []( int x, int y, int z ) {
        return z == 0 && x >= 0 && x < 20 && y >= 0 && y < 20 ? 0.1f : 0.0f;
    };
    // Straight down onto tile (7, 5).
    const auto down = r3d::raycast_columns( { .origin = { 7.5f, 5.5f, 10.0f }, .dir = { 0.0f, 0.0f, -1.0f },
                                            .max_t = 100.0f, .height = floor_world } );
    REQUIRE( down.hit );
    CHECK( down.x == 7 );
    CHECK( down.y == 5 );
    CHECK( down.z == 0 );
    CHECK( down.face == r3d::hit_face::top );
    CHECK( approx( down.t, 9.9f ) );

    // A ray pointing away misses.
    const auto up = r3d::raycast_columns( { .origin = { 7.5f, 5.5f, 10.0f }, .dir = { 0.0f, 0.0f, 1.0f },
                                            .max_t = 100.0f, .height = floor_world } );
    CHECK_FALSE( up.hit );

    // Out of range misses.
    const auto short_ray = r3d::raycast_columns( { .origin = { 7.5f, 5.5f, 10.0f }, .dir = { 0.0f, 0.0f, -1.0f },
                           .max_t = 5.0f, .height = floor_world } );
    CHECK_FALSE( short_ray.hit );
}

TEST_CASE( "r3d_raycast_wall_blocks_ground_behind_it", "[r3d]" )
{
    const auto world = []( int x, int y, int z ) {
        if( z != 0 || x < 0 || x >= 20 || y < 0 || y >= 20 ) {
            return 0.0f;
        }
        return x == 10 ? 1.0f : 0.1f; // a full-height wall along x = 10
    };
    // Looking east and gently down: the ray is still above the wall's top when it reaches x = 10,
    // so it must hit the wall's west face before it ever reaches the floor.
    const auto dir = r3d::normalize( { 1.0f, 0.0f, -0.05f } );
    const auto h = r3d::raycast_columns( { .origin = { 2.5f, 8.5f, 0.9f }, .dir = dir, .max_t = 100.0f, .height = world } );
    REQUIRE( h.hit );
    CHECK( h.x == 10 );
    CHECK( h.face == r3d::hit_face::west );
}

TEST_CASE( "r3d_pick_finds_tile_drawn_under_cursor", "[r3d]" )
{
    // Floor everywhere on level 0 of the scene frame and a wall at tile (2, -3).
    const auto world = []( int x, int y, int z ) {
        if( z != 0 ) {
            return 0.0f;
        }
        return x == 2 && y == -3 ? 1.0f : 0.1f;
    };
    for( float yaw : {
             0.0f, 0.9f, 2.6f, 4.4f
         } ) {
        for( auto variant = 0; variant < 3; ++variant ) {
            const auto ortho = variant == 1;
            auto cam = r3d::camera{};
            cam.target = r3d::tile_to_world( { 0.5f, 0.5f, 0.5f } ); // as r3d::draw sets it
            cam.yaw = yaw;
            cam.orthographic = ortho;
            cam.z_scale = variant == 2 ? 1.6f : 1.0f; // taller storeys must not change picks
            const auto aspect = 1.5f;
            const auto vp = cam.view_projection( aspect );
            // The middle of the screen is the viewed tile.
            const auto centre = r3d::pick_cell( { .cam = cam, .ndc_x = 0.0f, .ndc_y = 0.0f, .aspect = aspect, .max_t = 500.0f, .height = world,
                                                  .floor_z = r3d::floor_height } );
            REQUIRE( centre.hit );
            CHECK( centre.x == 0 );
            CHECK( centre.y == 0 );
            CHECK( centre.z == 0 );
            // Where a floor tile's centre is drawn, that tile is picked (unless the wall covers it).
            for( auto ty = -4; ty <= 4; ++ty ) {
                for( auto tx = -4; tx <= 4; ++tx ) {
                    const auto top = r3d::vec3{ tx + 0.5f, ty + 0.5f, world( tx, ty, 0 ) };
                    const auto ndc = r3d::transform_point( vp, r3d::tile_to_world( top ) );
                    const auto h = r3d::pick_cell( { .cam = cam, .ndc_x = ndc.x, .ndc_y = ndc.y, .aspect = aspect, .max_t = 500.0f, .height = world,
                                                     .floor_z = r3d::floor_height } );
                    REQUIRE( h.hit );
                    const auto is_tile = h.x == tx && h.y == ty && h.z == 0;
                    const auto is_wall = h.x == 2 && h.y == -3;
                    CHECK( ( is_tile || is_wall ) );
                }
            }
        }
    }
}

TEST_CASE( "r3d_pick_falls_back_to_floor_plane", "[r3d]" )
{
    const auto nothing = []( int, int, int ) {
        return 0.0f;
    };
    auto cam = r3d::camera{};
    cam.target = r3d::tile_to_world( { 0.5f, 0.5f, 0.5f } );
    const auto h = r3d::pick_cell( { .cam = cam, .ndc_x = 0.0f, .ndc_y = 0.0f, .aspect = 1.0f, .max_t = 500.0f, .height = nothing,
                                     .floor_z = r3d::floor_height } );
    REQUIRE( h.hit );
    CHECK( h.x == 0 );
    CHECK( h.y == 0 );
    CHECK( h.z == 0 );
    CHECK( h.face == r3d::hit_face::top );
    // Above the horizon nothing can be picked.
    cam.pitch = r3d::camera::min_pitch;
    CHECK_FALSE( r3d::pick_cell( { .cam = cam, .ndc_x = 0.0f, .ndc_y = 1.0f, .aspect = 1.0f, .max_t = 500.0f, .height = nothing, .floor_z = r3d::floor_height } ).hit );
}

TEST_CASE( "r3d_raycast_matches_reference", "[r3d]" )
{
    auto rng = std::mt19937( 12345 );
    auto kind = std::uniform_int_distribution<int>( 0, 9 );
    auto unit = std::uniform_real_distribution<float>( 0.0f, 1.0f );

    const auto size = 12;
    const auto layers = 5;
    for( auto world_idx = 0; world_idx < 30; ++world_idx ) {
        auto heights = std::vector<float>( size * size * layers, 0.0f );
        for( float &h : heights ) {
            const auto k = kind( rng );
            h = k < 5 ? 0.0f : ( k < 7 ? 0.1f : ( k < 9 ? 0.5f : 1.0f ) );
        }
        const auto fn = [ &heights ]( int x, int y, int z ) {
            if( x < 0 || y < 0 || z < 0 || x >= size || y >= size || z >= layers ) {
                return 0.0f;
            }
            return heights[( z * size + y ) * size + x];
        };
        for( auto ray_idx = 0; ray_idx < 200; ++ray_idx ) {
            const auto o = r3d::vec3{ unit( rng ) *size, unit( rng ) *size, unit( rng ) *layers + 0.5f };
            const auto d = r3d::normalize( { unit( rng ) - 0.5f, unit( rng ) - 0.5f, unit( rng ) - 0.7f } );
            if( r3d::length( d ) < 0.5f ) {
                continue;
            }
            const auto fast = r3d::raycast_columns( { .origin = o, .dir = d, .max_t = 60.0f, .height = fn } );
            const auto slow = r3d::raycast_columns_reference( { .origin = o, .dir = d, .max_t = 60.0f, .height = fn,
                              .lo_x = 0, .lo_y = 0, .lo_z = 0, .hi_x = size - 1, .hi_y = size - 1, .hi_z = layers - 1 } );
            INFO( "world " << world_idx << " ray " << ray_idx );
            REQUIRE( fast.hit == slow.hit );
            if( fast.hit ) {
                CHECK( fast.x == slow.x );
                CHECK( fast.y == slow.y );
                CHECK( fast.z == slow.z );
                CHECK( approx( fast.t, slow.t, 1e-3f ) );
            }
        }
    }
}

TEST_CASE( "r3d_terrain_shapes", "[r3d]" )
{
    auto t = r3d::tile_traits{};
    CHECK( r3d::classify_terrain( t ).kind == r3d::shape_kind::floor );

    t = {};
    t.no_floor = true;
    CHECK( r3d::classify_terrain( t ).kind == r3d::shape_kind::empty );

    t = {};
    t.wall = true;
    CHECK( r3d::classify_terrain( t ).kind == r3d::shape_kind::block );

    t = {};
    t.move_cost = 0; // closed door or window: impassable and not flat
    CHECK( r3d::classify_terrain( t ).kind == r3d::shape_kind::block );

    t = {};
    t.move_cost = 0;
    t.flat = true;
    CHECK( r3d::classify_terrain( t ).kind == r3d::shape_kind::floor );

    t = {};
    t.tree = true;
    CHECK( r3d::classify_terrain( t ).kind == r3d::shape_kind::upright );

    t = {};
    t.shrub = true;
    const auto shrub = r3d::classify_terrain( t );
    CHECK( shrub.kind == r3d::shape_kind::upright );
    CHECK( shrub.height < 1.0f );

    // Windows, broken or not, and closed doors stand in the wall line as panes.
    t = {};
    t.window = true;
    CHECK( r3d::classify_terrain( t ).kind == r3d::shape_kind::pane );
    t.move_cost = 0;
    CHECK( r3d::classify_terrain( t ).kind == r3d::shape_kind::pane );
    t = {};
    t.door = true;
    t.move_cost = 0;
    CHECK( r3d::classify_terrain( t ).kind == r3d::shape_kind::pane );
    // An open door lies flat; a wall stays a wall.
    t.flat = true;
    t.move_cost = 2;
    CHECK( r3d::classify_terrain( t ).kind == r3d::shape_kind::floor );
    t = {};
    t.door = true;
    t.wall = true;
    t.move_cost = 0;
    CHECK( r3d::classify_terrain( t ).kind == r3d::shape_kind::block );
}

TEST_CASE( "r3d_cabinets_meet_adjacent_thin_wall_faces", "[r3d]" )
{
    const auto box = r3d::furniture_box_layout( { .ox0 = 0.1f, .oy0 = 0.06f, .ox1 = 0.9f, .oy1 = 1.0f,
                     .front = 0.78f, .low = false } );
    const auto backs = std::array<uint8_t, 4> {{
            r3d::face_north, r3d::face_east, r3d::face_south, r3d::face_west
        }
    };
    for( bool flip : {
             false, true
         } ) {
        for( auto facing = 0; facing < 4; ++facing ) {
            const auto original = r3d::place_furniture_box( { .box = box, .flip = flip, .facing = facing, .joined = 0, .wall_contacts = 0 } );
            const auto fitted = r3d::place_furniture_box( { .box = box, .flip = flip, .facing = facing, .joined = 0, .wall_contacts = backs[facing] } );
            // Adjacent wall centres are -0.5 or 1.5 in this tile's coordinates. The
            // cabinet meets the near face without moving its front or either end.
            CHECK( approx( fitted.x0, facing == 3 ? -0.5f + r3d::wall_half_thickness : original.x0 ) );
            CHECK( approx( fitted.x1, facing == 1 ? 1.5f - r3d::wall_half_thickness : original.x1 ) );
            CHECK( approx( fitted.y0, facing == 0 ? -0.5f + r3d::wall_half_thickness : original.y0 ) );
            CHECK( approx( fitted.y1, facing == 2 ? 1.5f - r3d::wall_half_thickness : original.y1 ) );
            CHECK( fitted.side_top == box.side_top );
            CHECK( fitted.lid_bottom == box.lid_bottom );
            const auto shallow = r3d::place_furniture_box( { .box = box, .flip = flip, .facing = facing, .joined = 0,
                                 .wall_contacts = backs[facing], .depth_scale = 0.5f } );
            if( ( facing & 1 ) == 0 ) {
                CHECK( approx( shallow.y1 - shallow.y0, ( fitted.y1 - fitted.y0 ) * 0.5f ) );
                CHECK( shallow.x0 == fitted.x0 );
                CHECK( shallow.x1 == fitted.x1 );
                CHECK( ( facing == 0 ? shallow.y0 == fitted.y0 : shallow.y1 == fitted.y1 ) );
            } else {
                CHECK( approx( shallow.x1 - shallow.x0, ( fitted.x1 - fitted.x0 ) * 0.5f ) );
                CHECK( shallow.y0 == fitted.y0 );
                CHECK( shallow.y1 == fitted.y1 );
                CHECK( ( facing == 1 ? shallow.x1 == fitted.x1 : shallow.x0 == fitted.x0 ) );
            }
        }
    }
    const auto row = r3d::place_furniture_box( { .box = box, .flip = false, .facing = 0,
                     .joined = r3d::face_west | r3d::face_east, .wall_contacts = r3d::face_north } );
    CHECK( row.x0 == 0.0f );
    CHECK( row.x1 == 1.0f );
    CHECK( approx( row.y0, -0.5f + r3d::wall_half_thickness ) );
}

TEST_CASE( "r3d_furniture_box_layout", "[r3d]" )
{
    // A sprite filling its whole cell: a whole-tile box, lid above the front.
    const auto full = r3d::furniture_box_layout( { .ox0 = 0.0f, .oy0 = 0.0f, .ox1 = 1.0f, .oy1 = 1.0f, .front = 0.4f, .low = false } );
    CHECK( approx( full.x0, 0.0f ) );
    CHECK( approx( full.x1, 1.0f ) );
    CHECK( approx( full.y0, 0.0f ) );
    CHECK( approx( full.y1, 1.0f ) );
    CHECK( approx( full.lid_bottom, 0.6f ) );
    CHECK( approx( full.side_top, 0.6f ) );
    CHECK( approx( full.side_bottom, 1.0f ) );

    // A narrow bench in the middle of its cell: the box is only as wide as the planks, and its
    // sides take the lid's front edge, so they line up with the top.
    const auto bench = r3d::furniture_box_layout( { .ox0 = 0.3f, .oy0 = 0.2f, .ox1 = 0.7f, .oy1 = 0.9f, .front = 0.4f, .low = true } );
    CHECK( approx( bench.x0, 0.3f ) );
    CHECK( approx( bench.x1, 0.7f ) );
    CHECK( approx( bench.col0, bench.x0 ) );
    CHECK( approx( bench.col1, bench.x1 ) );
    // Boxes stand against the back (north) of their tile, where a wall would be.
    CHECK( approx( bench.y0, 0.0f ) );
    CHECK( bench.y1 >= 0.5f );
    CHECK( bench.y1 <= 1.0f );
    // A tall piece whose art starts a little below the top still touches the back.
    const auto locker = r3d::furniture_box_layout( { .ox0 = 0.0f, .oy0 = 0.06f, .ox1 = 1.0f, .oy1 = 1.0f, .front = 0.78f, .low = false } );
    CHECK( approx( locker.y0, 0.0f ) );
    CHECK( approx( bench.side_bottom, bench.lid_bottom ) );
    CHECK( bench.side_top < bench.side_bottom );

    // No opaque pixels known: fall back to the whole sprite.
    const auto none = r3d::furniture_box_layout( { .ox0 = 0.0f, .oy0 = 0.0f, .ox1 = 0.0f, .oy1 = 0.0f, .front = 0.4f, .low = false } );
    CHECK( approx( none.x1, 1.0f ) );
    CHECK( approx( none.y1, 1.0f ) );
}

TEST_CASE( "r3d_shrub_ground_is_separate_from_upright_foliage", "[r3d]" )
{
    auto traits = r3d::tile_traits{};
    traits.shrub = true;
    const auto bush = r3d::classify_terrain( traits );
    CHECK( bush.kind == r3d::shape_kind::upright );
    CHECK( bush.ground_backdrop );
    CHECK( bush.height > r3d::floor_height );

    traits = {};
    traits.tree = true;
    CHECK_FALSE( r3d::classify_terrain( traits ).ground_backdrop );
    CHECK_FALSE( r3d::classify_furniture( {} ).ground_backdrop );
}

TEST_CASE( "r3d_natural_furniture_is_a_solid_volume", "[r3d]" )
{
    // Japanese knotweed is passable, SHORT and SHRUB. It must not become a bench.
    auto knotweed = r3d::tile_traits{};
    knotweed.shrub = true;
    knotweed.low = true;
    const auto plant = r3d::classify_furniture( knotweed );
    CHECK( plant.kind == r3d::shape_kind::block );
    CHECK( plant.slab == 0.0f );
    CHECK( plant.height > r3d::floor_height );

    // Small, medium and large boulders respectively use TINY, SHORT and impassable.
    auto previous_height = r3d::floor_height;
    for( auto size = 0; size < 3; ++size ) {
        auto rock = r3d::tile_traits{};
        rock.boulder = true;
        rock.tiny = size == 0;
        rock.low = size == 1;
        rock.move_cost = size == 2 ? 0 : 100;
        const auto volume = r3d::classify_furniture( rock );
        CHECK( volume.kind == r3d::shape_kind::block );
        CHECK( volume.slab == 0.0f );
        CHECK( volume.height > previous_height );
        CHECK( volume.height <= 1.0f );
        previous_height = volume.height;
    }
}

TEST_CASE( "r3d_furniture_shapes", "[r3d]" )
{
    // Small walkable furniture (chairs) stands up as a sprite.
    auto t = r3d::tile_traits{};
    const auto ordinary = r3d::classify_furniture( t );
    CHECK( ordinary.kind == r3d::shape_kind::upright );
    CHECK( ordinary.height > r3d::floor_height );

    t = {};
    t.tiny = true;
    const auto tiny = r3d::classify_furniture( t );
    CHECK( tiny.kind == r3d::shape_kind::floor );
    CHECK( tiny.height < ordinary.height );

    // Impassable furniture (lockers, fridges) is a tall box.
    t = {};
    t.move_cost = 0;
    const auto locker = r3d::classify_furniture( t );
    CHECK( locker.kind == r3d::shape_kind::prism );
    CHECK( locker.height > ordinary.height );

    // Benches and counters are low boxes, lower than lockers.
    t = {};
    t.low = true;
    const auto bench = r3d::classify_furniture( t );
    CHECK( bench.kind == r3d::shape_kind::prism );
    CHECK( bench.height < locker.height );
    // Benches and tables are a thin top on legs, cabinets are solid.
    CHECK( bench.slab > 0.0f );
    CHECK( bench.slab < bench.height );
    CHECK( locker.slab == 0.0f );
    t = {};
    t.flat_surface = true;
    CHECK( r3d::classify_furniture( t ).kind == r3d::shape_kind::prism );
    t.container = true;
    CHECK( r3d::classify_furniture( t ).slab == 0.0f );

    t = {};
    t.flat = true;
    const auto rug = r3d::classify_furniture( t );
    CHECK( rug.kind == r3d::shape_kind::floor );
    CHECK( rug.height < ordinary.height );
}

TEST_CASE( "r3d_face_visibility", "[r3d]" )
{
    const auto block = r3d::shape{ r3d::shape_kind::block, 1.0f };
    const auto floor = r3d::shape{ r3d::shape_kind::floor, r3d::floor_height };
    const auto air = r3d::shape{};

    // A lone block shows everything except its bottom.
    auto open = r3d::neighbourhood{};
    CHECK( r3d::visible_faces( block, open ) ==
           ( r3d::face_top | r3d::face_north | r3d::face_south | r3d::face_west | r3d::face_east ) );

    // Block with another block to the east and above: the east side is hidden, but the top
    // stays, since the level above may be cut away.
    auto packed = r3d::neighbourhood{};
    packed.east = block;
    packed.up = block;
    const auto faces = r3d::visible_faces( block, packed );
    CHECK( ( faces & r3d::face_east ) == 0 );
    CHECK( ( faces & r3d::face_top ) != 0 );
    CHECK( ( faces & r3d::face_west ) != 0 );

    // A block beside a floor still shows that side, because the floor is lower.
    auto beside_floor = r3d::neighbourhood{};
    beside_floor.south = floor;
    CHECK( ( r3d::visible_faces( block, beside_floor ) & r3d::face_south ) != 0 );

    // A floor only ever has a top face, even under a block.
    CHECK( r3d::visible_faces( floor, open ) == r3d::face_top );
    auto roofed = r3d::neighbourhood{};
    roofed.up = block;
    CHECK( r3d::visible_faces( floor, roofed ) == r3d::face_top );

    // Air, upright sprites and panes emit nothing through this path.
    CHECK( r3d::visible_faces( air, open ) == 0 );
    CHECK( r3d::visible_faces( { r3d::shape_kind::upright, 1.4f }, open ) == 0 );
    CHECK( r3d::visible_faces( { r3d::shape_kind::pane, 1.0f }, open ) == 0 );

    // A tall prism next to a short one shows its side; the short one's side is hidden.
    const auto tall = r3d::shape{ r3d::shape_kind::prism, 0.9f };
    const auto low = r3d::shape{ r3d::shape_kind::prism, 0.3f };
    auto tall_next_to_low = r3d::neighbourhood{};
    tall_next_to_low.east = low;
    CHECK( ( r3d::visible_faces( tall, tall_next_to_low ) & r3d::face_east ) != 0 );
    auto low_next_to_tall = r3d::neighbourhood{};
    low_next_to_tall.east = tall;
    CHECK( ( r3d::visible_faces( low, low_next_to_tall ) & r3d::face_east ) == 0 );
}

TEST_CASE( "r3d_wall_texture_matching", "[r3d]" )
{
    auto rules = r3d::wall_texture_rules{};
    rules.by_id = { { "t_wall", "plaster" }, { "t_wall_log_half", "logs" } };
    rules.by_prefix = { { "t_wall_", "plaster" }, { "t_wall_log", "logs" }, { "t_brick_wall", "brick" } };

    // Exact ids first, along the looks_like chain.
    CHECK( r3d::match_wall_texture( rules, { "t_wall" } ) == "plaster" );
    CHECK( r3d::match_wall_texture( rules, { "t_wall_log_half" } ) == "logs" );
    CHECK( r3d::match_wall_texture( rules, { "t_mod_wall", "t_wall" } ) == "plaster" );
    // Then the longest prefix.
    CHECK( r3d::match_wall_texture( rules, { "t_wall_log_widened" } ) == "logs" );
    CHECK( r3d::match_wall_texture( rules, { "t_wall_b" } ) == "plaster" );
    CHECK( r3d::match_wall_texture( rules, { "t_brick_wall_halfway" } ) == "brick" );
    // Nothing matches: the fallback, empty by default (keep the tileset sprite).
    CHECK( r3d::match_wall_texture( rules, { "t_vat" } ).empty() );
    rules.fallback = "concrete";
    CHECK( r3d::match_wall_texture( rules, { "t_vat" } ) == "concrete" );
}

TEST_CASE( "r3d_wall_atlas_cells", "[r3d]" )
{
    // 4 x 3 cells of 32 pixels.
    const auto first = r3d::atlas_cell( { .index = 0, .cell = 32, .atlas_w = 128, .atlas_h = 96 } );
    REQUIRE( first.valid );
    CHECK( approx( first.u0, 0.5f / 128.0f ) );
    CHECK( approx( first.v1, 31.5f / 96.0f ) );
    const auto fifth = r3d::atlas_cell( { .index = 5, .cell = 32, .atlas_w = 128, .atlas_h = 96 } );
    REQUIRE( fifth.valid );
    CHECK( approx( fifth.u0, 32.5f / 128.0f ) );
    CHECK( approx( fifth.v0, 32.5f / 96.0f ) );
    CHECK_FALSE( r3d::atlas_cell( { .index = 12, .cell = 32, .atlas_w = 128, .atlas_h = 96 } ).valid );
    CHECK_FALSE( r3d::atlas_cell( { .index = -1, .cell = 32, .atlas_w = 128, .atlas_h = 96 } ).valid );
}

TEST_CASE( "r3d_sky_direction_follows_calendar_convention", "[r3d]" )
{
    // Azimuth 0 is south: the sun stands to the south (world -y), at its altitude.
    const auto noon = r3d::sky_direction( 0.0f, 0.5f );
    CHECK( approx( noon.x, 0.0f ) );
    CHECK( noon.y < 0.0f );
    CHECK( approx( noon.z, std::sin( 0.5f ) ) );
    // Positive azimuth turns towards the west (world -x).
    const auto west = r3d::sky_direction( 1.5707963f, 0.0f );
    CHECK( approx( west.x, -1.0f ) );
    CHECK( approx( west.y, 0.0f ) );
    CHECK( approx( r3d::length( r3d::sky_direction( 0.3f, 0.7f ) ), 1.0f ) );
}

TEST_CASE( "r3d_sky_color_brightens_by_day_and_darkens_with_cloud", "[r3d]" )
{
    const auto night = r3d::sky_color( 0.0f, 0.0f, false );
    const auto day = r3d::sky_color( 1.0f, 0.0f, false );
    const auto overcast = r3d::sky_color( 1.0f, 1.0f, false );
    CHECK( day.b > night.b );
    CHECK( day.r + day.g + day.b > overcast.r + overcast.g + overcast.b );
    // Overcast is grey: its channels are equal.
    CHECK( approx( overcast.r, overcast.b ) );
    CHECK( r3d::sky_color( 0.0f, 0.0f, true ).r > day.r );
}

TEST_CASE( "r3d_wind_blows_away_from_its_direction", "[r3d]" )
{
    // A north wind (from 0 degrees) blows south: world -y.
    const auto north = r3d::wind_velocity( 0, 5.0f );
    CHECK( approx( north.x, 0.0f ) );
    CHECK( approx( north.y, -5.0f ) );
    // An east wind blows west.
    const auto east = r3d::wind_velocity( 90, 2.0f );
    CHECK( approx( east.x, -2.0f ) );
    CHECK( approx( east.y, 0.0f, 1e-3f ) );
}

TEST_CASE( "r3d_particles_drift_with_the_wind_and_stay_in_their_box", "[r3d]" )
{
    auto f = r3d::particle_field{};
    f.half_extent = 10.0f;
    f.height = 4.0f;
    f.fall_speed = 3.0f;
    f.wind = r3d::wind_velocity( 270, 2.0f ); // from the west: drifts east
    auto sum_dx = 0.0;
    auto sum_dz = 0.0;
    for( auto i = uint32_t{ 0 }; i < 200; ++i ) {
        for( float t : {
                 0.0f, 0.37f, 5.0f, 123.4f
             } ) {
            const auto p = r3d::particle_position( { .index = i, .seconds = t, .field = f, .world_offset = { 3.5f, -2.0f, 0.0f } } );
            CHECK( p.x >= -10.0f );
            CHECK( p.x < 10.0f + 1e-3f );
            CHECK( p.y >= -10.0f );
            CHECK( p.y < 10.0f + 1e-3f );
            CHECK( p.z >= 0.0f );
            CHECK( p.z < 4.0f + 1e-3f );
        }
        // Over a short step (no wrapping for most), particles move east and down.
        const auto a = r3d::particle_position( { .index = i, .seconds = 1.0f, .field = f } );
        const auto b = r3d::particle_position( { .index = i, .seconds = 1.05f, .field = f } );
        if( std::fabs( b.x - a.x ) < 1.0f && std::fabs( b.z - a.z ) < 1.0f ) {
            sum_dx += b.x - a.x;
            sum_dz += b.z - a.z;
        }
    }
    CHECK( sum_dx > 0.0 );
    CHECK( sum_dz < 0.0 );
    // The same particle at the same time is in the same place: the picture does not flicker.
    CHECK( approx( r3d::particle_position( { .index = 7, .seconds = 2.0f, .field = f } ).x,
                   r3d::particle_position( { .index = 7, .seconds = 2.0f, .field = f } ).x ) );
}

TEST_CASE( "r3d_fog_grows_with_sight_penalty_and_distance", "[r3d]" )
{
    CHECK( r3d::fog_density( 1.0f ) == 0.0f );
    CHECK( r3d::fog_density( 1.7f ) > r3d::fog_density( 1.1f ) );
    const auto d = r3d::fog_density( 1.7f );
    CHECK( r3d::fog_amount( 0.0f, d ) == 0.0f );
    CHECK( r3d::fog_amount( 10.0f, d ) < r3d::fog_amount( 40.0f, d ) );
    CHECK( r3d::fog_amount( 200.0f, d ) > 0.99f );
}

TEST_CASE( "r3d_bearings_match_the_camera", "[r3d]" )
{
    CHECK( approx( r3d::map_bearing( 0.0f, -1.0f ), 0.0f ) );          // north
    CHECK( approx( r3d::map_bearing( 1.0f, 0.0f ), 1.5707963f ) );     // east
    CHECK( approx( r3d::map_bearing( 0.0f, 1.0f ), 3.1415927f ) );     // south
    CHECK( approx( r3d::map_bearing( -1.0f, 0.0f ), 4.712389f ) );     // west
    for( auto step = -12; step <= 12; ++step ) {
        auto cam = r3d::camera{};
        cam.yaw = 0.2617994f * static_cast<float>( step );
        const auto fwd = r3d::camera_forward_bearing( cam.yaw );
        const auto eighths = static_cast<float>( cam.forward_eighths() ) * 0.7853982f;
        CHECK( std::fabs( r3d::bearing_delta( eighths, fwd ) ) <= 0.3927f + 1e-3f );
    }
    CHECK( approx( r3d::bearing_delta( 6.1f, 0.1f ), 0.1f + 6.2831853f - 6.1f ) );
    CHECK( approx( r3d::bearing_delta( 0.1f, 6.1f ), -( 0.1f + 6.2831853f - 6.1f ) ) );
}

TEST_CASE( "r3d_edge_marks_sit_on_the_border_and_point_out", "[r3d]" )
{
    // On screen: no mark.
    CHECK( r3d::place_edge_mark( { .clip_x = 0.2f, .clip_y = -0.3f, .clip_w = 1.0f, .margin_x = 0.1f, .margin_y = 0.1f } ).on_screen );
    // Far to the right: on the right border, pointing right.
    const auto right = r3d::place_edge_mark( { .clip_x = 5.0f, .clip_y = 0.5f, .clip_w = 1.0f, .margin_x = 0.1f, .margin_y = 0.1f } );
    CHECK_FALSE( right.on_screen );
    CHECK( approx( right.x, 0.9f ) );
    CHECK( approx( right.y, 0.09f ) );
    CHECK( std::fabs( right.angle ) < 0.2f );
    // Up and left: on the top border, pointing up-left.
    const auto up = r3d::place_edge_mark( { .clip_x = -1.0f, .clip_y = 4.0f, .clip_w = 1.0f, .margin_x = 0.1f, .margin_y = 0.1f } );
    CHECK( approx( up.y, 0.9f ) );
    CHECK( up.x < 0.0f );
    CHECK( up.angle > 1.5707963f );
    // Behind the camera, the projection is mirrored back.
    const auto behind = r3d::place_edge_mark( { .clip_x = 0.5f, .clip_y = 0.0f, .clip_w = -1.0f, .margin_x = 0.1f, .margin_y = 0.1f } );
    CHECK_FALSE( behind.on_screen );
    CHECK( behind.x < 0.0f );
    const auto straight_back = r3d::place_edge_mark( { .clip_x = 0.0f, .clip_y = 0.0f, .clip_w = -1.0f, .margin_x = 0.1f, .margin_y = 0.1f } );
    CHECK( approx( straight_back.y, -0.9f ) );
}

