#include "avatar.h"
#include "calendar.h"
#include "catch/catch.hpp"
#include "game.h"
#include "map/field.h"
#include "map/map.h"
#include "map/mapdata.h"
#include "map_helpers.h"
#include "player_helpers.h"
#include "r3d_scene_rules.h"

#if defined(TILES)
#    include "r3d_decals.h"
#endif

TEST_CASE("r3d_overhead_cutaway_follows_visible_interiors", "[r3d][vision]") {
    clear_avatar();
    clear_map();
    auto& here = get_map();
    auto& you = get_avatar();
    const auto outside = tripoint_bub_ms(60, 60, 0);
    // BN counts a tile next to an uncovered square as outside: keep a full
    // 3 x 3 roof neighbourhood above the interior sample.
    const auto inside = outside + tripoint_rel_ms(2, 0, 0);
    const auto hidden = outside + tripoint_rel_ms(4, 0, 0);
    you.setpos(outside);
    for (auto y = 59; y <= 61; ++y) {
        for (auto x = 61; x <= 65; ++x) {
            here.ter_set(tripoint_bub_ms(x, y, 0), x == 63 ? t_wall : t_floor);
            here.ter_set(tripoint_bub_ms(x, y, 1), ter_str_id("t_flat_roof").id());
        }
    }
    set_time(calendar::turn_zero + 12_hours);
    REQUIRE(here.is_outside(outside));
    REQUIRE_FALSE(here.is_outside(inside));
    REQUIRE(you.sees(inside));
    REQUIRE_FALSE(you.sees(hidden));

    CHECK(r3d::hide_overhead_tile(here, inside + tripoint_rel_ms(0, 0, 1), 0));
    CHECK(r3d::hide_overhead_tile(here, inside + tripoint_rel_ms(0, 0, 3), 0));
    CHECK_FALSE(r3d::hide_overhead_tile(here, inside, 0));
    CHECK_FALSE(r3d::hide_overhead_tile(here, outside + tripoint_rel_ms(0, 0, 1), 0));
    CHECK_FALSE(r3d::hide_overhead_tile(here, hidden + tripoint_rel_ms(0, 0, 1), 0));
    // Viewing the roof level itself must retain that floor.
    CHECK_FALSE(r3d::hide_overhead_tile(here, inside + tripoint_rel_ms(0, 0, 1), 1));

    here.ter_set(outside + tripoint_rel_ms(3, 0, 0), ter_str_id("t_door_o").id());
    set_time(calendar::turn);
    REQUIRE(you.sees(hidden));
    CHECK(r3d::hide_overhead_tile(here, hidden + tripoint_rel_ms(0, 0, 1), 0));
}

TEST_CASE("r3d_fields_follow_live_priority_connections_and_visibility", "[r3d][field][vision]") {
    clear_avatar();
    clear_map();
    auto& here = get_map();
    auto& you = get_avatar();
    const auto p = tripoint_bub_ms(61, 60, 0);
    you.setpos(tripoint_bub_ms(60, 60, 0));
    set_time(calendar::turn_zero + 12_hours);
    here.add_field(p, fd_blood, 1);
    here.add_field(p + point_rel_ms::south(), fd_blood, 1);
    here.add_field(p + point_rel_ms::north(), fd_blood, 1);
    here.add_field(p + point_rel_ms::east(), fd_bile, 1);
    const auto blood = r3d::visible_field_at(here, you, p);
    REQUIRE(blood);
    CHECK(blood->id == fd_blood);
    CHECK(blood->display_field);
    CHECK(blood->display_items);
    CHECK(blood->connections == 9);

    here.add_field(p, fd_web, 1);
    const auto web = r3d::visible_field_at(here, you, p);
    REQUIRE(web);
    CHECK(web->id == fd_web);
    CHECK_FALSE(web->display_items);
    CHECK(web->connections == 0);
    here.remove_field(p, fd_web);
    here.remove_field(p, fd_blood);
    const auto cleared = r3d::visible_field_at(here, you, p);
    CHECK_FALSE(cleared);

    const auto hidden = tripoint_bub_ms(64, 60, 0);
    for (auto y = 58; y <= 62; ++y) { here.ter_set(tripoint_bub_ms(63, y, 0), t_wall); }
    here.add_field(hidden, fd_blood, 1);
    set_time(calendar::turn);
    REQUIRE_FALSE(you.sees(hidden));
    CHECK_FALSE(r3d::visible_field_at(here, you, hidden));
}

#if defined(TILES)
TEST_CASE("r3d_ground_decals_preserve_sprite_transforms_and_layers", "[r3d][field]") {
    auto tile = r3d::tile_info{};
    tile.backdrop.valid = true;
    tile.sprite = r3d::sprite_ref{
        .owner = {},
        .u0 = 0.2f,
        .v0 = 0.3f,
        .u1 = 0.6f,
        .v1 = 0.8f,
        .size_x = 2.0f,
        .quarter_turns = 1,
        .offset_x = -0.5f,
        .flip = true,
        .valid = true};
    auto sink = r3d::quad_sink{};
    r3d::add_ground_decal(
        {.sink = sink, .tile = tile, .x = 10.0f, .y = 20.0f, .z = 0.1f, .color = 0xff123456u});
    const auto mesh = sink.finish();
    REQUIRE(mesh.vertices.size() == 8);
    REQUIRE(mesh.light.size() == 8);
    // A quarter turn about the tile centre takes the oversized horizontal sprite
    // from [9.5, 11.5] x [20, 21] to [10, 11] x [19.5, 21.5].
    const auto& first = mesh.vertices[4];
    const auto& opposite = mesh.vertices[6];
    CHECK(first.x == Approx(11.0f));
    CHECK(first.y == Approx(-19.5f));
    CHECK(opposite.x == Approx(10.0f));
    CHECK(opposite.y == Approx(-21.5f));
    CHECK(first.u == Approx(0.6f));
    CHECK(first.v == Approx(0.3f));
    CHECK(opposite.u == Approx(0.2f));
    CHECK(opposite.v == Approx(0.8f));
    CHECK(first.z > mesh.vertices.front().z);
    CHECK(mesh.light.front() == 0x00123456u);
    CHECK(mesh.light.back() == 0x00123456u);

    tile.sprite.valid = false;
    tile.backdrop.valid = false;
    auto empty_sink = r3d::quad_sink{};
    r3d::add_ground_decal(
        {.sink = empty_sink, .tile = tile, .x = 0.0f, .y = 0.0f, .z = 0.0f, .color = 0xffffffu});
    CHECK(empty_sink.finish().empty());
}
#endif
