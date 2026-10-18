#if defined(TILES)

#include "r3d_decals.h"

namespace r3d
{

auto add_ground_decal( const ground_decal_options &options ) -> void
{
    auto lift = 0.0f;
    for( const auto *sprite : { &options.tile.backdrop, &options.tile.sprite } ) {
        if( !sprite->valid ) {
            continue;
        }
        const auto &s = *sprite;
        const auto x = options.x + s.offset_x;
        const auto y = options.y + s.offset_y;
        const auto z = options.z + lift;
        auto corners = corners_t{{ { x, y, z }, { x + s.size_x, y, z },
                { x + s.size_x, y + s.size_y, z }, { x, y + s.size_y, z }
            }};
        for( auto &corner : corners ) {
            for( auto turn = 0; turn < ( s.quarter_turns & 3 ); ++turn ) {
                const auto dx = corner[0] - options.x - 0.5f;
                const auto dy = corner[1] - options.y - 0.5f;
                corner[0] = options.x + 0.5f - dy;
                corner[1] = options.y + 0.5f + dx;
            }
        }
        const auto left = s.flip ? s.u1 : s.u0;
        const auto right = s.flip ? s.u0 : s.u1;
        const auto uvs = uvs_t{{ { left, s.v0 }, { right, s.v0 },
                { right, s.v1 }, { left, s.v1 }
            }};
        options.sink.add_colored( { .sprite = s, .corners = corners, .uvs = uvs,
                                    .ref = {}, .color = options.color & 0x00ffffffu } );
        lift += 0.004f;
    }
}

} // namespace r3d

#endif // defined(TILES)
