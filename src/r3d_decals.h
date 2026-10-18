#pragma once
#ifndef CATA_SRC_R3D_DECALS_H
#define CATA_SRC_R3D_DECALS_H

#if defined(TILES)

#include "r3d_mesh.h"

namespace r3d
{

struct ground_decal_options {
    quad_sink &sink;
    const tile_info &tile;
    float x;
    float y;
    float z;
    uint32_t color;
};

/// Ground sprites preserve tile offsets, rotation and mirroring, with separate layer depths.
auto add_ground_decal( const ground_decal_options &options ) -> void;

} // namespace r3d

#endif // defined(TILES)
#endif // CATA_SRC_R3D_DECALS_H
