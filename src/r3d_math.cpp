#include "r3d_math.h"

#include <utility>

namespace r3d
{

auto operator*( const mat4 &a, const mat4 &b ) -> mat4
{
    auto out = mat4{};
    for( auto col = static_cast<int>( 0 ); col < 4; ++col ) {
        for( auto row = static_cast<int>( 0 ); row < 4; ++row ) {
            auto sum = static_cast<float>( 0.0f );
            for( auto k = static_cast<int>( 0 ); k < 4; ++k ) {
                sum += a.at( row, k ) * b.at( k, col );
            }
            out.at( row, col ) = sum;
        }
    }
    return out;
}

auto transform_point( const mat4 &mat, const vec3 &p, float *w_out ) -> vec3
{
    const auto x = static_cast<float>( mat.at( 0, 0 ) * p.x + mat.at( 0, 1 ) * p.y + mat.at( 0,
                                       2 ) * p.z + mat.at( 0, 3 ) );
    const auto y = static_cast<float>( mat.at( 1, 0 ) * p.x + mat.at( 1, 1 ) * p.y + mat.at( 1,
                                       2 ) * p.z + mat.at( 1, 3 ) );
    const auto z = static_cast<float>( mat.at( 2, 0 ) * p.x + mat.at( 2, 1 ) * p.y + mat.at( 2,
                                       2 ) * p.z + mat.at( 2, 3 ) );
    const auto w = static_cast<float>( mat.at( 3, 0 ) * p.x + mat.at( 3, 1 ) * p.y + mat.at( 3,
                                       2 ) * p.z + mat.at( 3, 3 ) );
    if( w_out ) {
        *w_out = w;
    }
    if( w != 0.0f && w != 1.0f ) {
        return { .x = x / w, .y = y / w, .z = z / w };
    }
    return { .x = x, .y = y, .z = z };
}

auto look_at( const vec3 &eye, const vec3 &target, const vec3 &up ) -> mat4
{
    const auto f = vec3( normalize( target - eye ) );
    const auto s = vec3( normalize( cross( f, up ) ) );
    const auto u = vec3( cross( s, f ) );
    auto out = mat4{};
    out.at( 0, 0 ) = s.x;
    out.at( 0, 1 ) = s.y;
    out.at( 0, 2 ) = s.z;
    out.at( 0, 3 ) = -dot( s, eye );
    out.at( 1, 0 ) = u.x;
    out.at( 1, 1 ) = u.y;
    out.at( 1, 2 ) = u.z;
    out.at( 1, 3 ) = -dot( u, eye );
    out.at( 2, 0 ) = -f.x;
    out.at( 2, 1 ) = -f.y;
    out.at( 2, 2 ) = -f.z;
    out.at( 2, 3 ) = dot( f, eye );
    return out;
}

auto perspective( const perspective_options &options ) -> mat4
{
    auto fovy_rad = options.fovy_rad;
    auto aspect = options.aspect;
    auto near_plane = options.near_plane;
    auto far_plane = options.far_plane;

    const auto f = static_cast<float>( 1.0f / std::tan( fovy_rad * 0.5f ) );
    auto out = mat4{};
    out.m.fill( 0.0f );
    out.at( 0, 0 ) = f / aspect;
    out.at( 1, 1 ) = f;
    out.at( 2, 2 ) = far_plane / ( near_plane - far_plane );
    out.at( 2, 3 ) = near_plane * far_plane / ( near_plane - far_plane );
    out.at( 3, 2 ) = -1.0f;
    return out;
}

auto orthographic( const orthographic_options &options ) -> mat4
{
    auto half_height = options.half_height;
    auto aspect = options.aspect;
    auto near_plane = options.near_plane;
    auto far_plane = options.far_plane;

    const auto half_width = static_cast<float>( half_height * aspect );
    auto out = mat4{};
    out.m.fill( 0.0f );
    out.at( 0, 0 ) = 1.0f / half_width;
    out.at( 1, 1 ) = 1.0f / half_height;
    out.at( 2, 2 ) = 1.0f / ( near_plane - far_plane );
    out.at( 2, 3 ) = near_plane / ( near_plane - far_plane );
    out.at( 3, 3 ) = 1.0f;
    return out;
}

auto invert( const mat4 &in, mat4 &out ) -> bool
{
    // Gauss-Jordan elimination with partial pivoting on a double-precision copy.
    double a[4][8];
    for( auto r = static_cast<int>( 0 ); r < 4; ++r ) {
        for( auto c = static_cast<int>( 0 ); c < 4; ++c ) {
            a[r][c] = in.at( r, c );
            a[r][c + 4] = r == c ? 1.0 : 0.0;
        }
    }
    for( auto col = static_cast<int>( 0 ); col < 4; ++col ) {
        auto pivot = static_cast<int>( col );
        for( auto r = static_cast<int>( col + 1 ); r < 4; ++r ) {
            if( std::fabs( a[r][col] ) > std::fabs( a[pivot][col] ) ) {
                pivot = r;
            }
        }
        if( std::fabs( a[pivot][col] ) < 1e-12 ) {
            return false;
        }
        if( pivot != col ) {
            for( auto c = static_cast<int>( 0 ); c < 8; ++c ) {
                std::swap( a[pivot][c], a[col][c] );
            }
        }
        const auto inv = static_cast<double>( 1.0 / a[col][col] );
        for( auto c = static_cast<int>( 0 ); c < 8; ++c ) {
            a[col][c] *= inv;
        }
        for( auto r = static_cast<int>( 0 ); r < 4; ++r ) {
            if( r == col ) {
                continue;
            }
            const auto factor = static_cast<double>( a[r][col] );
            for( auto c = static_cast<int>( 0 ); c < 8; ++c ) {
                a[r][c] -= factor * a[col][c];
            }
        }
    }
    for( auto r = static_cast<int>( 0 ); r < 4; ++r ) {
        for( auto c = static_cast<int>( 0 ); c < 4; ++c ) {
            out.at( r, c ) = static_cast<float>( a[r][c + 4] );
        }
    }
    return true;
}

} // namespace r3d
