#include "r3d_wall_rules.h"

namespace r3d
{

auto match_wall_texture( const wall_texture_rules &rules,
                         const std::vector<std::string> &chain ) -> std::string
{
    for( const std::string &id : chain ) {
        const auto it = rules.by_id.find( id );
        if( it != rules.by_id.end() ) {
            return it->second;
        }
    }
    for( const std::string &id : chain ) {
        const std::string *best = nullptr;
        auto best_len = size_t( 0 );
        for( const auto &rule : rules.by_prefix ) {
            const auto &prefix = rule.first;
            if( !prefix.empty() && prefix.size() > best_len &&
                id.compare( 0, prefix.size(), prefix ) == 0 ) {
                best = &rule.second;
                best_len = prefix.size();
            }
        }
        if( best ) {
            return *best;
        }
    }
    return rules.fallback;
}

auto atlas_cell( const atlas_cell_options &options ) -> uv_rect
{
    auto index = options.index;
    auto cell = options.cell;
    auto atlas_w = options.atlas_w;
    auto atlas_h = options.atlas_h;

    auto r = uv_rect{};
    if( index < 0 || cell <= 0 || atlas_w < cell || atlas_h < cell ) {
        return r;
    }
    const auto columns = static_cast<int>( atlas_w / cell );
    const auto col = static_cast<int>( index % columns );
    const auto row = static_cast<int>( index / columns );
    if( ( row + 1 ) * cell > atlas_h ) {
        return r;
    }
    const auto w = static_cast<float>( atlas_w );
    const auto h = static_cast<float>( atlas_h );
    r.u0 = ( static_cast<float>( col * cell ) + 0.5f ) / w;
    r.v0 = ( static_cast<float>( row * cell ) + 0.5f ) / h;
    r.u1 = ( static_cast<float>( ( col + 1 ) * cell ) - 0.5f ) / w;
    r.v1 = ( static_cast<float>( ( row + 1 ) * cell ) - 0.5f ) / h;
    r.valid = true;
    return r;
}

} // namespace r3d
