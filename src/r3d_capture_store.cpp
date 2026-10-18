#include "r3d_capture_store.h"

#include <memory>
#include <sqlite3.h>

namespace r3d
{
auto snapshot_database( const std::filesystem::path &source,
                        const std::filesystem::path &destination ) -> std::expected<void, std::string>
{
    auto raw_source = static_cast<sqlite3 *>( nullptr );
    auto raw_destination = static_cast<sqlite3 *>( nullptr );
    const auto opened = sqlite3_open_v2( source.string().c_str(), &raw_source, SQLITE_OPEN_READONLY,
                                         nullptr );
    const auto close = []( sqlite3 * db ) -> void { sqlite3_close( db ); };
    const auto input = std::unique_ptr<sqlite3, decltype( close )>( raw_source, close );
    if( opened != SQLITE_OK ) { return std::unexpected( "Cannot open saved database" ); }
    const auto created = sqlite3_open_v2( destination.string().c_str(), &raw_destination,
                                          SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nullptr );
    const auto output = std::unique_ptr<sqlite3, decltype( close )>( raw_destination, close );
    if( created != SQLITE_OK ) { return std::unexpected( "Cannot create snapshot database" ); }
    auto backup = sqlite3_backup_init( output.get(), "main", input.get(), "main" );
    if( !backup ) { return std::unexpected( sqlite3_errmsg( output.get() ) ); }
    const auto copied = sqlite3_backup_step( backup, -1 );
    const auto finished = sqlite3_backup_finish( backup );
    if( copied != SQLITE_DONE || finished != SQLITE_OK ) {
        return std::unexpected( "Cannot snapshot committed database pages" );
    }
    if( sqlite3_exec( output.get(), "PRAGMA journal_mode=DELETE", nullptr, nullptr,
                      nullptr ) != SQLITE_OK ) {
        return std::unexpected( "Cannot finalize snapshot journal" );
    }
    return {};
}
}
