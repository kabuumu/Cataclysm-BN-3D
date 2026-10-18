#include "catch/catch.hpp"
#include "r3d_capture_store.h"

#include <filesystem>
#include <memory>
#include <sqlite3.h>

TEST_CASE( "renderer_snapshot_includes_committed_wal", "[r3d][save]" )
{
    const auto directory = std::filesystem::temp_directory_path() / "bn-r3d-snapshot-test";
    REQUIRE( std::filesystem::create_directory( directory ) );
    const auto cleanup = std::unique_ptr < const std::filesystem::path,
    decltype( []( const auto * path ) {
        std::filesystem::remove_all( *path );
    } ) > ( &directory );
    const auto source = directory / "world.sqlite3";
    const auto target = directory / "snapshot.sqlite3";
    auto raw = static_cast<sqlite3 *>( nullptr );
    REQUIRE( sqlite3_open( source.string().c_str(), &raw ) == SQLITE_OK );
    const auto database = std::unique_ptr<sqlite3, decltype( &sqlite3_close )>( raw, sqlite3_close );
    REQUIRE( sqlite3_exec( database.get(), "PRAGMA journal_mode=WAL; PRAGMA wal_autocheckpoint=0;"
                           "CREATE TABLE state(turn INTEGER); INSERT INTO state VALUES(12345);",
                           nullptr, nullptr, nullptr ) == SQLITE_OK );
    REQUIRE( std::filesystem::exists( source.string() + "-wal" ) );
    REQUIRE( r3d::snapshot_database( source, target ).has_value() );
    auto saved_raw = static_cast<sqlite3 *>( nullptr );
    REQUIRE( sqlite3_open( target.string().c_str(), &saved_raw ) == SQLITE_OK );
    const auto saved = std::unique_ptr<sqlite3, decltype( &sqlite3_close )>( saved_raw, sqlite3_close );
    auto statement_raw = static_cast<sqlite3_stmt *>( nullptr );
    REQUIRE( sqlite3_prepare_v2( saved.get(), "SELECT turn FROM state", -1, &statement_raw,
                                 nullptr ) == SQLITE_OK );
    const auto statement = std::unique_ptr<sqlite3_stmt, decltype( &sqlite3_finalize )>( statement_raw,
                           sqlite3_finalize );
    REQUIRE( sqlite3_step( statement.get() ) == SQLITE_ROW );
    CHECK( sqlite3_column_int( statement.get(), 0 ) == 12345 );
    CHECK_FALSE( r3d::snapshot_database( directory / "missing.sqlite3",
                                         directory / "bad.sqlite3" ).has_value() );
}
