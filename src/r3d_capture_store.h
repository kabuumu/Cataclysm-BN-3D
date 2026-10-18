#pragma once

#include <expected>
#include <filesystem>
#include <string>

namespace r3d
{
/// Snapshot a committed SQLite save, including pages still in its live WAL.
auto snapshot_database( const std::filesystem::path &source,
                        const std::filesystem::path &destination ) -> std::expected<void, std::string>;
}
