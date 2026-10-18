#include "catch/catch.hpp"
#include "r3d_capture_messages.h"

TEST_CASE( "renderer_capture_preserves_nested_message_read_state", "[r3d]" )
{
    REQUIRE_FALSE( Messages::capture_read_frozen );
    {
        const auto outer = Messages::capture_read_guard{};
        REQUIRE( Messages::capture_read_frozen );
        {
            const auto inner = Messages::capture_read_guard{};
            REQUIRE( Messages::capture_read_frozen );
        }
        REQUIRE( Messages::capture_read_frozen );
    }
    REQUIRE_FALSE( Messages::capture_read_frozen );
}
