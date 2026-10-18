#pragma once
#ifndef CATA_SRC_R3D_CAPTURE_MESSAGES_H
#define CATA_SRC_R3D_CAPTURE_MESSAGES_H

namespace Messages
{
inline auto capture_read_frozen = false;

/// Prevent screenshot redraws from marking sidebar messages as read.
class capture_read_guard
{
        bool previous_;
    public:
        capture_read_guard() : previous_( capture_read_frozen ) { capture_read_frozen = true; }
        ~capture_read_guard() { capture_read_frozen = previous_; }
        capture_read_guard( const capture_read_guard & ) = delete;
        auto operator=( const capture_read_guard & ) -> capture_read_guard & = delete; // *NOPAD*
};
}
#endif
