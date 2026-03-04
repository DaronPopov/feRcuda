#pragma once

namespace fer {

enum class StatusCode : int {
    OK = 0,
    INVALID_ARGUMENT = 1,
    NOT_FOUND = 2,
    INTERNAL_ERROR = 3,
};

struct Status {
    StatusCode code;
    const char* message;

    constexpr bool ok() const { return code == StatusCode::OK; }

    static constexpr Status ok_status() {
        return {StatusCode::OK, "ok"};
    }

    static constexpr Status invalid_argument(const char* msg) {
        return {StatusCode::INVALID_ARGUMENT, msg};
    }

    static constexpr Status not_found(const char* msg) {
        return {StatusCode::NOT_FOUND, msg};
    }

    static constexpr Status internal_error(const char* msg) {
        return {StatusCode::INTERNAL_ERROR, msg};
    }
};

} // namespace fer
