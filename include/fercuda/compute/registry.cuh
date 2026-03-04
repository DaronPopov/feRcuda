#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>

#include "fercuda/core/runtime.cuh"
#include "fercuda/core/status.cuh"

namespace fer {

enum class OpTag : uint16_t {
    MATMUL = 1,
    LAYER_NORM = 2,
};

struct OpTagHash {
    std::size_t operator()(OpTag t) const {
        return static_cast<std::size_t>(t);
    }
};

class KernelRegistry {
public:
    template<typename Spec, typename Impl>
    void register_impl(OpTag tag, const char* name) {
        Entry e;
        e.spec_size = sizeof(Spec);
        e.name = name;
        e.validate = &validate_adapter<Spec, Impl>;
        e.launch = &launch_adapter<Spec, Impl>;
        entries_[tag] = e;
    }

    template<typename Spec>
    Status validate(OpTag tag, const Spec& spec) const {
        const Entry* e = find(tag);
        if (!e) return Status::not_found("op tag not registered");
        if (e->spec_size != sizeof(Spec)) {
            return Status::invalid_argument("spec type mismatch for op tag");
        }
        return e->validate(static_cast<const void*>(&spec));
    }

    template<typename Spec>
    Status launch(OpTag tag, const Spec& spec, const OpContext& ctx) const {
        const Entry* e = find(tag);
        if (!e) return Status::not_found("op tag not registered");
        if (e->spec_size != sizeof(Spec)) {
            return Status::invalid_argument("spec type mismatch for op tag");
        }
        return e->launch(static_cast<const void*>(&spec), ctx);
    }

private:
    struct Entry {
        std::size_t spec_size = 0;
        const char* name = "";
        Status (*validate)(const void*) = nullptr;
        Status (*launch)(const void*, const OpContext&) = nullptr;
    };

    const Entry* find(OpTag tag) const {
        auto it = entries_.find(tag);
        if (it == entries_.end()) return nullptr;
        return &it->second;
    }

    template<typename Spec, typename Impl>
    static Status validate_adapter(const void* spec) {
        return Impl::validate(*static_cast<const Spec*>(spec));
    }

    template<typename Spec, typename Impl>
    static Status launch_adapter(const void* spec, const OpContext& ctx) {
        return Impl::launch(*static_cast<const Spec*>(spec), ctx);
    }

    std::unordered_map<OpTag, Entry, OpTagHash> entries_;
};

} // namespace fer
