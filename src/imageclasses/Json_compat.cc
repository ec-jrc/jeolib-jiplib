#include "Json_compat.h"

namespace json_util {

Json::Value& get_member(Json::Value& node, const std::string& key) {
#if HAS_STRING_VIEW
    // Use this to match modern Conda lib symbols
    return node[std::string_view(key)];
#else
    // Fallback for older systems (C++11/14)
    return node[key];
#endif
}

Json::Value& get_member(Json::Value& node, const char* key) {
#if HAS_STRING_VIEW
    // Use this to match modern Conda lib symbols
    return node[std::string_view(key)];
#else
    // Fallback for older systems (C++11/14)
    return node[key];
#endif
}

}
