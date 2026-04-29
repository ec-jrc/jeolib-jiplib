#ifndef JIPLIB_JSON_COMPAT_H
#define JIPLIB_JSON_COMPAT_H

#if __cplusplus >= 201703L
    #include <string_view>
    #define HAS_STRING_VIEW 1
#else
    #define HAS_STRING_VIEW 0
#endif

#include <json/json.h>
#include <string>

namespace json_util {
    // Returns a reference to the member, compatible with various JsonCpp versions
    Json::Value& get_member(Json::Value& node, const std::string& key);
    Json::Value& get_member(Json::Value& node, const char* key);
}

#endif
