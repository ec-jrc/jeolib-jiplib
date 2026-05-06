#include "Json_compat.h"

namespace json_util {

Json::Value& get_member(Json::Value& node, const std::string& key) {
    return node[key];
}

Json::Value& get_member(Json::Value& node, const char* key) {
    return node[key];
}

}
