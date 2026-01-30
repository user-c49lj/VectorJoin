#ifndef JSON_WRAPPER_H
#define JSON_WRAPPER_H

#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <sstream>

// Simple JSON writer class to replace jsoncpp dependency
class JsonWriter {
private:
    std::ostringstream oss;
    bool first_element = true;
    
public:
    void startObject() {
        oss << "{";
        first_element = true;
    }
    
    void endObject() {
        oss << "}";
    }
    
    void startArray() {
        oss << "[";
        first_element = true;
    }
    
    void endArray() {
        oss << "]";
    }
    
    void addKey(const std::string& key) {
        if (!first_element) oss << ",";
        oss << "\"" << key << "\":";
        first_element = false;
    }
    
    void addString(const std::string& value) {
        oss << "\"" << value << "\"";
    }
    
    void addNumber(double value) {
        oss << value;
    }
    
    void addNumber(int value) {
        oss << value;
    }
    
    void addBool(bool value) {
        oss << (value ? "true" : "false");
    }
    
    void addNull() {
        oss << "null";
    }
    
    std::string toString() const {
        return oss.str();
    }
};

// Simple JSON value class
class JsonValue {
private:
    enum Type { NULL_TYPE, BOOL, NUMBER, STRING, ARRAY, OBJECT };
    Type type = NULL_TYPE;
    bool bool_value = false;
    double number_value = 0.0;
    std::string string_value;
    std::vector<JsonValue> array_value;
    std::map<std::string, JsonValue> object_value;
    
public:
    JsonValue() = default;
    JsonValue(bool b) : type(BOOL), bool_value(b) {}
    JsonValue(int n) : type(NUMBER), number_value(n) {}
    JsonValue(double n) : type(NUMBER), number_value(n) {}
    JsonValue(const std::string& s) : type(STRING), string_value(s) {}
    
    bool isNull() const { return type == NULL_TYPE; }
    bool isBool() const { return type == BOOL; }
    bool isNumber() const { return type == NUMBER; }
    bool isString() const { return type == STRING; }
    bool isArray() const { return type == ARRAY; }
    bool isObject() const { return type == OBJECT; }
    
    bool asBool() const { return bool_value; }
    double asDouble() const { return number_value; }
    int asInt() const { return static_cast<int>(number_value); }
    const std::string& asString() const { return string_value; }
    
    void append(const JsonValue& value) {
        if (type != ARRAY) {
            type = ARRAY;
            array_value.clear();
        }
        array_value.push_back(value);
    }
    
    void set(const std::string& key, const JsonValue& value) {
        if (type != OBJECT) {
            type = OBJECT;
            object_value.clear();
        }
        object_value[key] = value;
    }
    
    const JsonValue& operator[](const std::string& key) const {
        static const JsonValue null_value;
        auto it = object_value.find(key);
        return it != object_value.end() ? it->second : null_value;
    }
    
    const JsonValue& operator[](size_t index) const {
        static const JsonValue null_value;
        return index < array_value.size() ? array_value[index] : null_value;
    }
    
    size_t size() const {
        return type == ARRAY ? array_value.size() : 
               type == OBJECT ? object_value.size() : 0;
    }
};

// Simple JSON reader class
class JsonReader {
private:
    std::string json_str;
    size_t pos = 0;
    
    void skipWhitespace() {
        while (pos < json_str.length() && std::isspace(json_str[pos])) pos++;
    }
    
    std::string readString() {
        if (json_str[pos] != '"') return "";
        pos++; // skip opening quote
        std::string result;
        while (pos < json_str.length() && json_str[pos] != '"') {
            if (json_str[pos] == '\\') {
                pos++;
                if (pos < json_str.length()) {
                    result += json_str[pos];
                }
            } else {
                result += json_str[pos];
            }
            pos++;
        }
        if (pos < json_str.length()) pos++; // skip closing quote
        return result;
    }
    
    double readNumber() {
        size_t start = pos;
        while (pos < json_str.length() && 
               (std::isdigit(json_str[pos]) || json_str[pos] == '.' || 
                json_str[pos] == '-' || json_str[pos] == 'e' || json_str[pos] == 'E')) {
            pos++;
        }
        return std::stod(json_str.substr(start, pos - start));
    }
    
    JsonValue parseValue() {
        skipWhitespace();
        if (pos >= json_str.length()) return JsonValue();
        
        char c = json_str[pos];
        if (c == '"') {
            return JsonValue(readString());
        } else if (c == 't' && json_str.substr(pos, 4) == "true") {
            pos += 4;
            return JsonValue(true);
        } else if (c == 'f' && json_str.substr(pos, 5) == "false") {
            pos += 5;
            return JsonValue(false);
        } else if (c == 'n' && json_str.substr(pos, 4) == "null") {
            pos += 4;
            return JsonValue();
        } else if (c == '[') {
            JsonValue result;
            pos++; // skip '['
            skipWhitespace();
            while (pos < json_str.length() && json_str[pos] != ']') {
                result.append(parseValue());
                skipWhitespace();
                if (pos < json_str.length() && json_str[pos] == ',') {
                    pos++;
                    skipWhitespace();
                }
            }
            if (pos < json_str.length()) pos++; // skip ']'
            return result;
        } else if (c == '{') {
            JsonValue result;
            pos++; // skip '{'
            skipWhitespace();
            while (pos < json_str.length() && json_str[pos] != '}') {
                std::string key = readString();
                skipWhitespace();
                if (pos < json_str.length() && json_str[pos] == ':') {
                    pos++;
                    result.set(key, parseValue());
                }
                skipWhitespace();
                if (pos < json_str.length() && json_str[pos] == ',') {
                    pos++;
                    skipWhitespace();
                }
            }
            if (pos < json_str.length()) pos++; // skip '}'
            return result;
        } else if (std::isdigit(c) || c == '-') {
            return JsonValue(readNumber());
        }
        
        return JsonValue();
    }
    
public:
    JsonValue parse(const std::string& json) {
        json_str = json;
        pos = 0;
        return parseValue();
    }
};

#endif // JSON_WRAPPER_H 