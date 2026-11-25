// Small JSON value: parser and writer. Objects keep the order of their keys, and numbers that
// were parsed keep their spelling, so a document can be written back the way it came in.
#pragma once
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace gfx {

class Json {
  public:
    enum Type { Null, Bool, Number, String, Array, Object };
    using Member = std::pair<std::string, Json>;

    Json() = default;
    Json(std::nullptr_t) {}
    Json(bool b) : type_(Bool), b_(b) {}
    Json(int v) : type_(Number), n_((double)v), s_(std::to_string(v)) {}
    Json(int64_t v) : type_(Number), n_((double)v), s_(std::to_string(v)) {}
    Json(uint64_t v) : type_(Number), n_((double)v), s_(std::to_string(v)) {}
    Json(double v);
    Json(const char *s) : type_(String), s_(s) {}
    Json(std::string s) : type_(String), s_(std::move(s)) {}
    Json(std::string_view s) : type_(String), s_(s) {}
    static Json array() { Json j; j.type_ = Array; return j; }
    static Json object() { Json j; j.type_ = Object; return j; }
    // Throws std::runtime_error with the position of the problem.
    static Json parse(std::string_view text);
    static Json parse_number_(std::string_view lexeme);   // internal

    Type type() const { return type_; }
    bool is_null() const { return type_ == Null; }
    bool is_bool() const { return type_ == Bool; }
    bool is_number() const { return type_ == Number; }
    bool is_string() const { return type_ == String; }
    bool is_array() const { return type_ == Array; }
    bool is_object() const { return type_ == Object; }

    // Lenient readers: a value of another type gives the default.
    bool boolean(bool def = false) const { return type_ == Bool ? b_ : def; }
    double number(double def = 0) const { return type_ == Number ? n_ : def; }
    int64_t integer(int64_t def = 0) const { return type_ == Number ? (int64_t)n_ : def; }
    const std::string &str() const;                       // empty unless a string
    const std::vector<Json> &items() const { return a_; }  // empty unless an array
    const std::vector<Member> &members() const { return o_; }
    // Member of an object; a null value if absent or not an object.
    const Json &operator[](std::string_view key) const;
    bool has(std::string_view key) const;

    Json &set(std::string key, Json v);   // object: replace or append
    Json &push(Json v);                   // array: append

    // spaced: ", " and ": " separators, as Python's json.dumps writes (chat templates embed
    // tool definitions that way). Otherwise compact.
    std::string dump(bool spaced = false) const;
    void dump_to(std::string &out, bool spaced) const;

  private:
    Type type_ = Null;
    bool b_ = false;
    double n_ = 0;
    std::string s_;                 // string value, or a number's spelling
    std::vector<Json> a_;
    std::vector<Member> o_;
};

// JSON string literal for text (with the quotes). Bytes that are not valid UTF-8 become U+FFFD.
void json_quote(std::string &out, std::string_view text);

}  // namespace gfx
