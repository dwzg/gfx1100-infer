// HTTP server speaking the Anthropic Messages API (POST /v1/messages with streaming, token
// counting, model listing), so that clients written for that API - Claude Code in particular -
// can use a local model.
#pragma once
#include <string>

#include "core/sampler.h"
#include "engine/context.h"
#include "prompt.h"

namespace gfx {

class GpuModel;
class Tokenizer;

struct ServerOptions {
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string api_key;              // if set, requests must present it (x-api-key or Authorization: Bearer)
    std::string model_id = "local";   // the name /v1/models reports
    std::string think = "auto";       // "auto": reason when the request enables thinking; "on"; "off"
    std::string effort;               // "low" / "medium" / "high"; empty: as the request asks
    bool keep_thinking = true;        // see RenderOptions
    SamplerParams sampling;           // used unless client_sampling and the request sets its own
    bool client_sampling = false;
    int default_max_tokens = 4096;
    bool log_prompts = false;         // write every prompt and reply to stderr
    std::string dump_dir;             // if set, every /v1/messages request body is written there (for debugging a client)
    ContextCache::Options cache;
};

// Serves until interrupted (SIGINT / SIGTERM). Returns a process exit code.
int serve(GpuModel &model, const Tokenizer &tok, const ChatTemplate &tmpl, const ServerOptions &opt);

}  // namespace gfx
