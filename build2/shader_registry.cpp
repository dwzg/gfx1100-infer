#include <cstdint>
#include <cstddef>
#include "vk/vk.h"
#include "shaders/matvec_f16.h"
#include "shaders/matvec_q4_k.h"
#include "shaders/matmat_q4_k.h"
#include "shaders/matmat2_q4_k.h"
#include "shaders/wmma_q4_k.h"
#include "shaders/wmmaemu_q4_k.h"
#include "shaders/expert_q4_k.h"
#include "shaders/pair_q4_k.h"
#include "shaders/expertpair_q4_k.h"
#include "shaders/matvec_q5_k.h"
#include "shaders/matmat_q5_k.h"
#include "shaders/matmat2_q5_k.h"
#include "shaders/wmma_q5_k.h"
#include "shaders/wmmaemu_q5_k.h"
#include "shaders/expert_q5_k.h"
#include "shaders/pair_q5_k.h"
#include "shaders/expertpair_q5_k.h"
#include "shaders/matvec_q6_k.h"
#include "shaders/matmat_q6_k.h"
#include "shaders/matmat2_q6_k.h"
#include "shaders/wmma_q6_k.h"
#include "shaders/wmmaemu_q6_k.h"
#include "shaders/expert_q6_k.h"
#include "shaders/pair_q6_k.h"
#include "shaders/expertpair_q6_k.h"
#include "shaders/matvec_q3_k.h"
#include "shaders/matmat_q3_k.h"
#include "shaders/matmat2_q3_k.h"
#include "shaders/wmma_q3_k.h"
#include "shaders/wmmaemu_q3_k.h"
#include "shaders/expert_q3_k.h"
#include "shaders/pair_q3_k.h"
#include "shaders/expertpair_q3_k.h"
#include "shaders/matvec_iq4_xs.h"
#include "shaders/matmat_iq4_xs.h"
#include "shaders/matmat2_iq4_xs.h"
#include "shaders/wmma_iq4_xs.h"
#include "shaders/wmmaemu_iq4_xs.h"
#include "shaders/expert_iq4_xs.h"
#include "shaders/pair_iq4_xs.h"
#include "shaders/expertpair_iq4_xs.h"
#include "shaders/matvec_iq4_nl.h"
#include "shaders/matmat_iq4_nl.h"
#include "shaders/matmat2_iq4_nl.h"
#include "shaders/wmma_iq4_nl.h"
#include "shaders/wmmaemu_iq4_nl.h"
#include "shaders/expert_iq4_nl.h"
#include "shaders/pair_iq4_nl.h"
#include "shaders/expertpair_iq4_nl.h"
#include "shaders/matvec_q4_0.h"
#include "shaders/matmat_q4_0.h"
#include "shaders/matmat2_q4_0.h"
#include "shaders/wmma_q4_0.h"
#include "shaders/wmmaemu_q4_0.h"
#include "shaders/expert_q4_0.h"
#include "shaders/pair_q4_0.h"
#include "shaders/expertpair_q4_0.h"
#include "shaders/matvec_q8_0.h"
#include "shaders/matmat_q8_0.h"
#include "shaders/matmat2_q8_0.h"
#include "shaders/wmma_q8_0.h"
#include "shaders/wmmaemu_q8_0.h"
#include "shaders/expert_q8_0.h"
#include "shaders/pair_q8_0.h"
#include "shaders/expertpair_q8_0.h"
#include "shaders/matvec_nvfp4.h"
#include "shaders/matmat_nvfp4.h"
#include "shaders/matmat2_nvfp4.h"
#include "shaders/wmma_nvfp4.h"
#include "shaders/wmmaemu_nvfp4.h"
#include "shaders/expert_nvfp4.h"
#include "shaders/pair_nvfp4.h"
#include "shaders/expertpair_nvfp4.h"
#include "shaders/matvec_iq3_s.h"
#include "shaders/matmat_iq3_s.h"
#include "shaders/matmat2_iq3_s.h"
#include "shaders/wmma_iq3_s.h"
#include "shaders/wmmaemu_iq3_s.h"
#include "shaders/expert_iq3_s.h"
#include "shaders/pair_iq3_s.h"
#include "shaders/expertpair_iq3_s.h"
#include "shaders/matvec_iq3_xxs.h"
#include "shaders/matmat_iq3_xxs.h"
#include "shaders/matmat2_iq3_xxs.h"
#include "shaders/wmma_iq3_xxs.h"
#include "shaders/wmmaemu_iq3_xxs.h"
#include "shaders/expert_iq3_xxs.h"
#include "shaders/pair_iq3_xxs.h"
#include "shaders/expertpair_iq3_xxs.h"
#include "shaders/matvec_iq2_s.h"
#include "shaders/matmat_iq2_s.h"
#include "shaders/matmat2_iq2_s.h"
#include "shaders/wmma_iq2_s.h"
#include "shaders/wmmaemu_iq2_s.h"
#include "shaders/expert_iq2_s.h"
#include "shaders/pair_iq2_s.h"
#include "shaders/expertpair_iq2_s.h"
#include "shaders/matvec_iq2_xs.h"
#include "shaders/matmat_iq2_xs.h"
#include "shaders/matmat2_iq2_xs.h"
#include "shaders/wmma_iq2_xs.h"
#include "shaders/wmmaemu_iq2_xs.h"
#include "shaders/expert_iq2_xs.h"
#include "shaders/pair_iq2_xs.h"
#include "shaders/expertpair_iq2_xs.h"
#include "shaders/matvec_iq2_xxs.h"
#include "shaders/matmat_iq2_xxs.h"
#include "shaders/matmat2_iq2_xxs.h"
#include "shaders/wmma_iq2_xxs.h"
#include "shaders/wmmaemu_iq2_xxs.h"
#include "shaders/expert_iq2_xxs.h"
#include "shaders/pair_iq2_xxs.h"
#include "shaders/expertpair_iq2_xxs.h"
#include "shaders/blocksum.h"
#include "shaders/memread.h"
#include "shaders/norm.h"
#include "shaders/swiglu.h"
#include "shaders/swiglu_pair.h"
#include "shaders/attn_prep.h"
#include "shaders/attn_prep_kv8.h"
#include "shaders/attn_score.h"
#include "shaders/attn_score_kv8.h"
#include "shaders/attn_soft.h"
#include "shaders/attn_vsum.h"
#include "shaders/attn_vsum_kv8.h"
#include "shaders/attn_combine.h"
#include "shaders/dn_prep.h"
#include "shaders/dn_update.h"
#include "shaders/dn_hist.h"
#include "shaders/matmat_f16.h"
#include "shaders/matmat2_f16.h"
#include "shaders/to_f16.h"
#include "shaders/matvec_f32.h"
#include "shaders/ink_mix.h"
#include "shaders/ink_ctl_low.h"
#include "shaders/ink_ctl_out.h"
#include "shaders/ink_conv.h"
#include "shaders/ink_route.h"
#include "shaders/ink_prep.h"
#include "shaders/ink_prep_kv8.h"
#include "shaders/ink_score.h"
#include "shaders/ink_score_kv8.h"
#include "shaders/ink_wscore.h"
#include "shaders/ink_wscore_kv8.h"
#include "shaders/ink_soft.h"
#include "shaders/ink_wsoft.h"
#include "shaders/ink_wvsum.h"
#include "shaders/ink_wvsum_kv8.h"
namespace gfx {
const std::vector<ShaderBlob> &all_shaders() {
  static const std::vector<ShaderBlob> v = {
    {"matvec_f16", spv_matvec_f16, sizeof(spv_matvec_f16)},
    {"matvec_q4_k", spv_matvec_q4_k, sizeof(spv_matvec_q4_k)},
    {"matmat_q4_k", spv_matmat_q4_k, sizeof(spv_matmat_q4_k)},
    {"matmat2_q4_k", spv_matmat2_q4_k, sizeof(spv_matmat2_q4_k)},
    {"wmma_q4_k", spv_wmma_q4_k, sizeof(spv_wmma_q4_k)},
    {"wmmaemu_q4_k", spv_wmmaemu_q4_k, sizeof(spv_wmmaemu_q4_k)},
    {"expert_q4_k", spv_expert_q4_k, sizeof(spv_expert_q4_k)},
    {"pair_q4_k", spv_pair_q4_k, sizeof(spv_pair_q4_k)},
    {"expertpair_q4_k", spv_expertpair_q4_k, sizeof(spv_expertpair_q4_k)},
    {"matvec_q5_k", spv_matvec_q5_k, sizeof(spv_matvec_q5_k)},
    {"matmat_q5_k", spv_matmat_q5_k, sizeof(spv_matmat_q5_k)},
    {"matmat2_q5_k", spv_matmat2_q5_k, sizeof(spv_matmat2_q5_k)},
    {"wmma_q5_k", spv_wmma_q5_k, sizeof(spv_wmma_q5_k)},
    {"wmmaemu_q5_k", spv_wmmaemu_q5_k, sizeof(spv_wmmaemu_q5_k)},
    {"expert_q5_k", spv_expert_q5_k, sizeof(spv_expert_q5_k)},
    {"pair_q5_k", spv_pair_q5_k, sizeof(spv_pair_q5_k)},
    {"expertpair_q5_k", spv_expertpair_q5_k, sizeof(spv_expertpair_q5_k)},
    {"matvec_q6_k", spv_matvec_q6_k, sizeof(spv_matvec_q6_k)},
    {"matmat_q6_k", spv_matmat_q6_k, sizeof(spv_matmat_q6_k)},
    {"matmat2_q6_k", spv_matmat2_q6_k, sizeof(spv_matmat2_q6_k)},
    {"wmma_q6_k", spv_wmma_q6_k, sizeof(spv_wmma_q6_k)},
    {"wmmaemu_q6_k", spv_wmmaemu_q6_k, sizeof(spv_wmmaemu_q6_k)},
    {"expert_q6_k", spv_expert_q6_k, sizeof(spv_expert_q6_k)},
    {"pair_q6_k", spv_pair_q6_k, sizeof(spv_pair_q6_k)},
    {"expertpair_q6_k", spv_expertpair_q6_k, sizeof(spv_expertpair_q6_k)},
    {"matvec_q3_k", spv_matvec_q3_k, sizeof(spv_matvec_q3_k)},
    {"matmat_q3_k", spv_matmat_q3_k, sizeof(spv_matmat_q3_k)},
    {"matmat2_q3_k", spv_matmat2_q3_k, sizeof(spv_matmat2_q3_k)},
    {"wmma_q3_k", spv_wmma_q3_k, sizeof(spv_wmma_q3_k)},
    {"wmmaemu_q3_k", spv_wmmaemu_q3_k, sizeof(spv_wmmaemu_q3_k)},
    {"expert_q3_k", spv_expert_q3_k, sizeof(spv_expert_q3_k)},
    {"pair_q3_k", spv_pair_q3_k, sizeof(spv_pair_q3_k)},
    {"expertpair_q3_k", spv_expertpair_q3_k, sizeof(spv_expertpair_q3_k)},
    {"matvec_iq4_xs", spv_matvec_iq4_xs, sizeof(spv_matvec_iq4_xs)},
    {"matmat_iq4_xs", spv_matmat_iq4_xs, sizeof(spv_matmat_iq4_xs)},
    {"matmat2_iq4_xs", spv_matmat2_iq4_xs, sizeof(spv_matmat2_iq4_xs)},
    {"wmma_iq4_xs", spv_wmma_iq4_xs, sizeof(spv_wmma_iq4_xs)},
    {"wmmaemu_iq4_xs", spv_wmmaemu_iq4_xs, sizeof(spv_wmmaemu_iq4_xs)},
    {"expert_iq4_xs", spv_expert_iq4_xs, sizeof(spv_expert_iq4_xs)},
    {"pair_iq4_xs", spv_pair_iq4_xs, sizeof(spv_pair_iq4_xs)},
    {"expertpair_iq4_xs", spv_expertpair_iq4_xs, sizeof(spv_expertpair_iq4_xs)},
    {"matvec_iq4_nl", spv_matvec_iq4_nl, sizeof(spv_matvec_iq4_nl)},
    {"matmat_iq4_nl", spv_matmat_iq4_nl, sizeof(spv_matmat_iq4_nl)},
    {"matmat2_iq4_nl", spv_matmat2_iq4_nl, sizeof(spv_matmat2_iq4_nl)},
    {"wmma_iq4_nl", spv_wmma_iq4_nl, sizeof(spv_wmma_iq4_nl)},
    {"wmmaemu_iq4_nl", spv_wmmaemu_iq4_nl, sizeof(spv_wmmaemu_iq4_nl)},
    {"expert_iq4_nl", spv_expert_iq4_nl, sizeof(spv_expert_iq4_nl)},
    {"pair_iq4_nl", spv_pair_iq4_nl, sizeof(spv_pair_iq4_nl)},
    {"expertpair_iq4_nl", spv_expertpair_iq4_nl, sizeof(spv_expertpair_iq4_nl)},
    {"matvec_q4_0", spv_matvec_q4_0, sizeof(spv_matvec_q4_0)},
    {"matmat_q4_0", spv_matmat_q4_0, sizeof(spv_matmat_q4_0)},
    {"matmat2_q4_0", spv_matmat2_q4_0, sizeof(spv_matmat2_q4_0)},
    {"wmma_q4_0", spv_wmma_q4_0, sizeof(spv_wmma_q4_0)},
    {"wmmaemu_q4_0", spv_wmmaemu_q4_0, sizeof(spv_wmmaemu_q4_0)},
    {"expert_q4_0", spv_expert_q4_0, sizeof(spv_expert_q4_0)},
    {"pair_q4_0", spv_pair_q4_0, sizeof(spv_pair_q4_0)},
    {"expertpair_q4_0", spv_expertpair_q4_0, sizeof(spv_expertpair_q4_0)},
    {"matvec_q8_0", spv_matvec_q8_0, sizeof(spv_matvec_q8_0)},
    {"matmat_q8_0", spv_matmat_q8_0, sizeof(spv_matmat_q8_0)},
    {"matmat2_q8_0", spv_matmat2_q8_0, sizeof(spv_matmat2_q8_0)},
    {"wmma_q8_0", spv_wmma_q8_0, sizeof(spv_wmma_q8_0)},
    {"wmmaemu_q8_0", spv_wmmaemu_q8_0, sizeof(spv_wmmaemu_q8_0)},
    {"expert_q8_0", spv_expert_q8_0, sizeof(spv_expert_q8_0)},
    {"pair_q8_0", spv_pair_q8_0, sizeof(spv_pair_q8_0)},
    {"expertpair_q8_0", spv_expertpair_q8_0, sizeof(spv_expertpair_q8_0)},
    {"matvec_nvfp4", spv_matvec_nvfp4, sizeof(spv_matvec_nvfp4)},
    {"matmat_nvfp4", spv_matmat_nvfp4, sizeof(spv_matmat_nvfp4)},
    {"matmat2_nvfp4", spv_matmat2_nvfp4, sizeof(spv_matmat2_nvfp4)},
    {"wmma_nvfp4", spv_wmma_nvfp4, sizeof(spv_wmma_nvfp4)},
    {"wmmaemu_nvfp4", spv_wmmaemu_nvfp4, sizeof(spv_wmmaemu_nvfp4)},
    {"expert_nvfp4", spv_expert_nvfp4, sizeof(spv_expert_nvfp4)},
    {"pair_nvfp4", spv_pair_nvfp4, sizeof(spv_pair_nvfp4)},
    {"expertpair_nvfp4", spv_expertpair_nvfp4, sizeof(spv_expertpair_nvfp4)},
    {"matvec_iq3_s", spv_matvec_iq3_s, sizeof(spv_matvec_iq3_s)},
    {"matmat_iq3_s", spv_matmat_iq3_s, sizeof(spv_matmat_iq3_s)},
    {"matmat2_iq3_s", spv_matmat2_iq3_s, sizeof(spv_matmat2_iq3_s)},
    {"wmma_iq3_s", spv_wmma_iq3_s, sizeof(spv_wmma_iq3_s)},
    {"wmmaemu_iq3_s", spv_wmmaemu_iq3_s, sizeof(spv_wmmaemu_iq3_s)},
    {"expert_iq3_s", spv_expert_iq3_s, sizeof(spv_expert_iq3_s)},
    {"pair_iq3_s", spv_pair_iq3_s, sizeof(spv_pair_iq3_s)},
    {"expertpair_iq3_s", spv_expertpair_iq3_s, sizeof(spv_expertpair_iq3_s)},
    {"matvec_iq3_xxs", spv_matvec_iq3_xxs, sizeof(spv_matvec_iq3_xxs)},
    {"matmat_iq3_xxs", spv_matmat_iq3_xxs, sizeof(spv_matmat_iq3_xxs)},
    {"matmat2_iq3_xxs", spv_matmat2_iq3_xxs, sizeof(spv_matmat2_iq3_xxs)},
    {"wmma_iq3_xxs", spv_wmma_iq3_xxs, sizeof(spv_wmma_iq3_xxs)},
    {"wmmaemu_iq3_xxs", spv_wmmaemu_iq3_xxs, sizeof(spv_wmmaemu_iq3_xxs)},
    {"expert_iq3_xxs", spv_expert_iq3_xxs, sizeof(spv_expert_iq3_xxs)},
    {"pair_iq3_xxs", spv_pair_iq3_xxs, sizeof(spv_pair_iq3_xxs)},
    {"expertpair_iq3_xxs", spv_expertpair_iq3_xxs, sizeof(spv_expertpair_iq3_xxs)},
    {"matvec_iq2_s", spv_matvec_iq2_s, sizeof(spv_matvec_iq2_s)},
    {"matmat_iq2_s", spv_matmat_iq2_s, sizeof(spv_matmat_iq2_s)},
    {"matmat2_iq2_s", spv_matmat2_iq2_s, sizeof(spv_matmat2_iq2_s)},
    {"wmma_iq2_s", spv_wmma_iq2_s, sizeof(spv_wmma_iq2_s)},
    {"wmmaemu_iq2_s", spv_wmmaemu_iq2_s, sizeof(spv_wmmaemu_iq2_s)},
    {"expert_iq2_s", spv_expert_iq2_s, sizeof(spv_expert_iq2_s)},
    {"pair_iq2_s", spv_pair_iq2_s, sizeof(spv_pair_iq2_s)},
    {"expertpair_iq2_s", spv_expertpair_iq2_s, sizeof(spv_expertpair_iq2_s)},
    {"matvec_iq2_xs", spv_matvec_iq2_xs, sizeof(spv_matvec_iq2_xs)},
    {"matmat_iq2_xs", spv_matmat_iq2_xs, sizeof(spv_matmat_iq2_xs)},
    {"matmat2_iq2_xs", spv_matmat2_iq2_xs, sizeof(spv_matmat2_iq2_xs)},
    {"wmma_iq2_xs", spv_wmma_iq2_xs, sizeof(spv_wmma_iq2_xs)},
    {"wmmaemu_iq2_xs", spv_wmmaemu_iq2_xs, sizeof(spv_wmmaemu_iq2_xs)},
    {"expert_iq2_xs", spv_expert_iq2_xs, sizeof(spv_expert_iq2_xs)},
    {"pair_iq2_xs", spv_pair_iq2_xs, sizeof(spv_pair_iq2_xs)},
    {"expertpair_iq2_xs", spv_expertpair_iq2_xs, sizeof(spv_expertpair_iq2_xs)},
    {"matvec_iq2_xxs", spv_matvec_iq2_xxs, sizeof(spv_matvec_iq2_xxs)},
    {"matmat_iq2_xxs", spv_matmat_iq2_xxs, sizeof(spv_matmat_iq2_xxs)},
    {"matmat2_iq2_xxs", spv_matmat2_iq2_xxs, sizeof(spv_matmat2_iq2_xxs)},
    {"wmma_iq2_xxs", spv_wmma_iq2_xxs, sizeof(spv_wmma_iq2_xxs)},
    {"wmmaemu_iq2_xxs", spv_wmmaemu_iq2_xxs, sizeof(spv_wmmaemu_iq2_xxs)},
    {"expert_iq2_xxs", spv_expert_iq2_xxs, sizeof(spv_expert_iq2_xxs)},
    {"pair_iq2_xxs", spv_pair_iq2_xxs, sizeof(spv_pair_iq2_xxs)},
    {"expertpair_iq2_xxs", spv_expertpair_iq2_xxs, sizeof(spv_expertpair_iq2_xxs)},
    {"blocksum", spv_blocksum, sizeof(spv_blocksum)},
    {"memread", spv_memread, sizeof(spv_memread)},
    {"norm", spv_norm, sizeof(spv_norm)},
    {"swiglu", spv_swiglu, sizeof(spv_swiglu)},
    {"swiglu_pair", spv_swiglu_pair, sizeof(spv_swiglu_pair)},
    {"attn_prep", spv_attn_prep, sizeof(spv_attn_prep)},
    {"attn_prep_kv8", spv_attn_prep_kv8, sizeof(spv_attn_prep_kv8)},
    {"attn_score", spv_attn_score, sizeof(spv_attn_score)},
    {"attn_score_kv8", spv_attn_score_kv8, sizeof(spv_attn_score_kv8)},
    {"attn_soft", spv_attn_soft, sizeof(spv_attn_soft)},
    {"attn_vsum", spv_attn_vsum, sizeof(spv_attn_vsum)},
    {"attn_vsum_kv8", spv_attn_vsum_kv8, sizeof(spv_attn_vsum_kv8)},
    {"attn_combine", spv_attn_combine, sizeof(spv_attn_combine)},
    {"dn_prep", spv_dn_prep, sizeof(spv_dn_prep)},
    {"dn_update", spv_dn_update, sizeof(spv_dn_update)},
    {"dn_hist", spv_dn_hist, sizeof(spv_dn_hist)},
    {"matmat_f16", spv_matmat_f16, sizeof(spv_matmat_f16)},
    {"matmat2_f16", spv_matmat2_f16, sizeof(spv_matmat2_f16)},
    {"to_f16", spv_to_f16, sizeof(spv_to_f16)},
    {"matvec_f32", spv_matvec_f32, sizeof(spv_matvec_f32)},
    {"ink_mix", spv_ink_mix, sizeof(spv_ink_mix)},
    {"ink_ctl_low", spv_ink_ctl_low, sizeof(spv_ink_ctl_low)},
    {"ink_ctl_out", spv_ink_ctl_out, sizeof(spv_ink_ctl_out)},
    {"ink_conv", spv_ink_conv, sizeof(spv_ink_conv)},
    {"ink_route", spv_ink_route, sizeof(spv_ink_route)},
    {"ink_prep", spv_ink_prep, sizeof(spv_ink_prep)},
    {"ink_prep_kv8", spv_ink_prep_kv8, sizeof(spv_ink_prep_kv8)},
    {"ink_score", spv_ink_score, sizeof(spv_ink_score)},
    {"ink_score_kv8", spv_ink_score_kv8, sizeof(spv_ink_score_kv8)},
    {"ink_wscore", spv_ink_wscore, sizeof(spv_ink_wscore)},
    {"ink_wscore_kv8", spv_ink_wscore_kv8, sizeof(spv_ink_wscore_kv8)},
    {"ink_soft", spv_ink_soft, sizeof(spv_ink_soft)},
    {"ink_wsoft", spv_ink_wsoft, sizeof(spv_ink_wsoft)},
    {"ink_wvsum", spv_ink_wvsum, sizeof(spv_ink_wvsum)},
    {"ink_wvsum_kv8", spv_ink_wvsum_kv8, sizeof(spv_ink_wvsum_kv8)},
  };
  return v;
}
}
