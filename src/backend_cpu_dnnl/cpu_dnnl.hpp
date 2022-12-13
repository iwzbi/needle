#ifndef _CPU_DNNL_H
#define _CPU_DNNL_H

#include "oneapi/dnnl/dnnl.hpp"

namespace needle {
namespace cpu_dnnl {

typedef float scalar_t;
using namespace dnnl;
using tag = memory::format_tag;
using dt = memory::data_type;

void conv_forword_dnnl(scalar_t* input, scalar_t* weight, scalar_t* output,
                       const memory::dim N, const memory::dim H,
                       const memory::dim W, const memory::dim C_in,
                       const memory::dim C_out, const memory::dim K,
                       const memory::dim S, const memory::dim P);

inline void write_to_dnnl_memory(scalar_t* ptr, dnnl::memory& mem) {
  size_t size = mem.get_desc().get_size();
  if (!ptr) throw std::runtime_error("origin handle is nullptr.");
  uint8_t* dnnl_mem_handle = static_cast<uint8_t*>(mem.get_data_handle());
  if (!dnnl_mem_handle)
    throw std::runtime_error("dnnl_mem_handle returned nullptr.");
  for (size_t i = 0; i < size; ++i) dnnl_mem_handle[i] = ((uint8_t*)ptr)[i];
}

inline void read_from_dnnl_memory(scalar_t* ptr, dnnl::memory& mem) {
  size_t size = mem.get_desc().get_size();
  if (!ptr) throw std::runtime_error("origin handle is nullptr.");
  uint8_t* dnnl_mem_handle = static_cast<uint8_t*>(mem.get_data_handle());
  if (!dnnl_mem_handle)
    throw std::runtime_error("dnnl_mem_handle returned nullptr.");
  for (size_t i = 0; i < size; ++i) ((uint8_t*)ptr)[i] = dnnl_mem_handle[i];
}

}  // namespace cpu_dnnl
}  // namespace needle

#endif