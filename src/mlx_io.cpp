// IO bindings for MLX
#include "mlx_helpers.hpp"
#include <mlx/io.h>
#include <Rcpp.h>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

using namespace Rcpp;
using namespace rmlx;
using namespace mlx::core;

namespace {

StreamOrDevice cpu_device() {
  return Device(Device::cpu);
}

array to_device(const array& arr, const StreamOrDevice& target) {
  return astype(arr, arr.dtype(), target);
}

std::unordered_map<std::string, array> list_to_array_map(const List& tensor_ptrs,
                                                         const CharacterVector& names,
                                                         const StreamOrDevice& target_device) {
  if (tensor_ptrs.size() != names.size()) {
    Rcpp::stop("Tensor names and values length mismatch.");
  }
  std::unordered_map<std::string, array> result;
  for (int i = 0; i < tensor_ptrs.size(); ++i) {
    MlxArrayWrapper* wrapper = get_mlx_wrapper(tensor_ptrs[i]);
    std::string name = as<std::string>(names[i]);
    array arr = wrapper->get();
    result.emplace(std::move(name), to_device(arr, target_device));
  }
  return result;
}

CharacterVector map_to_character(const std::unordered_map<std::string, std::string>& meta) {
  std::vector<std::string> names;
  names.reserve(meta.size());
  for (const auto& kv : meta) {
    names.push_back(kv.first);
  }
  std::sort(names.begin(), names.end());

  CharacterVector values(meta.size());
  CharacterVector r_names(meta.size());
  for (size_t i = 0; i < names.size(); ++i) {
    r_names[i] = names[i];
    values[i] = meta.at(names[i]);
  }
  values.attr("names") = r_names;
  return values;
}

GGUFMetaData build_meta_from_payload(const List& payload, const StreamOrDevice& target_device) {
  std::string type = as<std::string>(payload["type"]);
  if (type == "null") {
    return std::monostate{};
  }
  if (type == "array") {
    MlxArrayWrapper* wrapper = get_mlx_wrapper(payload["ptr"]);
    array arr = wrapper->get();
    return to_device(arr, target_device);
  }
  if (type == "string") {
    return as<std::string>(payload["value"]);
  }
  if (type == "string_vec") {
    CharacterVector vals(payload["value"]);
    std::vector<std::string> out(vals.size());
    for (int i = 0; i < vals.size(); ++i) {
      out[i] = as<std::string>(vals[i]);
    }
    return out;
  }
  Rcpp::stop("Unsupported GGUF metadata payload type: %s", type);
}

List wrap_gguf_metadata(const std::unordered_map<std::string, GGUFMetaData>& meta,
                        const StreamOrDevice& target_device,
                        const std::string& device_hint) {
  List out(meta.size());
  std::vector<std::string> names;
  names.reserve(meta.size());
  size_t idx = 0;
  for (const auto& kv : meta) {
    names.push_back(kv.first);
    const GGUFMetaData& value = kv.second;
    if (std::holds_alternative<std::monostate>(value)) {
      out[idx++] = R_NilValue;
    } else if (std::holds_alternative<array>(value)) {
      array arr = std::get<array>(value);
      array casted = to_device(arr, target_device);
      out[idx++] = wrap_array_as_mlx(casted, device_hint);
    } else if (std::holds_alternative<std::string>(value)) {
      out[idx++] = std::get<std::string>(value);
    } else if (std::holds_alternative<std::vector<std::string>>(value)) {
      const auto& vec = std::get<std::vector<std::string>>(value);
      CharacterVector chars(vec.size());
      for (size_t j = 0; j < vec.size(); ++j) {
        chars[j] = vec[j];
      }
      out[idx++] = chars;
    }
  }
  CharacterVector r_names(names.begin(), names.end());
  out.attr("names") = r_names;
  return out;
}

List wrap_tensor_map(const std::unordered_map<std::string, array>& tensor_map,
                     const StreamOrDevice& target_device,
                     const std::string& device_hint) {
  List tensors(tensor_map.size());
  std::vector<std::string> names;
  names.reserve(tensor_map.size());
  size_t idx = 0;
  for (const auto& kv : tensor_map) {
    names.push_back(kv.first);
    array casted = to_device(kv.second, target_device);
    tensors[idx++] = wrap_array_as_mlx(casted, device_hint);
  }
  CharacterVector r_names(names.begin(), names.end());
  tensors.attr("names") = r_names;
  return tensors;
}

} // namespace

// [[Rcpp::export]]
void cpp_mlx_save(SEXP xp_, std::string file) {
  MlxArrayWrapper* wrapper = get_mlx_wrapper(xp_);
  array arr = wrapper->get();
  array cpu_arr = to_device(arr, cpu_device());
  save(std::move(file), cpu_arr);
}

// [[Rcpp::export]]
SEXP cpp_mlx_load(std::string file, std::string device_str) {
  StreamOrDevice target_device = string_to_device(device_str);
  array cpu_arr = load(std::move(file), cpu_device());
  array result = to_device(cpu_arr, target_device);
  return make_mlx_xptr(std::move(result));
}

// [[Rcpp::export]]
void cpp_mlx_save_safetensors(List array_ptrs,
                              CharacterVector array_names,
                              CharacterVector metadata_names,
                              CharacterVector metadata_values,
                              std::string file) {
  auto tensor_map = list_to_array_map(array_ptrs, array_names, cpu_device());
  std::unordered_map<std::string, std::string> meta_map;
  for (int i = 0; i < metadata_names.size(); ++i) {
    meta_map.emplace(as<std::string>(metadata_names[i]), as<std::string>(metadata_values[i]));
  }
  save_safetensors(std::move(file), std::move(tensor_map), std::move(meta_map));
}

// [[Rcpp::export]]
List cpp_mlx_load_safetensors(std::string file, std::string device_str) {
  StreamOrDevice target_device = string_to_device(device_str);
  auto loaded = load_safetensors(std::move(file), cpu_device());

  List tensors = wrap_tensor_map(loaded.first, target_device, device_str);
  CharacterVector metadata = map_to_character(loaded.second);

  return List::create(
      Named("tensors") = tensors,
      Named("metadata") = metadata);
}

// [[Rcpp::export]]
void cpp_mlx_save_gguf(List array_ptrs,
                       CharacterVector array_names,
                       List metadata_payload,
                       CharacterVector metadata_names,
                       std::string file) {
  if (metadata_payload.size() != metadata_names.size()) {
    Rcpp::stop("Metadata names and values length mismatch.");
  }

  auto tensor_map = list_to_array_map(array_ptrs, array_names, cpu_device());
  std::unordered_map<std::string, GGUFMetaData> meta_map;
  for (int i = 0; i < metadata_payload.size(); ++i) {
    List payload(metadata_payload[i]);
    meta_map.emplace(as<std::string>(metadata_names[i]), build_meta_from_payload(payload, cpu_device()));
  }
  save_gguf(std::move(file), std::move(tensor_map), std::move(meta_map));
}

// [[Rcpp::export]]
List cpp_mlx_load_gguf(std::string file, std::string device_str) {
  StreamOrDevice target_device = string_to_device(device_str);
  auto loaded = load_gguf(std::move(file), cpu_device());

  List tensors = wrap_tensor_map(loaded.first, target_device, device_str);
  List metadata = wrap_gguf_metadata(loaded.second, target_device, device_str);

  return List::create(
      Named("tensors") = tensors,
      Named("metadata") = metadata);
}
