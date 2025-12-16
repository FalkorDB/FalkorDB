// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "cpu_features_macros.h"

#ifdef CPU_FEATURES_ARCH_LOONGARCH
#if defined(CPU_FEATURES_OS_LINUX)

#include "cpuinfo_loongarch.h"

////////////////////////////////////////////////////////////////////////////////
// Definitions for introspection.
////////////////////////////////////////////////////////////////////////////////
#define INTROSPECTION_TABLE                            \
  LINE(LOONGARCH_CPUCFG, CPUCFG, "cfg", HWCAP_LOONGARCH_CPUCFG, 0) \
  LINE(LOONGARCH_LAM, LAM, "lam", HWCAP_LOONGARCH_LAM, 0) \
  LINE(LOONGARCH_UAL, UAL, "ual", HWCAP_LOONGARCH_UAL, 0) \
  LINE(LOONGARCH_FPU, FPU, "fpu", HWCAP_LOONGARCH_FPU, 0) \
  LINE(LOONGARCH_LSX, LSX, "lsx", HWCAP_LOONGARCH_LSX, 0) \
  LINE(LOONGARCH_LASX, LASX, "lasx", HWCAP_LOONGARCH_LASX, 0) \
  LINE(LOONGARCH_CRC32, CRC32, "crc32", HWCAP_LOONGARCH_CRC32, 0) \
  LINE(LOONGARCH_COMPLEX, COMPLEX, "complex", HWCAP_LOONGARCH_COMPLEX, 0) \
  LINE(LOONGARCH_CRYPTO, CRYPTO, "crypto", HWCAP_LOONGARCH_CRYPTO, 0) \
  LINE(LOONGARCH_LVZ, LVZ, "lvz", HWCAP_LOONGARCH_LVZ, 0) \
  LINE(LOONGARCH_LBT_X86, LBT_X86, "lbt_x86", HWCAP_LOONGARCH_LBT_X86, 0) \
  LINE(LOONGARCH_LBT_ARM, LBT_ARM, "lbt_arm", HWCAP_LOONGARCH_LBT_ARM, 0) \
  LINE(LOONGARCH_LBT_MIPS, LBT_MIPS, "lbt_mips", HWCAP_LOONGARCH_LBT_MIPS, 0) \
  LINE(LOONGARCH_PTW, PTW, "ptw", HWCAP_LOONGARCH_PTW, 0)
#define INTROSPECTION_PREFIX LoongArch
#define INTROSPECTION_ENUM_PREFIX LOONGARCH
#include "define_introspection_and_hwcaps.inl"

////////////////////////////////////////////////////////////////////////////////
// Implementation.
////////////////////////////////////////////////////////////////////////////////

#include <stdbool.h>
#include <stdio.h>

#include "internal/filesystem.h"
#include "internal/stack_line_reader.h"

static const LoongArchInfo kEmptyLoongArchInfo;

static bool HandleLoongArchLine(const LineResult result, LoongArchInfo* const info) {
  StringView line = result.line;
  StringView key, value;
  if (CpuFeatures_StringView_GetAttributeKeyValue(line, &key, &value)) {
    if (CpuFeatures_StringView_IsEquals(key, str("Features"))) {
      for (size_t i = 0; i < LOONGARCH_LAST_; ++i) {
        kSetters[i](&info->features, CpuFeatures_StringView_HasWord(
                                         value, kCpuInfoFlags[i], ' '));
      }
    }
  }
  return !result.eof;
}

static void FillProcCpuInfoData(LoongArchInfo* const info) {
  const int fd = CpuFeatures_OpenFile("/proc/cpuinfo");
  if (fd >= 0) {
    StackLineReader reader;
    StackLineReader_Initialize(&reader, fd);
    for (;;) {
      if (!HandleLoongArchLine(StackLineReader_NextLine(&reader), info)) break;
    }
    CpuFeatures_CloseFile(fd);
  }
}

LoongArchInfo GetLoongArchInfo(void) {
  LoongArchInfo info = kEmptyLoongArchInfo;
  FillProcCpuInfoData(&info);
  return info;
}

#endif  // defined(CPU_FEATURES_OS_LINUX)
#endif  // CPU_FEATURES_ARCH_LOONGARCH
