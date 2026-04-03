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

#ifndef CPU_FEATURES_INCLUDE_CPUINFO_LOONGARCH_H_
#define CPU_FEATURES_INCLUDE_CPUINFO_LOONGARCH_H_

#include "cpu_features_cache_info.h"
#include "cpu_features_macros.h"

#if !defined(CPU_FEATURES_ARCH_LOONGARCH)
#error "Including cpuinfo_loongarch.h from a non-loongarch target."
#endif

CPU_FEATURES_START_CPP_NAMESPACE

typedef struct {
  // Base
  int CPUCFG : 1;    // Instruction for Identify CPU Features 

  // Extension
  int LAM : 1;       // Extension for Atomic Memory Access Instructions
  int UAL : 1;       // Extension for Non-Aligned Memory Access
  int FPU : 1;       // Extension for Basic Floating-Point Instructions
  int LSX : 1;       // Extension for Loongson SIMD eXtension
  int LASX : 1;      // Extension for Loongson Advanced SIMD eXtension
  int CRC32 : 1;     // Extension for Cyclic Redundancy Check Instructions
  int COMPLEX : 1;   // Extension for Complex Vector Operation Instructions
  int CRYPTO : 1;    // Extension for Encryption And Decryption Vector Instructions
  int LVZ : 1;       // Extension for Virtualization
  int LBT_X86 : 1;   // Extension for X86 Binary Translation Extension
  int LBT_ARM : 1;   // Extension for ARM Binary Translation Extension
  int LBT_MIPS : 1;  // Extension for MIPS Binary Translation Extension
  int PTW : 1;       // Extension for Page Table Walker

} LoongArchFeatures;

typedef struct {
  LoongArchFeatures features;
} LoongArchInfo;

typedef enum {
  LOONGARCH_CPUCFG,
  LOONGARCH_LAM,
  LOONGARCH_UAL,
  LOONGARCH_FPU,
  LOONGARCH_LSX,
  LOONGARCH_LASX,
  LOONGARCH_CRC32,
  LOONGARCH_COMPLEX,
  LOONGARCH_CRYPTO,
  LOONGARCH_LVZ,
  LOONGARCH_LBT_X86,
  LOONGARCH_LBT_ARM,
  LOONGARCH_LBT_MIPS,
  LOONGARCH_PTW,
  LOONGARCH_LAST_,
} LoongArchFeaturesEnum;

LoongArchInfo GetLoongArchInfo(void);
int GetLoongArchFeaturesEnumValue(const LoongArchFeatures* features,
                              LoongArchFeaturesEnum value);
const char* GetLoongArchFeaturesEnumName(LoongArchFeaturesEnum);

CPU_FEATURES_END_CPP_NAMESPACE

#endif  // CPU_FEATURES_INCLUDE_CPUINFO_LOONGARCH_H_
