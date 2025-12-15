// Copyright 2021 Google LLC
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

#ifdef CPU_FEATURES_ARCH_AARCH64
#if defined(CPU_FEATURES_OS_MACOS) || defined(CPU_FEATURES_OS_IPHONE)

#include "impl_aarch64__base_implementation.inl"

#if !defined(HAVE_SYSCTLBYNAME)
#error "Darwin needs support for sysctlbyname"
#endif
#include <sys/sysctl.h>

#if defined(CPU_FEATURES_MOCK_SYSCTL_AARCH64)
extern bool GetDarwinSysCtlByName(const char*);
extern int GetDarwinSysCtlByNameValue(const char* name);
#else
static int GetDarwinSysCtlByNameValue(const char* name) {
  int enabled;
  size_t enabled_len = sizeof(enabled);
  const int failure = sysctlbyname(name, &enabled, &enabled_len, NULL, 0);
  return failure ? 0 : enabled;
}

static bool GetDarwinSysCtlByName(const char* name) {
  return GetDarwinSysCtlByNameValue(name) != 0;
}
#endif

static const Aarch64Info kEmptyAarch64Info;

Aarch64Info GetAarch64Info(void) {
  Aarch64Info info = kEmptyAarch64Info;

  // Handling Darwin platform through sysctlbyname.
  info.implementer = GetDarwinSysCtlByNameValue("hw.cputype");
  info.variant = GetDarwinSysCtlByNameValue("hw.cpusubtype");
  info.part = GetDarwinSysCtlByNameValue("hw.cpufamily");
  info.revision = GetDarwinSysCtlByNameValue("hw.cpusubfamily");

  info.features.fp = GetDarwinSysCtlByName("hw.optional.floatingpoint");
  info.features.asimd = GetDarwinSysCtlByName("hw.optional.AdvSIMD");
  info.features.aes = GetDarwinSysCtlByName("hw.optional.arm.FEAT_AES");
  info.features.pmull = GetDarwinSysCtlByName("hw.optional.arm.FEAT_PMULL");
  info.features.sha1 = GetDarwinSysCtlByName("hw.optional.arm.FEAT_SHA1");
  info.features.sha2 = GetDarwinSysCtlByName("hw.optional.arm.FEAT_SHA256");
  info.features.crc32 = GetDarwinSysCtlByName("hw.optional.armv8_crc32");
  info.features.atomics = GetDarwinSysCtlByName("hw.optional.arm.FEAT_LSE");
  info.features.fphp = GetDarwinSysCtlByName("hw.optional.arm.FEAT_FP16");
  info.features.asimdhp = GetDarwinSysCtlByName("hw.optional.arm.AdvSIMD_HPFPCvt");
  info.features.asimdrdm = GetDarwinSysCtlByName("hw.optional.arm.FEAT_RDM");
  info.features.jscvt = GetDarwinSysCtlByName("hw.optional.arm.FEAT_JSCVT");
  info.features.fcma = GetDarwinSysCtlByName("hw.optional.arm.FEAT_FCMA");
  info.features.lrcpc = GetDarwinSysCtlByName("hw.optional.arm.FEAT_LRCPC");
  info.features.dcpop = GetDarwinSysCtlByName("hw.optional.arm.FEAT_DPB");
  info.features.sha3 = GetDarwinSysCtlByName("hw.optional.arm.FEAT_SHA3");
  info.features.asimddp = GetDarwinSysCtlByName("hw.optional.arm.FEAT_DotProd");
  info.features.sha512 = GetDarwinSysCtlByName("hw.optional.arm.FEAT_SHA512");
  info.features.asimdfhm = GetDarwinSysCtlByName("hw.optional.arm.FEAT_FHM");
  info.features.dit = GetDarwinSysCtlByName("hw.optional.arm.FEAT_DIT");
  info.features.uscat = GetDarwinSysCtlByName("hw.optional.arm.FEAT_LSE2");
  info.features.flagm = GetDarwinSysCtlByName("hw.optional.arm.FEAT_FlagM");
  info.features.ssbs = GetDarwinSysCtlByName("hw.optional.arm.FEAT_SSBS");
  info.features.sb = GetDarwinSysCtlByName("hw.optional.arm.FEAT_SB");
  info.features.flagm2 = GetDarwinSysCtlByName("hw.optional.arm.FEAT_FlagM2");
  info.features.frint = GetDarwinSysCtlByName("hw.optional.arm.FEAT_FRINTTS");
  info.features.i8mm = GetDarwinSysCtlByName("hw.optional.arm.FEAT_I8MM");
  info.features.bf16 = GetDarwinSysCtlByName("hw.optional.arm.FEAT_BF16");
  info.features.bti = GetDarwinSysCtlByName("hw.optional.arm.FEAT_BTI");

  return info;
}

#endif  // defined(CPU_FEATURES_OS_MACOS) || defined(CPU_FEATURES_OS_IPHONE)
#endif  // CPU_FEATURES_ARCH_AARCH64
