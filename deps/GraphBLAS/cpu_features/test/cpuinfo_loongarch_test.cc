// Copyright 2022 Google LLC
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

#include "cpuinfo_loongarch.h"

#include "filesystem_for_testing.h"
#include "gtest/gtest.h"
#include "hwcaps_for_testing.h"

namespace cpu_features {
namespace {

TEST(CpuinfoLoongArchvTest, UnknownFromCpuInfo) {
  ResetHwcaps();
  auto& fs = GetEmptyFilesystem();
  fs.CreateFile("/proc/cpuinfo", R"(
system type		: generic-loongson-machine

processor		: 0
package			: 0
core			: 0
CPU Family		: Loongson-64bit
Model Name		: Loongson-3A5000-HV
CPU Revision		: 0x11
FPU Revision		: 0x00
CPU MHz			: 2500.00
BogoMIPS		: 5000.00
TLB Entries		: 2112
Address Sizes		: 48 bits physical, 48 bits virtual
ISA			: loongarch32 loongarch64
Features		: cpucfg lam ual fpu lsx lasx crc32 complex crypto lvz lbt_x86 lbt_arm lbt_mips
Hardware Watchpoint	: yes, iwatch count: 8, dwatch count: 8

processor		: 1
package			: 0
core			: 1
CPU Family		: Loongson-64bit
Model Name		: Loongson-3A5000-HV
CPU Revision		: 0x11
FPU Revision		: 0x00
CPU MHz			: 2500.00
BogoMIPS		: 5000.00
TLB Entries		: 2112
Address Sizes		: 48 bits physical, 48 bits virtual
ISA			: loongarch32 loongarch64
Features		: cpucfg lam ual fpu lsx lasx crc32 complex crypto lvz lbt_x86 lbt_arm lbt_mips
Hardware Watchpoint	: yes, iwatch count: 8, dwatch count: 8

processor		: 2
package			: 0
core			: 2
CPU Family		: Loongson-64bit
Model Name		: Loongson-3A5000-HV
CPU Revision		: 0x11
FPU Revision		: 0x00
CPU MHz			: 2500.00
BogoMIPS		: 5000.00
TLB Entries		: 2112
Address Sizes		: 48 bits physical, 48 bits virtual
ISA			: loongarch32 loongarch64
Features		: cpucfg lam ual fpu lsx lasx crc32 complex crypto lvz lbt_x86 lbt_arm lbt_mips
Hardware Watchpoint	: yes, iwatch count: 8, dwatch count: 8

processor		: 3
package			: 0
core			: 3
CPU Family		: Loongson-64bit
Model Name		: Loongson-3A5000-HV
CPU Revision		: 0x11
FPU Revision		: 0x00
CPU MHz			: 2500.00
BogoMIPS		: 5000.00
TLB Entries		: 2112
Address Sizes		: 48 bits physical, 48 bits virtual
ISA			: loongarch32 loongarch64
Features		: cpucfg lam ual fpu lsx lasx crc32 complex crypto lvz lbt_x86 lbt_arm lbt_mips
Hardware Watchpoint	: yes, iwatch count: 8, dwatch count: 8)");
  const auto info = GetLoongArchInfo();
  EXPECT_FALSE(info.features.CPUCFG);
  EXPECT_TRUE(info.features.LAM);
  EXPECT_TRUE(info.features.UAL);
  EXPECT_TRUE(info.features.FPU);
  EXPECT_TRUE(info.features.LSX);
  EXPECT_TRUE(info.features.LASX);
  EXPECT_TRUE(info.features.CRC32);
  EXPECT_TRUE(info.features.COMPLEX);
  EXPECT_TRUE(info.features.CRYPTO);
  EXPECT_TRUE(info.features.LVZ);
  EXPECT_TRUE(info.features.LBT_X86);
  EXPECT_TRUE(info.features.LBT_ARM);
  EXPECT_TRUE(info.features.LBT_MIPS);
}

TEST(CpuinfoLoongArchvTest, QemuCpuInfo) {
  ResetHwcaps();
  auto& fs = GetEmptyFilesystem();
  fs.CreateFile("/proc/cpuinfo", R"(
system type		: generic-loongson-machine

processor		: 0
package			: 0
core			: 0
CPU Family		: Loongson-64bit
Model Name		: Loongson-3A5000
CPU Revision		: 0x10
FPU Revision		: 0x01
CPU MHz			: 2000.00
BogoMIPS		: 4000.00
TLB Entries		: 2112
Address Sizes		: 48 bits physical, 48 bits virtual
ISA			: loongarch32 loongarch64
Features		: cpucfg lam ual fpu crc32
Hardware Watchpoint	: yes, iwatch count: 0, dwatch count: 0

processor		: 1
package			: 0
core			: 1
CPU Family		: Loongson-64bit
Model Name		: Loongson-3A5000
CPU Revision		: 0x10
FPU Revision		: 0x01
CPU MHz			: 2000.00
BogoMIPS		: 4000.00
TLB Entries		: 2112
Address Sizes		: 48 bits physical, 48 bits virtual
ISA			: loongarch32 loongarch64
Features		: cpucfg lam ual fpu crc32
Hardware Watchpoint	: yes, iwatch count: 0, dwatch count: 0

processor		: 2
package			: 0
core			: 2
CPU Family		: Loongson-64bit
Model Name		: Loongson-3A5000
CPU Revision		: 0x10
FPU Revision		: 0x01
CPU MHz			: 2000.00
BogoMIPS		: 4000.00
TLB Entries		: 2112
Address Sizes		: 48 bits physical, 48 bits virtual
ISA			: loongarch32 loongarch64
Features		: cpucfg lam ual fpu crc32
Hardware Watchpoint	: yes, iwatch count: 0, dwatch count: 0

processor		: 3
package			: 0
core			: 3
CPU Family		: Loongson-64bit
Model Name		: Loongson-3A5000
CPU Revision		: 0x10
FPU Revision		: 0x01
CPU MHz			: 2000.00
BogoMIPS		: 4000.00
TLB Entries		: 2112
Address Sizes		: 48 bits physical, 48 bits virtual
ISA			: loongarch32 loongarch64
Features		: cpucfg lam ual fpu crc32
Hardware Watchpoint	: yes, iwatch count: 0, dwatch count: 0)");
  const auto info = GetLoongArchInfo();
  EXPECT_FALSE(info.features.CPUCFG);
  EXPECT_TRUE(info.features.LAM);
  EXPECT_TRUE(info.features.UAL);
  EXPECT_TRUE(info.features.FPU);
  EXPECT_TRUE(info.features.CRC32);
}

}  // namespace
}  // namespace cpu_features
