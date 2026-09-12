/* Copyright 2026 flashFloat authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/

#pragma once
#include "../tensor/tensor_view_ref.h"

#ifndef DEBUG_BLOCK
#define DEBUG_BLOCK false
#endif

namespace xpu {

template <typename T, int M, int N>
struct alignas(128) SharedBlock {
    T data[M][N];
};

} // namespace xpu
