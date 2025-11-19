# Lacuna-Kernels 重构计划

## 目标
按 Array API 标准语义重组内核布局：
- 一个方法一个 .rs 文件
- 每个文件包含该方法对所有类型的实现（CSR/CSC/COO/COOND）
- 使用文件夹按 Array API 分类组织

## 现有代码分布

### arith.rs (1447 lines)
**Functions**:
- `add_csr_f64_i64`, `add_csc_f64_i64`, `add_coond_f64_i64`
- `sub_csr_f64_i64`, `sub_csc_f64_i64`, `sub_coond_f64_i64`
- `hadamard_csr_f64_i64`, `hadamard_csc_f64_i64`, `hadamard_coond_f64_i64`, `hadamard_broadcast_coond_f64_i64`
- `mul_scalar_f64` (CSR), `mul_scalar_csc_f64`, `mul_scalar_coo_f64`, `mul_scalar_coond_f64`

**Helpers** (private): `add_row_count`, `add_row_fill`, `sub_row_count`, `sub_row_fill`, `hadamard_row_count`, `hadamard_row_fill`, `i64_to_usize`, `usize_to_i64`, `build_strides_row_major`

### reduce.rs (27254 bytes)
**Functions**:
- `sum_f64` (CSR), `sum_csc_f64`, `sum_coo_f64`, `sum_coond_f64`
- `row_sums_f64` (CSR), `row_sums_csc_f64`, `row_sums_coo_f64`
- `col_sums_f64` (CSR), `col_sums_csc_f64`, `col_sums_coo_f64`
- `mean_coond_f64`, `reduce_mean_axes_coond_f64_i64`
- `reduce_sum_axes_coond_f64_i64`

### spmv.rs (20127 bytes)
**Functions**:
- `spmv_f64_i64` (CSR), `spmv_csc_f64_i64`, `spmv_coo_f64_i64`, `spmv_coond_f64_i64`

### spmm.rs (13951 bytes)
**Functions**:
- `spmm_f64_i64`, `spmm_auto_f64_i64` (CSR)
- `spmm_csc_f64_i64`, `spmm_coo_f64_i64`, `spmm_coond_f64_i64`

### transform.rs (26959 bytes)
**Functions**:
- `transpose_f64_i64` (CSR), `transpose_csc_f64_i64`, `transpose_coo_f64_i64`
- `permute_axes_coond_f64_i64`
- `reshape_coond_f64_i64`

### convert.rs (13765 bytes)
**Functions**:
- `csr_to_csc_f64_i64`, `csc_to_csr_f64_i64`
- `csr_to_coo_f64_i64`, `coo_to_csr_f64_i64`
- `csc_to_coo_f64_i64`, `coo_to_csc_f64_i64`
- `coond_axes_to_csr_f64_i64`, `coond_axes_to_csc_f64_i64`
- `coond_mode_to_csr_f64_i64`, `coond_mode_to_csc_f64_i64`

### cleanup.rs (11870 bytes)
**Functions**:
- `prune_eps`, `prune_eps_coo`, `prune_eps_coond`, `prune_eps_csc`
- `eliminate_zeros`, `eliminate_zeros_coo`, `eliminate_zeros_coond`, `eliminate_zeros_csc`

## 新目录结构

```
crates/lacuna-kernels/src/
  lib.rs
  util.rs                    # shared utilities (i64_to_usize, etc.)
  
  elementwise/
    mod.rs
    add.rs                   # add: CSR/CSC/COO/COOND + helpers
    subtract.rs              # subtract: CSR/CSC/COOND + helpers
    multiply.rs              # hadamard (all) + hadamard_broadcast + mul_scalar (all)
    
  linalg/
    mod.rs
    matmul.rs                # spmv + spmm for all formats
    matrix_transpose.rs      # transpose for CSR/CSC/COO
    
  statistical/
    mod.rs
    sum.rs                   # sum + row_sums + col_sums + reduce_sum_axes
    mean.rs                  # mean + reduce_mean_axes
    
  manipulation/
    mod.rs
    permute_dims.rs          # permute_axes for COOND
    reshape.rs               # reshape for COOND
    
  data_type_functions/
    mod.rs
    astype.rs                # all format conversions
    
  utility/
    mod.rs
    prune.rs                 # prune_eps for all formats
    eliminate_zeros.rs       # eliminate_zeros for all formats
```

## 迁移映射表

| 旧文件 | 函数 | 新位置 |
|--------|------|--------|
| arith.rs | add_* | elementwise/add.rs |
| arith.rs | sub_* | elementwise/subtract.rs |
| arith.rs | hadamard_*, mul_scalar_* | elementwise/multiply.rs |
| reduce.rs | sum_*, row_sums_*, col_sums_*, reduce_sum_* | statistical/sum.rs |
| reduce.rs | mean_*, reduce_mean_* | statistical/mean.rs |
| spmv.rs, spmm.rs | all | linalg/matmul.rs |
| transform.rs | transpose_* | linalg/matrix_transpose.rs |
| transform.rs | permute_axes_* | manipulation/permute_dims.rs |
| transform.rs | reshape_* | manipulation/reshape.rs |
| convert.rs | all | data_type_functions/astype.rs |
| cleanup.rs | prune_* | utility/prune.rs |
| cleanup.rs | eliminate_zeros_* | utility/eliminate_zeros.rs |

## 迁移步骤

1. ✅ 创建新目录结构和 mod.rs 文件
2. ✅ 创建重导出桥接层（新模块暂时 re-export 旧模块）
3. ✅ 更新 lib.rs 使用新路径导出
4. 🔄 下一步：将实际代码从旧文件迁移到新文件（待执行）
5. ⏳ 删除旧文件（arith.rs, reduce.rs, spmv.rs, spmm.rs, transform.rs, convert.rs, cleanup.rs）
6. ⏳ 验证编译通过

## 当前状态（过渡阶段）

### 已完成
- ✅ 新目录结构已建立
- ✅ 重导出桥接：新模块通过 `pub use crate::old_module::*` 暂时复用旧实现
- ✅ lib.rs 已更新为从新路径导出
- ✅ 公共API保持不变（Python绑定无需修改）

### 过渡架构
```
新公开模块 (elementwise/linalg/statistical/...)
    ↓ pub use
旧私有模块 (arith/reduce/spmv/spmm/transform/convert/cleanup)
    ↓ 实际实现
Rust 内核代码
```

### 下一步行动
将实际代码从旧文件迁移到新文件，每个新文件包含：
1. 该算子对所有格式的实现（CSR/CSC/COO/COOND）
2. 必要的私有辅助函数
3. 适当的文档注释对齐 Array API 语义

## 注意事项

- 保持辅助函数（如 add_row_count）与主函数在同一文件
- 共享的 utility 函数（i64_to_usize 等）提取到 util.rs
- 保持函数签名不变，确保 Python 绑定层无需修改
- 每个新文件都包含必要的 imports 和 allow 声明
