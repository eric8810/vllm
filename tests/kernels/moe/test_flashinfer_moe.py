# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from tests.kernels.moe.utils import make_test_weights
from tests.kernels.quantization.nvfp4_utils import (FLOAT4_E2M1_MAX,
                                                    FLOAT8_E4M3_MAX,
                                                    dequantize_nvfp4_to_dtype)
from tests.kernels.utils import torch_moe
from vllm import _custom_ops as ops
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.flashinfer_cutlass_moe import (
    FlashInferExperts, is_valid_flashinfer_cutlass_fused_moe)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    is_valid_flashinfer_cutlass_fused_moe_blockwise)
from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEModularKernel)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoEP)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_cutlass_fused_moe

if not has_flashinfer_cutlass_fused_moe(
) or not current_platform.has_device_capability(100):
    pytest.skip("Requires flashinfer_cutlass_fused_moe and nvfp4 support",
                allow_module_level=True)

MNK_FACTORS = [
    (2, 1024, 1024),
    (2, 1024, 1536),
    (2, 3072, 1024),
    (2, 3072, 1536),
    (64, 1024, 1024),
    (64, 1024, 1536),
    (64, 3072, 1024),
    (64, 2048, 1536),
    (224, 1024, 1024),
    (224, 1024, 1536),
]


@pytest.mark.parametrize("m,n,k", MNK_FACTORS)
@pytest.mark.parametrize("e", [40, 64, 256])
#@pytest.mark.parametrize("e", [128, 256])
@pytest.mark.parametrize("topk", [1, 6, 8])
@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
@torch.inference_mode()
def test_flashinfer_fp4_moe_no_graph(m: int, n: int, k: int, e: int, topk: int,
                                     dtype: torch.dtype):
    current_platform.seed_everything(7)
    with set_current_vllm_config(
            VllmConfig(parallel_config=ParallelConfig(
                pipeline_parallel_size=1))):

        a = torch.randn((m, k), device="cuda", dtype=dtype) / 10

        quant_blocksize = 16

        (_, w1_q, w1_blockscale,
         w1_gs), (_, w2_q, w2_blockscale, w2_gs) = make_test_weights(
             e,
             n,
             k,
             in_dtype=dtype,
             quant_dtype="nvfp4",
             block_shape=None,  # use quant_blocksize?
             per_act_token_quant=False,
         )

        score = torch.randn((m, e), device="cuda", dtype=dtype)
        topk_weights, topk_ids, _ = fused_topk(a,
                                               score,
                                               topk,
                                               renormalize=False)

        a1_gs = torch.ones((e, ), device="cuda", dtype=torch.float32)
        a2_gs = torch.ones((e, ), device="cuda", dtype=torch.float32)

        assert is_valid_flashinfer_cutlass_fused_moe(a, w1_q, w2_q)

        assert w1_gs is not None
        assert w2_gs is not None
        assert w1_blockscale is not None
        assert w2_blockscale is not None

        flashinfer_experts = FusedMoEModularKernel(
            MoEPrepareAndFinalizeNoEP(),
            FlashInferExperts(
                a1_gscale=a1_gs,
                g1_alphas=(1 / w1_gs),
                a2_gscale=a2_gs,
                g2_alphas=(1 / w2_gs),
                out_dtype=dtype,
                quant_dtype="nvfp4",
            ))

        flashinfer_output = flashinfer_experts(
            hidden_states=a,
            w1=w1_q,
            w1_scale=w1_blockscale,
            w2=w2_q,
            w2_scale=w2_blockscale,
            a1_scale=a1_gs,
            a2_scale=a2_gs,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )

        # Reference check:
        a_global_scale = ((FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) /
                          torch.amax(a.flatten(), dim=-1)).to(torch.float32)
        a_fp4, a_scale_interleaved = ops.scaled_fp4_quant(a, a_global_scale)
        _, m_k = a_fp4.shape
        a_in_dtype = dequantize_nvfp4_to_dtype(a_fp4,
                                               a_scale_interleaved,
                                               a_global_scale,
                                               dtype=a.dtype,
                                               device=a.device,
                                               block_size=quant_blocksize)

        w1_d = torch.empty((e, 2 * n, k), device="cuda", dtype=dtype)
        w2_d = torch.empty((e, k, n), device="cuda", dtype=dtype)

        for idx in range(0, e):
            w1_d[idx] = dequantize_nvfp4_to_dtype(w1_q[idx],
                                                  w1_blockscale[idx],
                                                  w1_gs[idx],
                                                  dtype=dtype,
                                                  device=w1_q.device,
                                                  block_size=quant_blocksize)
            w2_d[idx] = dequantize_nvfp4_to_dtype(w2_q[idx],
                                                  w2_blockscale[idx],
                                                  w2_gs[idx],
                                                  dtype=dtype,
                                                  device=w2_q.device,
                                                  block_size=quant_blocksize)

        torch_output = torch_moe(a_in_dtype, w1_d, w2_d, score, topk)

        torch.testing.assert_close(torch_output,
                                   flashinfer_output,
                                   atol=1e-1,
                                   rtol=1e-1)


@pytest.mark.skipif(not current_platform.is_device_capability(100), 
                    reason="Requires B200 GPU (SM100) for block-wise FP8")
@pytest.mark.parametrize("m,n,k", [(64, 1024, 1024), (224, 1024, 1536)])
@pytest.mark.parametrize("e", [64, 128])
@pytest.mark.parametrize("topk", [1, 4])
@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
@torch.inference_mode()
def test_flashinfer_cutlass_block_fp8(m: int, n: int, k: int, e: int, 
                                       topk: int, dtype: torch.dtype):
    """Test block-wise FP8 quantization for FlashInfer CUTLASS MoE"""
    current_platform.seed_everything(42)
    
    # Block shape for B200
    block_shape = (128, 128)
    
    # Create test input
    hidden_states = torch.randn(m, k, dtype=dtype, device="cuda")
    
    # Create block-quantized weights (placeholder - would need actual quantization)
    w1 = torch.randint(-128, 127, (e, 2*n, k), dtype=torch.uint8, device="cuda")
    w2 = torch.randint(-128, 127, (e, k, n), dtype=torch.uint8, device="cuda")
    
    # Create block-wise scales
    block_m, block_k = block_shape
    w1_scale_shape = (e, (2*n + block_m - 1) // block_m, 
                      (k + block_k - 1) // block_k)
    w2_scale_shape = (e, (k + block_m - 1) // block_m, 
                      (n + block_k - 1) // block_k)
    
    w1_scale = torch.rand(w1_scale_shape, dtype=torch.float32, device="cuda") * 0.1
    w2_scale = torch.rand(w2_scale_shape, dtype=torch.float32, device="cuda") * 0.1
    
    # Test block-wise validity check
    assert is_valid_flashinfer_cutlass_fused_moe_blockwise(
        hidden_states, w1, w2, block_shape=block_shape
    ), "Block-wise FlashInfer CUTLASS should be supported on B200"
    
    # Test invalid block shapes
    invalid_block_shape = (64, 64)
    assert not is_valid_flashinfer_cutlass_fused_moe_blockwise(
        hidden_states, w1, w2, block_shape=invalid_block_shape
    ), "Invalid block shape should not be supported"
    
    # Test FlashInferExperts with block shape
    g1_alphas = torch.ones(e, dtype=torch.float32, device="cuda") * 0.1
    g2_alphas = torch.ones(e, dtype=torch.float32, device="cuda") * 0.1
    a1_gscale = torch.tensor(1.0, dtype=torch.float32, device="cuda")
    a2_gscale = torch.tensor(1.0, dtype=torch.float32, device="cuda")
    
    experts = FlashInferExperts(
        g1_alphas=g1_alphas,
        g2_alphas=g2_alphas,
        a1_gscale=a1_gscale,
        a2_gscale=a2_gscale,
        out_dtype=dtype,
        quant_dtype=torch.float8_e4m3fn,
        block_shape=block_shape,
    )
    
    assert experts.is_block_quantized, "Expert should be marked as block-quantized"
    assert experts.block_shape == block_shape, "Block shape should be preserved"
    
    # Test block scale preparation
    prepared_w1_scale = experts._prepare_block_scales(
        w1_scale[0], w1[0].shape, block_m, block_k
    )
    assert prepared_w1_scale is not None, "Block scales should be prepared"
    assert prepared_w1_scale.is_contiguous(), "Block scales should be contiguous"


if __name__ == "__main__":
    test_flashinfer_fp4_moe_no_graph((2, 1024, 1024), 40, 1, torch.half)
