# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for DeepSeek-OCR's multimodal preprocessing kwargs."""

import math

import pytest

from vllm.multimodal import MULTIMODAL_REGISTRY

from ....conftest import ImageTestAssets
from ...utils import build_model_context


def calculate_expected_tokens(
    image_width: int,
    image_height: int,
    base_size: int,
    image_size: int,
    crop_mode: bool,
    min_crops: int = 2,
    max_crops: int = 6,
) -> int:
    """
    Calculate expected number of visual tokens for a given image configuration.
    
    Formula:
    - global_tokens = (base_size // 16 // 4) * ((base_size // 16 // 4) + 1) + 1
    - local_tokens = (crop_h * crop_w > 1) ? 
        (crop_h * h2) * (crop_w * w2 + 1) : 0
    - total = global_tokens + local_tokens
    
    where h2 = w2 = ceil((image_size // 16) / 4)
    """
    patch_size = 16
    downsample_ratio = 4
    
    # Determine crop ratio
    if crop_mode and (image_width > 640 or image_height > 640):
        # Calculate aspect ratios
        from vllm.transformers_utils.processors.deepseek_ocr import (
            calculate_aspect_ratios,
            find_closest_aspect_ratio,
        )
        aspect_ratio = image_width / image_height
        target_ratios = calculate_aspect_ratios(min_crops, max_crops)
        crop_ratio = find_closest_aspect_ratio(
            aspect_ratio, target_ratios, image_width, image_height, image_size
        )
        num_width_tiles, num_height_tiles = crop_ratio
    else:
        num_width_tiles, num_height_tiles = 1, 1
    
    # Calculate tokens
    h = w = math.ceil((base_size // patch_size) / downsample_ratio)
    h2 = w2 = math.ceil((image_size // patch_size) / downsample_ratio)
    
    global_views_tokens = h * (w + 1)
    if num_width_tiles > 1 or num_height_tiles > 1:
        local_views_tokens = (num_height_tiles * h2) * (num_width_tiles * w2 + 1)
    else:
        local_views_tokens = 0
    
    return global_views_tokens + local_views_tokens + 1


@pytest.mark.parametrize("model_id", ["deepseek-ai/DeepSeek-OCR"])
@pytest.mark.parametrize(
    ("mm_processor_kwargs", "expected_mode"),
    [
        # Tiny mode
        ({"base_size": 512, "image_size": 512, "crop_mode": False}, "tiny"),
        # Small mode
        ({"base_size": 640, "image_size": 640, "crop_mode": False}, "small"),
        # Base mode
        ({"base_size": 1024, "image_size": 1024, "crop_mode": False}, "base"),
        # Large mode
        ({"base_size": 1280, "image_size": 1280, "crop_mode": False}, "large"),
        # Gundam mode (default)
        ({"base_size": 1024, "image_size": 640, "crop_mode": True}, "gundam"),
        # Empty kwargs (should use defaults = Gundam)
        ({}, "gundam_default"),
    ],
)
@pytest.mark.parametrize("kwargs_on_init", [True, False])
def test_mode_configurations(
    image_assets: ImageTestAssets,
    model_id: str,
    mm_processor_kwargs: dict[str, int | bool],
    expected_mode: str,
    kwargs_on_init: bool,
):
    """Test all DeepSeek-OCR modes with proper token counting."""
    # Get configuration values (use defaults for empty kwargs)
    if mm_processor_kwargs:
        base_size = mm_processor_kwargs["base_size"]
        image_size = mm_processor_kwargs["image_size"]
        crop_mode = mm_processor_kwargs["crop_mode"]
    else:
        # Gundam defaults
        base_size = 1024
        image_size = 640
        crop_mode = True
    
    ctx = build_model_context(
        model_id,
        mm_processor_kwargs=mm_processor_kwargs if kwargs_on_init else None,
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    hf_processor_mm_kwargs = {} if kwargs_on_init else mm_processor_kwargs

    # Use a small test image
    test_image = image_assets[0].pil_image
    image_width, image_height = test_image.size
    
    prompt = "<image>"
    mm_data = {"image": [test_image]}

    processed_inputs = processor.apply(prompt, mm_data, hf_processor_mm_kwargs)

    # Calculate expected token count
    expected_tokens = calculate_expected_tokens(
        image_width, image_height, base_size, image_size, crop_mode
    )

    # Get image token from processor
    from vllm.transformers_utils.processors.deepseek_ocr import (
        DeepseekOCRProcessor,
    )
    hf_processor = ctx.get_hf_processor(DeepseekOCRProcessor, **hf_processor_mm_kwargs)
    image_token_id = hf_processor.image_token_id

    # Count image tokens
    img_tok_count = processed_inputs["prompt_token_ids"].count(image_token_id)
    
    assert img_tok_count == expected_tokens, (
        f"Mode: {expected_mode}, kwargs_on_init: {kwargs_on_init}, "
        f"Expected {expected_tokens} tokens, got {img_tok_count}"
    )


@pytest.mark.parametrize("model_id", ["deepseek-ai/DeepSeek-OCR"])
@pytest.mark.parametrize(
    "crop_mode",
    [True, False],
)
def test_crop_mode_impact(
    image_assets: ImageTestAssets,
    model_id: str,
    crop_mode: bool,
):
    """Test that crop mode produces different token counts for large images."""
    # Use a larger test image
    test_image = image_assets[0].pil_image
    # Resize to be larger than 640x640 to trigger cropping
    test_image = test_image.resize((800, 800))
    image_width, image_height = test_image.size
    
    mm_processor_kwargs = {
        "base_size": 1024,
        "image_size": 640,
        "crop_mode": crop_mode,
    }
    
    ctx = build_model_context(
        model_id,
        mm_processor_kwargs=mm_processor_kwargs,
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    prompt = "<image>"
    mm_data = {"image": [test_image]}

    processed_inputs = processor.apply(prompt, mm_data, {})

    # Calculate expected token count
    expected_tokens = calculate_expected_tokens(
        image_width, image_height, 1024, 640, crop_mode
    )

    # Get image token from processor
    from vllm.transformers_utils.processors.deepseek_ocr import (
        DeepseekOCRProcessor,
    )
    hf_processor = ctx.get_hf_processor(DeepseekOCRProcessor, **mm_processor_kwargs)
    image_token_id = hf_processor.image_token_id

    # Count image tokens
    img_tok_count = processed_inputs["prompt_token_ids"].count(image_token_id)
    
    assert img_tok_count == expected_tokens

    # With crop_mode=True and large image, we should get more tokens
    if crop_mode:
        # Calculate what we'd get without cropping
        no_crop_tokens = calculate_expected_tokens(
            image_width, image_height, 1024, 640, False
        )
        assert img_tok_count > no_crop_tokens, (
            "Crop mode should produce more tokens for large images"
        )


@pytest.mark.parametrize("model_id", ["deepseek-ai/DeepSeek-OCR"])
@pytest.mark.parametrize(
    ("min_crops", "max_crops"),
    [
        (1, 4),
        (2, 6),
        (2, 9),
    ],
)
def test_min_max_crops(
    image_assets: ImageTestAssets,
    model_id: str,
    min_crops: int,
    max_crops: int,
):
    """Test that min_crops and max_crops affect tiling for large images."""
    # Use a larger test image
    test_image = image_assets[0].pil_image
    test_image = test_image.resize((1000, 1000))
    image_width, image_height = test_image.size
    
    mm_processor_kwargs = {
        "base_size": 1024,
        "image_size": 640,
        "crop_mode": True,
        "min_crops": min_crops,
        "max_crops": max_crops,
    }
    
    ctx = build_model_context(
        model_id,
        mm_processor_kwargs=mm_processor_kwargs,
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    prompt = "<image>"
    mm_data = {"image": [test_image]}

    processed_inputs = processor.apply(prompt, mm_data, {})

    # Calculate expected token count
    expected_tokens = calculate_expected_tokens(
        image_width, image_height, 1024, 640, True, min_crops, max_crops
    )

    # Get image token from processor
    from vllm.transformers_utils.processors.deepseek_ocr import (
        DeepseekOCRProcessor,
    )
    hf_processor = ctx.get_hf_processor(DeepseekOCRProcessor, **mm_processor_kwargs)
    image_token_id = hf_processor.image_token_id

    # Count image tokens
    img_tok_count = processed_inputs["prompt_token_ids"].count(image_token_id)
    
    assert img_tok_count == expected_tokens


@pytest.mark.parametrize("model_id", ["deepseek-ai/DeepSeek-OCR"])
@pytest.mark.parametrize("num_images", [1, 2, 3])
def test_multiple_images(
    image_assets: ImageTestAssets,
    model_id: str,
    num_images: int,
):
    """Test processing batches of images with correct token counts."""
    mm_processor_kwargs = {
        "base_size": 1024,
        "image_size": 640,
        "crop_mode": True,
    }
    
    ctx = build_model_context(
        model_id,
        mm_processor_kwargs=mm_processor_kwargs,
        limit_mm_per_prompt={"image": num_images},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    # Use multiple images
    test_images = [image_assets[0].pil_image] * num_images
    prompt = "<image>" * num_images
    mm_data = {"image": test_images}

    processed_inputs = processor.apply(prompt, mm_data, {})

    # Calculate expected token count for one image
    test_image = test_images[0]
    image_width, image_height = test_image.size
    expected_tokens_per_image = calculate_expected_tokens(
        image_width, image_height, 1024, 640, True
    )
    expected_total_tokens = expected_tokens_per_image * num_images

    # Get image token from processor
    from vllm.transformers_utils.processors.deepseek_ocr import (
        DeepseekOCRProcessor,
    )
    hf_processor = ctx.get_hf_processor(DeepseekOCRProcessor, **mm_processor_kwargs)
    image_token_id = hf_processor.image_token_id

    # Count image tokens
    img_tok_count = processed_inputs["prompt_token_ids"].count(image_token_id)
    
    assert img_tok_count == expected_total_tokens


@pytest.mark.parametrize("model_id", ["deepseek-ai/DeepSeek-OCR"])
def test_different_sized_images(
    image_assets: ImageTestAssets,
    model_id: str,
):
    """Test processing images of different sizes in the same batch."""
    mm_processor_kwargs = {
        "base_size": 1024,
        "image_size": 640,
        "crop_mode": True,
    }
    
    ctx = build_model_context(
        model_id,
        mm_processor_kwargs=mm_processor_kwargs,
        limit_mm_per_prompt={"image": 2},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    # Use two images with different sizes
    small_image = image_assets[0].pil_image.resize((400, 400))
    large_image = image_assets[0].pil_image.resize((1200, 800))
    
    test_images = [small_image, large_image]
    prompt = "<image><image>"
    mm_data = {"image": test_images}

    processed_inputs = processor.apply(prompt, mm_data, {})

    # Calculate expected token count for each image
    expected_tokens_small = calculate_expected_tokens(400, 400, 1024, 640, True)
    expected_tokens_large = calculate_expected_tokens(1200, 800, 1024, 640, True)
    expected_total_tokens = expected_tokens_small + expected_tokens_large

    # Get image token from processor
    from vllm.transformers_utils.processors.deepseek_ocr import (
        DeepseekOCRProcessor,
    )
    hf_processor = ctx.get_hf_processor(DeepseekOCRProcessor, **mm_processor_kwargs)
    image_token_id = hf_processor.image_token_id

    # Count image tokens
    img_tok_count = processed_inputs["prompt_token_ids"].count(image_token_id)
    
    assert img_tok_count == expected_total_tokens


@pytest.mark.parametrize("model_id", ["deepseek-ai/DeepSeek-OCR"])
@pytest.mark.parametrize(
    ("mode_name", "base_size", "image_size", "expected_visual_tokens",
     "expected_total_tokens"),
    [
        # Native resolution modes as documented in DeepSeek-OCR README
        # https://github.com/deepseek-ai/DeepSeek-OCR#support-modes
        # 
        # The README documents "pure visual tokens" (h×h grid), but the actual
        # implementation includes structural tokens for sequence formatting:
        # - h×h visual feature tokens
        # - h newline/separator tokens (one per row)
        # - 1 end token
        # Total = h×(h+1) + 1
        #
        # For example, Tiny mode (512×512):
        # - h = ceil((512/16)/4) = 8
        # - Pure visual: 8×8 = 64 tokens (documented)
        # - Total: 8×(8+1) + 1 = 73 tokens (actual)
        ("tiny", 512, 512, 64, 73),    # 8×8 grid
        ("small", 640, 640, 100, 111),  # 10×10 grid
        ("base", 1024, 1024, 256, 273), # 16×16 grid
        ("large", 1280, 1280, 400, 421), # 20×20 grid
    ],
)
def test_native_resolution_token_counts(
    image_assets: ImageTestAssets,
    model_id: str,
    mode_name: str,
    base_size: int,
    image_size: int,
    expected_visual_tokens: int,
    expected_total_tokens: int,
):
    """
    Test native resolution modes to verify token counts match documentation.
    
    This test verifies both:
    1. The pure visual feature token count (h×h) as documented in README
    2. The actual total token count (h×(h+1)+1) including structural tokens
    
    The difference accounts for:
    - Newline/separator tokens added per row for 2D structure
    - An end-of-image token
    """
    mm_processor_kwargs = {
        "base_size": base_size,
        "image_size": image_size,
        "crop_mode": False,  # Native resolution, no cropping
    }
    
    ctx = build_model_context(
        model_id,
        mm_processor_kwargs=mm_processor_kwargs,
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    # Create an image at exactly the native resolution
    from PIL import Image
    test_image = Image.new("RGB", (image_size, image_size), color="red")
    
    prompt = "<image>"
    mm_data = {"image": [test_image]}

    processed_inputs = processor.apply(prompt, mm_data, {})

    # Get image token from processor
    from vllm.transformers_utils.processors.deepseek_ocr import (
        DeepseekOCRProcessor,
    )
    hf_processor = ctx.get_hf_processor(DeepseekOCRProcessor, **mm_processor_kwargs)
    image_token_id = hf_processor.image_token_id

    # Count image tokens
    actual_tokens = processed_inputs["prompt_token_ids"].count(image_token_id)
    
    # Verify the actual token count matches expected total
    assert actual_tokens == expected_total_tokens, (
        f"Mode: {mode_name}, Expected total tokens: {expected_total_tokens}, "
        f"Got: {actual_tokens}"
    )
    
    # Calculate h to verify the relationship
    h = math.ceil((base_size // 16) / 4)
    pure_visual = h * h
    total_with_structure = h * (h + 1) + 1
    
    # Verify the documented pure visual token count
    assert pure_visual == expected_visual_tokens, (
        f"Mode: {mode_name}, Pure visual tokens mismatch. "
        f"Expected: {expected_visual_tokens}, Got: {pure_visual}"
    )
    
    # Verify the relationship between pure visual and total tokens
    assert total_with_structure == expected_total_tokens, (
        f"Mode: {mode_name}, Token count formula mismatch. "
        f"h={h}, h×(h+1)+1={total_with_structure}, Expected: {expected_total_tokens}"
    )

