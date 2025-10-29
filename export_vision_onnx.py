import argparse
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer


def build_patches_and_grid(pixel_values, temporal_patch_size, patch_size, merge_size):
    assert pixel_values.dim() == 4, "pixel_values 必须是 (N, C, H, W)"
    N, C, H, W = pixel_values.shape
    if H % patch_size != 0 or W % patch_size != 0:
        raise ValueError(f"H({H}) 与 W({W}) 必须能被 patch_size({patch_size}) 整除")
    if (H // patch_size) % merge_size != 0 or (W // patch_size) % merge_size != 0:
        raise ValueError(
            f"(H/patch_size, W/patch_size)=({H//patch_size},{W//patch_size}) 必须能被 merge_size({merge_size}) 整除"
        )
    if N == 1:
        pixel_values = pixel_values.repeat(temporal_patch_size, 1, 1, 1)
    elif N % temporal_patch_size != 0:
        repeat_time = temporal_patch_size - (N % temporal_patch_size)
        repeat_image = pixel_values[-1:, ...].repeat(repeat_time, 1, 1, 1)
        pixel_values = torch.cat((pixel_values, repeat_image), dim=0)

    grid_t = pixel_values.shape[0] // temporal_patch_size
    grid_h = H // patch_size
    grid_w = W // patch_size

    patches = pixel_values.reshape(
        grid_t,
        temporal_patch_size,
        C,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flatten_patches = patches.reshape(
        grid_t * grid_h * grid_w, C * temporal_patch_size * patch_size * patch_size
    )
    grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.int32, device=flatten_patches.device)
    return flatten_patches, grid_thw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='模型路径')
    parser.add_argument('--batch', type=int, default=1, required=False, help='batch size')
    parser.add_argument('--height', type=int, default=476, required=False, help='图像高度')
    parser.add_argument('--width', type=int, default=476, required=False, help='图像宽度')
    parser.add_argument('--savepath', type=str, default='vision_encoder.onnx', required=False, help='保存路径')
    args = parser.parse_args()

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager",
    ).eval()
    _ = AutoTokenizer.from_pretrained(args.path, trust_remote_code=True, use_fast=False)

    vcfg = model.visual.config
    merge_size = int(vcfg.spatial_merge_size)
    patch_size = int(vcfg.patch_size)
    temporal_patch_size = int(vcfg.temporal_patch_size)

    # 构造输入
    N, C, H, W = int(args.batch), 3, int(args.height), int(args.width)
    pixel_values = torch.randn(N, C, H, W, dtype=torch.float32)

    with torch.no_grad():
        fp, gthw = build_patches_and_grid(pixel_values, temporal_patch_size, patch_size, merge_size)
        vision_features = model.visual(fp, gthw)
        print(f"视觉特征维度: {vision_features.shape}")
        print(f"视觉token数量: {vision_features.shape[0]}")

    def top_forward(pixel_values_in):
        fp, gthw = build_patches_and_grid(pixel_values_in, temporal_patch_size, patch_size, merge_size)
        return model.visual(fp, gthw)

    model.forward = top_forward

    torch.onnx.export(
        model,
        (pixel_values,),
        args.savepath,
        opset_version=17,
        input_names=["pixel_values"],
        output_names=["vision_features"],
    )


if __name__ == '__main__':
    main()


