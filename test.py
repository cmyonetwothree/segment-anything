import os

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, SamPredictor


def main() -> None:
    # 1. 选择设备
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 2. 模型与权重路径（沿用你在 exp1 中使用的配置）
    model_type = "vit_b"
    checkpoint_path = "checkpoints/sam_vit_b_01ec64.pth"

    if not os.path.exists(checkpoint_path):
        print(f"未找到模型权重文件: {checkpoint_path}")
        print("请先下载 sam_vit_b_01ec64.pth 到 checkpoints/ 目录后再运行。")
        return

    print("开始加载模型...")
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    print("✅ 模型加载完成\n")

    # 3. 读取测试图片（使用仓库自带的 dog 图片）
    image_path = "notebooks/images/dog.jpg"
    if not os.path.exists(image_path):
        print(f"未找到测试图片: {image_path}")
        return

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"读取图片: {image_path}, 尺寸: {image.shape}")

    # 4. 设置图像到预测器
    predictor.set_image(image)
    print("✅ 图像编码完成\n")

    # 5. 在图像中心点击一次，做一个最简单的测试
    h, w = image.shape[:2]
    input_point = np.array([[w // 2, h // 2]])
    input_label = np.array([1])  # 1 表示前景点

    print(f"在图像中心打点坐标: {input_point[0].tolist()}")
    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    # 6. 选取得分最高的掩码
    best_idx = np.argmax(scores)
    best_mask = masks[best_idx]
    best_score = float(scores.max())
    print(f"最佳掩码得分: {best_score:.3f}")
    print(f"掩码像素面积: {int(best_mask.sum())} 像素\n")

    # 7. 可视化并保存结果
    os.makedirs("results", exist_ok=True)
    save_path = "results/test_sam_dog.png"

    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.imshow(best_mask, alpha=0.5, cmap="jet")
    plt.scatter(
        input_point[0, 0],
        input_point[0, 1],
        color="red",
        s=200,
        marker="*",
        edgecolors="white",
        linewidths=2,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✅ 测试结果已保存到: {save_path}")


if __name__ == "__main__":
    main()

