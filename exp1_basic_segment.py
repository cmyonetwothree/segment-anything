import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

print("=== 实验1：基础点击分割 ===\n")

# 1. 加载模型
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"使用设备: {device}")

sam = sam_model_registry["vit_b"](checkpoint="checkpoints/sam_vit_b_01ec64.pth")
sam.to(device=device)
predictor = SamPredictor(sam)
print("✅ 模型加载完成\n")

# 2. 读取图像
image = cv2.imread("test_images/truck.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(f"图像尺寸: {image.shape}")

# 3. 设置图像
predictor.set_image(image)
print("✅ 图像编码完成\n")

# 4. 定义不同的点击位置进行实验
test_points = [
    {"name": "卡车中心", "point": [500, 375], "label": 1},
    {"name": "卡车左侧", "point": [300, 400], "label": 1},
    {"name": "背景天空", "point": [100, 100], "label": 1},
]

# 5. 对每个点进行分割
fig, axes = plt.subplots(1, len(test_points) + 1, figsize=(20, 5))

# 显示原图
axes[0].imshow(image)
axes[0].set_title('原始图像', fontsize=12)
axes[0].axis('off')

for idx, test in enumerate(test_points):
    print(f"实验 {idx+1}: 点击 {test['name']} - 坐标 {test['point']}")
    
    input_point = np.array([test['point']])
    input_label = np.array([test['label']])
    
    # 执行分割
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    
    # 选择最佳掩码
    best_mask = masks[np.argmax(scores)]
    print(f"  最高分数: {scores.max():.3f}")
    print(f"  分割区域面积: {best_mask.sum()} 像素\n")
    
    # 可视化
    axes[idx+1].imshow(image)
    axes[idx+1].imshow(best_mask, alpha=0.5, cmap='jet')
    axes[idx+1].scatter(test['point'][0], test['point'][1], 
                       color='red', s=200, marker='*', edgecolors='white', linewidths=2)
    axes[idx+1].set_title(f"{test['name']}\n分数: {scores.max():.3f}", fontsize=12)
    axes[idx+1].axis('off')

plt.tight_layout()
plt.savefig('results/exp1_basic_segment.png', dpi=150, bbox_inches='tight')
print("✅ 结果已保存到 results/exp1_basic_segment.png")
plt.show()