import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

print("=== 实验4：自动分割所有对象 ===\n")

device = "cpu"
print(f"使用设备: {device}")
sam = sam_model_registry["vit_b"](checkpoint="checkpoints/sam_vit_b_01ec64.pth")
sam.to(device=device)

# 创建自动掩码生成器（不同参数配置）
configs = [
    {
        "name": "快速模式",
        "params": {
            "points_per_side": 16,
            "pred_iou_thresh": 0.88,
            "stability_score_thresh": 0.95,
        }
    },
    {
        "name": "标准模式",
        "params": {
            "points_per_side": 32,
            "pred_iou_thresh": 0.86,
            "stability_score_thresh": 0.92,
        }
    },
    {
        "name": "精细模式",
        "params": {
            "points_per_side": 64,
            "pred_iou_thresh": 0.84,
            "stability_score_thresh": 0.90,
        }
    }
]

# 读取图像
image = cv2.imread("test_images/truck.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(1, len(configs) + 1, figsize=(20, 5))

# 原图
axes[0].imshow(image)
axes[0].set_title('原始图像', fontsize=12)
axes[0].axis('off')

for idx, config in enumerate(configs):
    print(f"\n实验 {idx+1}: {config['name']}")
    
    # 创建生成器
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        **config['params']
    )
    
    # 生成掩码
    print("  正在生成掩码...")
    import time
    start = time.time()
    masks = mask_generator.generate(image)
    elapsed = time.time() - start
    
    print(f"  检测到 {len(masks)} 个对象")
    print(f"  耗时: {elapsed:.2f}秒")
    
    # 可视化
    def show_anns(anns, ax):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        
        ax.imshow(image)
        img = np.ones((image.shape[0], image.shape[1], 4))
        img[:,:,3] = 0
        
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.5]])
            img[m] = color_mask
        
        ax.imshow(img)
    
    show_anns(masks, axes[idx+1])
    axes[idx+1].set_title(f"{config['name']}\n{len(masks)}个对象, {elapsed:.1f}秒", 
                         fontsize=12)
    axes[idx+1].axis('off')

plt.tight_layout()
plt.savefig('results/exp4_auto_segment.png', dpi=150, bbox_inches='tight')
print("\n✅ 结果已保存到 results/exp4_auto_segment.png")
plt.show()