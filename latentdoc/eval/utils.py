import torch
import torch.nn.functional as F

def compute_similarity_matrix(input_tensor, block_size=1024):
    # 将32*32*768张量展平为1024*768
    flattened_tensor = input_tensor.view(-1, input_tensor.size(-1))
    num_vectors = flattened_tensor.size(0)

    # 初始化相似度矩阵
    similarity_matrix = torch.zeros(num_vectors, num_vectors)

    # 分块计算相似度矩阵
    for i in range(0, num_vectors, block_size):
        end_i = min(i + block_size, num_vectors)
        for j in range(0, num_vectors, block_size):
            end_j = min(j + block_size, num_vectors)

            # 计算当前块的相似度
            block_i = flattened_tensor[i:end_i]
            block_j = flattened_tensor[j:end_j]
            
            similarity_block = F.cosine_similarity(block_i.unsqueeze(1), block_j.unsqueeze(0), dim=-1)
            similarity_matrix[i:end_i, j:end_j] = similarity_block

    return similarity_matrix

def visualize_similarity_matrix(similarity_matrix):
    # 将相似度矩阵转换为numpy数组
    sim_matrix_np = similarity_matrix.detach().cpu().numpy()
    
    # 创建一个新的图像
    plt.figure(figsize=(10, 10))
    
    # 使用imshow显示相似度矩阵
    plt.imshow(sim_matrix_np, cmap='viridis')
    
    # 添加颜色条
    plt.colorbar()
    
    # 添加标题
    plt.title('Similarity Matrix')
    
    plt.savefig('similarity_matrix.png')

import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_heatmap(original_image , heatmap, patch_size=32, alpha=0.6):
    # 读取原始图像和热力图
    original_image = cv2.resize(original_image,(1024,1024), interpolation=cv2.INTER_NEAREST)
    
    # 扩展热力图
    def expand_heatmap(heatmap, patch_size, original_size):
        h, w = heatmap.shape
        new_h, new_w = h * patch_size, w * patch_size
        expanded_heatmap = np.zeros((new_h, new_w), dtype=np.float64)
        for i in range(h):
            for j in range(w):
                expanded_heatmap[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = heatmap[i, j]
        # expanded_heatmap_resized = cv2.resize(expanded_heatmap, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
        return expanded_heatmap
    
    # 将扩展后的热力图转换为伪彩色图像
    def heatmap_to_color(heatmap):
        return cv2.applyColorMap(heatmap, cv2.COLORMAP_BONE)
    
    # 融合热力图和原始图像
    def blend_images(heatmap_color, original_image, alpha):
        beta = 1 - alpha
        fused_image = cv2.addWeighted(heatmap_color, alpha, original_image, beta, 0)
        return fused_image
    
    # 原始图像的尺寸
    original_size = original_image.shape[:2]
    
    # 扩展热力图
    expanded_heatmap = expand_heatmap(heatmap, patch_size, original_size)*255
    expanded_heatmap=expanded_heatmap.astype(np.uint8)
    # 将扩展后的热力图转换为伪彩色图像
    heatmap_color = heatmap_to_color(expanded_heatmap)
    
    # 融合热力图和原始图像
    fused_image = blend_images(heatmap_color, original_image, alpha)
    
    # 显示结果
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(original_image)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    
    plt.title('Pseudocolor Heatmap')

    plt.imshow(heatmap_color, cmap='bone')
    plt.colorbar()
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title('Fused Image')
    # plt.colorbar()
    plt.imshow(fused_image)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('heatmap.png')


import os
import fnmatch

def find_test_json_files(root_path):
    """
    递归查找指定目录下所有名为"DocVQA_pred.json"的文件，并返回它们的绝对路径列表。
    
    :param root_path: 查找的根目录
    :return: 包含所有匹配文件绝对路径的列表
    """
    matches = []
    
    for root, dirnames, filenames in os.walk(root_path):
        for filename in fnmatch.filter(filenames, 'DocVQA_pred.json'):
            matches.append(os.path.join(root, filename))
    
    return matches

if __name__=="__main__":
    parser=initialize_argparse()
    args=parser.parse_args()
    model_path=args.model_path
    img_path=args.img_path
    output_name=args.output_name
    downsample_rate=args.downsample_rate
    resolution=args.resolution
    model=model_init(model_path,downsample_rate)
    resolution=(resolution,resolution)
    transform= transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            # transforms.Normalize(IMAGENET_DEFAULT_MEAN,IMAGENET_DEFAULT_STD)
        ]
    )
    criterion = nn.MSELoss().cuda()
    img=Image.open(img_path).convert('RGB')
    loss,input,recon,encode=infer_single_img(model,img,transform,criterion,resolution)
    similarity_matrix=compute_similarity_matrix(encode.permute(0,2,3,1).unsqueeze(0))
    visualize_similarity_matrix(similarity_matrix)
    ori_img=cv2.imread(img_path)
    ori_img=cv2.cvtColor(ori_img,cv2.COLOR_BGR2RGB)
    # visualize_similarity_heatmap(similarity_matrix[200].reshape(32,32))
    visualize_heatmap(ori_img,similarity_matrix[200].reshape(32,32))
    print(loss)
    plt.figure(figsize=(11, 10)) 
    plt.imshow(recon)
    plt.gray()
    plt.axis('off')
    plt.savefig(output_name)
    plt.figure(figsize=(11, 10)) 
    plt.imshow(input)
    plt.gray()
    plt.axis('off')
    plt.savefig('input.png')