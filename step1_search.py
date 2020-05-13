#第一步：程序引用包
import cv2
import selectivesearch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


#第二步：执行搜索工具,展示搜索结果
image2="images/test2.png"
#用cv2读取图片
img = cv2.imread(image2,cv2.IMREAD_UNCHANGED)
#白底黑字图 改为黑底白字图
img=255-img


'''selectivesearch 调用selectivesearch函数 对图片目标进行搜索
#Parameters
----------
    im_orig : 类型ndarray
        Input image  
        输入图片
    scale : int
        Free parameter. Higher means larger clusters in felzenszwalb segmentation.
        自由参数。在felzenszwalb分割中，较高的聚类数意味着较大的聚类数。
    sigma : float
        Width of Gaussian kernel for felzenszwalb segmentation.
        用于felzenszwalb分割的高斯核宽度。
    min_size : int
        Minimum component size for felzenszwalb segmentation.
        felzenszwalb分割的最小分量大小
'''
img_lbl, regions =selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=20)
print (regions[0])
print (len(regions)) #共搜索到199个区域
# 接下来我们把窗口和图像打印出来，对它有个直观认识
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
ax.imshow(img)
for reg in regions:
    x, y, w, h = reg['rect']
    rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=1)
    ax.add_patch(rect)
plt.show()