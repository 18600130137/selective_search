#第一步：程序引用包
import cv2
import selectivesearch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#step1
image2="images/test2.png"
#用cv2读取图片
img = cv2.imread(image2,cv2.IMREAD_UNCHANGED)
#白底黑字图 改为黑底白字图
img=255-img
img_lbl, regions =selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=20)

print('start count===',len(regions))
#第二步：过滤掉冗余的窗口
#1）第一次过滤
candidates = []
for r in regions:
    # 重复的不要
    if r['rect'] in candidates:
        continue
    # 太小和太大的不要
    if r['size'] < 200 or r['size']>20000:
        continue
    x, y, w, h = r['rect']
    # 太不方的不要
    if w / h > 1.5 or h / w > 2:
        continue
    candidates.append((x,y,w,h))

##('len(candidates)', 34) 一次过滤后剩余34个窗
print ('len(candidates)',len(candidates))
#2)第二次过滤 大圈套小圈的目标 只保留大圈
num_array=[]
for i in candidates:
    if len(num_array)==0:
        num_array.append(i)
    else:
        content=False
        replace=-1
        index=0
        for j in num_array:
            ##新窗口在小圈 则滤除
            if i[0]>=j[0] and i[0]+i[2]<=j[0]+j[2] and i[1]>=j[1] and i[1]+i[3]<=j[1]+j[3]:
                content=True
                break
            ##新窗口不在小圈 而在老窗口外部 替换老窗口
            elif i[0]<=j[0] and i[0]+i[2]>=j[0]+j[2] and i[1]<=j[1] and i[1]+i[3]>=j[1]+j[3]:
                replace=index
                break
            index+=1
            if not content:
                if replace>=0:
                    num_array[replace]=i
                else:
                    num_array.append(i)
#窗口过滤完之后的数量
len=len(num_array)
#二次过滤后剩余10个窗
print('len====',len)
#3)对过滤完的窗口进行展示
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
ax.imshow(img)
for x, y, w, h in num_array:
    rect = mpatches.Rectangle(
    (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
    ax.add_patch(rect)
plt.show()