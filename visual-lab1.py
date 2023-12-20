import numpy as np
import  matplotlib.pyplot as plt
import tkinter as tk
import  cv2


def Methodsad():
    global method
    method='SAD'
def Methodncc():
    global method
    method='NCC'
def read_pics(lr):
    global image_l
    global image_r
    image_l = cv2.imread(lr+'/left.jpg',0)
    image_r = cv2.imread(lr+'/right.jpg',0)

def stopapp():
    global lr
    lr=text.get()
    app.destroy()    
##########################创建用户交互界面################################
app=tk.Tk()
app.title("双目立体匹配与重建")
app.geometry() 
text=tk.Entry(app)
lb1=tk.Label(app,text="输入文件夹路径，具体到数字")
text.grid(row=3,column=1)
lb1.grid(row=0,column=1)

lb2=tk.Label(app,text="选择立体匹配算法")
lb2.grid(row=4,column=1)
bt1=tk.Button(text='SAD',width=20,bg='pink',command=Methodsad)
bt2=tk.Button(text='NCC',width=20,bg='green',command=Methodncc)
bt1.grid(row=6,column=1)
bt2.grid(row=7,column=1)
bt3=tk.Button(text='确定',width=20,bg='yellow',command=stopapp)
bt3.grid(row=70,column=1)
tk.mainloop()



read_pics(lr)

'''image_l = cv2.imread('C:/Users/Asakiyaa/Desktop/stereo-rectify/2/left.jpg',0)
image_r = cv2.imread('cd/right.jpg',0)'''

'''cv2.imshow('Left Image', image_l)
cv2.imshow('Right Image', image_r)
cv2.waitKey(0)'''


with open(lr+'/para.txt','r')as f:
    
    lines=f.readlines()
    i=0
    while i < len(lines):
        line=lines[i]
        if line.find('fx')!=-1:
            key,value=line.strip().split(':')
            fx=float(value)
        if line.find('fy')!=-1:
            key,value=line.strip().split(':')
            fy=float(value)
        if line.find('cx')!=-1:
            key,value=line.strip().split(':')
            cx=float(value)
        if line.find('cy')!=-1:
            key,value=line.strip().split(':')
            cy=float(value)
        if line.find('left RT:')!=-1 or line.find('Left RT:')!=-1:
            
            left_rt=[]
            for j in range(1,4):
                left_rt.append(lines[i+j].strip().split())
            i=i+3
        if line.find('right RT:')!=-1 or line.find('Right RT:')!=-1:
            
            right_rt=[]
            for j in range(1,4):
                right_rt.append(lines[i+j].strip().split())
            i=i+3
        i=i+1
left_rt=np.array(left_rt) 
right_rt=np.array(right_rt) 
# 提取左右相机的旋转矩阵和平移向量
R_lw = np.array(left_rt[:, :3], dtype=np.float64)
T_lw = np.array(left_rt[:, 3], dtype=np.float64)

R_rw = np.array(right_rt[:, :3], dtype=np.float64)
T_rw = np.array(right_rt[:, 3], dtype=np.float64)


#计算左右相机之间的旋转矩阵和平移向量
R_rl = R_rw @ R_lw.T
T_rl = T_rw - R_rl@T_lw



# 定义相机内部参数矩阵和畸变系数
camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]], dtype=np.float64)
distortion_coefficients = np.zeros(5, dtype=np.float64)

# 计算矫正映射
image_size = (image_l.shape[1], image_l.shape[0])
R1, R2, P1, P2, Q, valid_roi1, valid_roi2 = cv2.stereoRectify(
    camera_matrix, distortion_coefficients,
    camera_matrix, distortion_coefficients,
    image_size, R_rl, T_rl)

map1_left, map2_left = cv2.initUndistortRectifyMap(camera_matrix, distortion_coefficients, R1, P1, image_size, cv2.CV_16SC2)
map1_right, map2_right = cv2.initUndistortRectifyMap(camera_matrix, distortion_coefficients, R2, P2, image_size, cv2.CV_16SC2)

# 应用矫正映射
left_rectified = cv2.remap(image_l, map1_left, map2_left, cv2.INTER_AREA)
right_rectified = cv2.remap(image_r, map1_right, map2_right, cv2.INTER_AREA)


import time
# 记录开始时间
start_time = time.time()

#########################立体匹配##############################
def matching(method,left_wid,right_wid):
    if method=='SAD':
        return compute_sad(left_wid,right_wid)
    if method=='NCC':
        return compute_ncc(left_wid,right_wid)
        
def compute_sad(window1, window2):
    return np.sum(np.abs(window1 - window2))

def compute_ncc(window1, window2):
    mean1 = np.mean(window1)
    mean2 = np.mean(window2)
    numerator = np.sum((window1 - mean1) * (window2 - mean2))
    denominator = np.sqrt(np.sum((window1 - mean1)**2) * np.sum((window2 - mean2)**2)) 
    return numerator / denominator if denominator != 0 else 0    

if method=='SAD':
    dbest = {}
    wx_size=2
    wy_size=2
    dmax=3
else:
    dbest = {}
    wx_size=4
    wy_size=4
    dmax=6
# 计算焦距和基线距离
focal_length = fx  # 相机的焦距
baseline =T_rl[0]      # 两个摄像机之间的基线距离
for x in range(0,image_l.shape[1]):
    for y in range(0,image_l.shape[0]): 
        Sbest = float('inf')
        left_wid=left_rectified[y:y+wy_size,x:x+wx_size]
        for d in range(x,x+dmax):
            t = 0;
            right_wid=right_rectified[y:y+wy_size,d:d+wx_size]
            if d+wx_size>=image_l.shape[1]:
                continue
            t=matching(method,left_wid,right_wid)
            if(method=='SAD' and t < Sbest):
                Sbest = t
                dbest[(y,x)] = d-x
            elif(method=='NCC' and -t< Sbest):
                Sbest = t
                dbest[(y,x)] = d-x
                

                        


# 将视差转换为深度
depth_map = np.zeros((image_l.shape[0], image_l.shape[1]))
for (y, x), disparity in dbest.items():
    depth_map[y, x] = focal_length * baseline / disparity if disparity != 0 else 0


# 归一化深度数据
depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
depth_norm = np.uint8(depth_norm)

end_time = time.time()

# 计算执行时间
execution_time = end_time - start_time

##画下矫正图
cv2.imshow('Rectified Left Image', left_rectified)
cv2.imshow('Rectified Right Image', right_rectified)
cv2.waitKey(0)
cv2.destroyAllWindows()
# 使用 Matplotlib 显示深度图
plt.imshow(depth_norm, cmap='gray')
plt.colorbar()  # 添加颜色条以便查看深度值
plt.title(f"Depth Map,execution_time:{execution_time}s")
plt.show()

