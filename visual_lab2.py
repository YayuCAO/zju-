import cv2
import numpy as np
import sys


##计算两个向量的汉明距离函数
def NORM_HAMMING(vec1,vec2):
    sum = 0
    for i in range(len(vec1)):
        sum = sum + bin(vec1[i]^vec2[i]).count('1')
    return sum

##计算两个向量的L2距离
def NORM_L2(vec1, vec2):
    return np.sqrt(np.sum(np.square(vec1 - vec2)))

##最优特征点匹配函数（手写）
##最优特征点匹配函数（手写）
def match(method,des1,des2):
    matches = []
    mem1 = []#记录已经被匹配的特征点
    mem2 = []#记录已经被匹配的特征点
    if(method == 'orb'):
        ##进行mxn次循环，寻找最优匹配
        for de1 in range(len(des1)):
            mindistance = sys.maxsize
            for de2 in range(len(des2)):
                #print(des1[de1])
                if(mindistance > NORM_HAMMING(des1[de1],des2[de2]) and de1 not in mem1 and de2 not in mem2):
                    mindistance = NORM_HAMMING(des1[de1],des2[de2])
                    match1 = de1
                    match2 = de2
            for detemp in range(len(des1)):
                if(mindistance > NORM_HAMMING(des1[detemp],des2[match2]) and detemp not in mem1):
                    match1 = detemp
                    mindistance = NORM_HAMMING(des1[detemp],des2[match2])
            #print(maxdistance)
            #print(match1,match2)
            mem1.append(match1)##将已经匹配的点记录，之后不会再用这些点进行匹配
            mem2.append(match2)
            match = cv2.DMatch(match1,match2,0,mindistance)##将匹配结果初始化为DMatch类型
            matches.append(match)
    else:
        for de1 in range(len(des1)):
            mindistance = sys.maxsize
            for de2 in range(len(des2)):
                # print(des1[de1])
                if (mindistance > NORM_L2(des1[de1], des2[de2]) and de1 not in mem1 and de2 not in mem2):
                    mindistance = NORM_L2(des1[de1], des2[de2])
                    match1 = de1
                    match2 = de2
            for detemp in range(len(des1)):
                if (mindistance > NORM_L2(des1[detemp], des2[match2]) and detemp not in mem1):
                    match1 = detemp
                    mindistance = NORM_L2(des1[detemp], des2[match2])
            # print(maxdistance)
            mem1.append(match1)##将已经匹配的点记录，之后不会再用这些点进行匹配
            mem2.append(match2)
            match = cv2.DMatch(match1, match2, 0, mindistance)
            matches.append(match)
    #print(type(matches))
    #print(len(matches))
    matches = sorted(matches, key = lambda x: x.distance)#对匹配结果进行排序
    return matches

    

class Image_Stitching():
    def __init__(self) :
        self.ratio=0.85
        self.min_match=10
        self.orb=cv2.ORB_create()
        self.sift=cv2.SIFT_create()
        self.smoothing_window_size=800

    def registration(self,img1,img2,arg3,method):
        if method=='sift':
            kp1, des1 = self.sift.detectAndCompute(img1, None)
            kp2, des2 = self.sift.detectAndCompute(img2, None)
        if method=='orb':
             # 寻找关键点
            kp1 = self.orb.detect(img1)
            kp2 = self.orb.detect(img2)

            # 计算描述符
            kp1, des1 = self.orb.compute(img1, kp1) # 计算哪张图片的用哪张图片的关键点。
            kp2, des2 = self.orb.compute(img2, kp2)

        #matcher = cv2.BFMatcher()
        #raw_matches = matcher.knnMatch(des1, des2, k=2)
        matches=match(method,des1,des2) 
        #good_points = []
        good_matches=[]
        for m in matches:
            good_matches.append([m])
        '''for m1, m2 in raw_matches:
            if m1.distance < self.ratio * m2.distance:
                good_points.append((m1.trainIdx, m1.queryIdx))
                good_matches.append([m1])'''
        #img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, flags=2)
        #cv2.imshow("matching image",img3)
        if len(matches) > self.min_match:
            image1_kp = np.float32(
                [kp1[m.queryIdx].pt for m in matches[:100]])
            image2_kp = np.float32(
                [kp2[m.trainIdx].pt for m in matches[:100]])
            H, status = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC,5.0) 
        img = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good_matches[:100],None,flags=2)
        
        cv2.imwrite(arg3+f"\\match_{method}.jpg",img)
       

        return H

    def create_mask(self,img1,img2,version):
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 +width_img2
        offset = int(self.smoothing_window_size / 2)
        barrier = img1.shape[1] - int(self.smoothing_window_size / 2)
        mask = np.zeros((height_panorama, width_panorama))
        if version== 'left_image':
            mask[:, barrier - offset:barrier + offset ] = np.tile(np.linspace(1, 0, 2 * offset ).T, (height_panorama, 1))
            mask[:, :barrier - offset] = 1
        else:
            mask[:, barrier - offset :barrier + offset ] = np.tile(np.linspace(0, 1, 2 * offset ).T, (height_panorama, 1))
            mask[:, barrier + offset:] = 1
        return cv2.merge([mask, mask, mask])

    def blending(self,img1,img2,arg3,method):
        H = self.registration(img1,img2,arg3,method)
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 +width_img2

        panorama1 = np.zeros((height_panorama, width_panorama, 3))
        mask1 = self.create_mask(img1,img2,version='left_image')
        panorama1[0:img1.shape[0], 0:img1.shape[1], :] = img1
        panorama1 *= mask1
        mask2 = self.create_mask(img1,img2,version='right_image')
        panorama2 = cv2.warpPerspective(img2, H, (width_panorama, height_panorama))*mask2
        result=panorama1+panorama2

        rows, cols = np.where(result[:, :, 0] != 0)
        min_row, max_row = min(rows), max(rows) + 1
        min_col, max_col = min(cols), max(cols) + 1
        final_result = result[min_row:max_row, min_col:max_col, :]
        return final_result
def main(argv1,argv2,arg3,method):
    img1 = cv2.imread(argv1)
    img2 = cv2.imread(argv2)
    final=Image_Stitching().blending(img1,img2,arg3,method)
    cv2.imwrite(arg3+f"/panorama_{method}.jpg", final) 
if __name__ == '__main__':
    try: 
        import os
        path = 'C:\\Users\\Asakiyaa\\Desktop\\feature'
        for _,dirs,_ in os.walk(path):
            for dir in dirs[:5]:
                for _,_,pics in os.walk(os.path.join(path,dir)):
                    pic1=pics[0]
                    pic2=pics[1]
                    main(os.path.join(path,dir,pic1),os.path.join(path,dir,pic2),os.path.join(path,dir),'sift')
                    break

        
    except IndexError:
        print ("Please input two source images: ")
        print ("For example: python Image_Stitching.py '/Users/linrl3/Desktop/picture/p1.jpg' '/Users/linrl3/Desktop/picture/p2.jpg'")
    

