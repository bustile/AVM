import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
#left
DIM_l=(960, 540)
K_l=np.array([[458.6237932194047, 0.0, 473.7432579225916], [0.0, 343.4643637550696, 212.55748347758842], [0.0, 0.0, 1.0]])
D_l=np.array([[-0.0070037510714203895], [-0.11355037400221224], [0.18342377175699548], [-0.10355929786043042]])
#right
DIM_r=(960, 540)
K_r=np.array([[459.4733442236293, 0.0, 491.52456352262396], [0.0, 342.7959315962958, 230.45360652624822], [0.0, 0.0, 1.0]])
D_r=np.array([[0.016043128750740356], [-0.16553599700734845], [0.20986303042805304], [-0.08455228312970094]])

def undistort(img, direction):
    h,w = img.shape[:2]
    #right
    if direction == 0:
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K_r, D_r, np.eye(3), K_r, DIM_r, cv2.CV_16SC2)
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    #left
    else:
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K_l, D_l, np.eye(3), K_l, DIM_l, cv2.CV_16SC2)
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img


def bird_eye_view(img, angle):
    if angle == 0:
        src = np.float32([[0, 540], [960, 540], [0, 0], [960, 0]])
        #dst = np.float32([[140, 250], [400, 250], [0, 0], [540, 0]])
        dst = np.float32([[180, 320], [360, 320], [180, 0], [360, 0]])
        M = cv2.getPerspectiveTransform(src, dst)
        warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H))
        #plt.imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB))
        #plt.show()
        return warped_img

    elif angle == 1:
        src = np.float32([[0, 540], [960, 540], [0, 0], [960, 0]])
        #dst = np.float32([[540, 960], [540, 0], [360, 640], [360, 320]])
        dst = np.float32([[540, 960], [540, 0], [360, 960], [360, 0]])
        #[360, 640], [540, 960], [360, 320], [540, 0]
        M = cv2.getPerspectiveTransform(src, dst)
        warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H))
        #plt.imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB))
        #plt.show()
        return warped_img

    elif angle == 2:
        src = np.float32([[0, 540], [960, 540], [0, 0], [960, 0]])
        #dst = np.float32([[140, 710], [400, 710], [0, 960], [540, 960]])
        dst = np.float32([[180, 800], [360, 800], [180, 960], [360, 960]])
        M = cv2.getPerspectiveTransform(src, dst)
        warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H))
        #plt.imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB))
        #plt.show()
        return warped_img

    else:
        src = np.float32([[150, 540], [810, 540], [0, 0], [960, 0]])
        #dst = np.float32([[0, 0], [0, 960], [180, 320], [180, 640]])
        dst = np.float32([[0, 0], [0, 960], [180, 0], [180, 960]])
        M = cv2.getPerspectiveTransform(src, dst)
        warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H))
        #plt.imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB))
        #plt.show()
        return warped_img

IMAGE_W = 540
IMAGE_H = 960

img1 = cv2.imread('./front_iamge.jpg') # Read the test img
img2 = cv2.imread('./right_image.jpg')
img3 = cv2.imread('./back_image.jpg')
img4 = cv2.imread('./left_image.jpg')

img1 = img1[0:540, 200:760]
img1 = cv2.resize(img1, (960,540))
img2 = undistort(img2, 0)
img2 = img2[150:540, 0:910]
img2 = cv2.resize(img2, (960, 540))
img3 = img3[200:540, 50:850]
img3 = cv2.resize(img3, (960, 540))
img4 = undistort(img4, 1)
#cv2.imshow('undistort', img4)
#cv2.imshow('undis', img2)
img4 = img4[150:540, 0:960]
img4 = cv2.resize(img4, (960,540))


#cv2.imwrite('./distort.jpg', img4)
#img1 = cv2.resize(img1, dsize=(540, 960), interpolation=cv2.INTER_AREA)
#img2 = cv2.resize(img2, dsize=(540, 960), interpolation=cv2.INTER_AREA)
#img3 = cv2.resize(img3, dsize=(540, 960), interpolation=cv2.INTER_AREA)
#img4 = cv2.resize(img4, dsize=(540, 960), interpolation=cv2.INTER_AREA)
#dst2 = cv2.resize(src, dsize=(0, 0), fx=0.3, fy=0.7, interpolation=cv2.INTER_LINEAR)


bev_img1 = bird_eye_view(img1, 0)
bev_img2 = bird_eye_view(img2, 1)
bev_img3 = bird_eye_view(img3, 2)
bev_img4 = bird_eye_view(img4, 3)
#cv2.imshow('bev',bev_img4)
add = bev_img1 + bev_img2 + bev_img3 + bev_img4
add = cv2.resize(add, (600,600))
#print(add)
plt.imshow(cv2.cvtColor(add, cv2.COLOR_BGR2RGB))  # Show results
plt.show()

#img = img[200:(200+IMAGE_H), 0:IMAGE_W]