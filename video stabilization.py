import numpy as np
import cv2

def MLK(img,template,max_iter=500,min_norm=0.01):
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY).astype(np.float32)
    template_gray = cv2.cvtColor(template,cv2.COLOR_RGB2GRAY).astype(np.float32)
    dx = cv2.Sobel(template_gray,cv2.CV_64F,1,0,ksize=3)
    dy = cv2.Sobel(template_gray,cv2.CV_64F,0,1,ksize=3)

    dt = np.empty((template_gray.shape[0],template_gray.shape[1],1,2))
    dt[:,:,0,0] = dx
    dt[:,:,0,1] = dy
    
    x,y=img_gray.shape[0],img_gray.shape[1]
    j = np.zeros((x,y,6,2))
    j[:,:,0,0] = np.arange(y).reshape(1,-1).repeat(x,axis=0)
    j[:,:,1,1] = np.arange(y).reshape(1,-1).repeat(x,axis=0)
    j[:,:,2,0] = np.arange(x).reshape(-1,1).repeat(y,axis=1)
    j[:,:,3,1] = np.arange(x).reshape(-1,1).repeat(y,axis=1)


    j[:,:,4,0] = np.ones((x,y))
    j[:,:,5,1] = np.ones((x,y))


    
    steep_d = np.einsum('ijkl,ijml->ijkm',dt,j)
    steep_d_T = np.rollaxis(steep_d,3,2)
    m = np.einsum('ijkl,ijlm->ijkm',steep_d_T,steep_d)
    h = np.sum(np.sum(m,axis=0),axis=0)

    h_inv = np.linalg.pinv(h)
    pv = np.array([[0,0,0],[0,0,0]],dtype=np.float32)
    p1 = np.array([[1,0,0],[0,1,0]],dtype=np.float32)
    pn = 1
    iter = 0

    while pn > min_norm and iter < max_iter:
        img_warp = cv2.warpAffine(img_gray,pv+p1,(img_gray.shape[1],img_gray.shape[0])).astype(np.float32)
        
        error_img = (img_warp - template_gray).reshape((img_gray.shape[0],img_gray.shape[1],1,1))
        summ_m = np.einsum('ijkl,ijlm->ijkm',steep_d_T,error_img)
        summ = np.sum(np.sum(summ_m,axis=0),axis=0)
        dp = np.dot(h_inv,summ)
        pv_copy = pv.copy()

        pv[0,0] = pv_copy[0,0] + dp[0] + pv_copy[0,0]*dp[0] + pv_copy[0,1]*dp[1]
        pv[1,1] = pv_copy[1,1] + dp[3] + pv_copy[1,0]*dp[2] + pv_copy[1,1]*dp[3]
        pv[0,2] = pv_copy[0,2] + dp[4] + pv_copy[0,0]*dp[4] + pv_copy[0,1]*dp[5]

        pv[1,0] = pv_copy[1,0] + dp[1] + pv_copy[1,0]*dp[0] + pv_copy[1,1]*dp[1]
        pv[1,2] = pv_copy[1,2] + dp[5] + pv_copy[1,0]*dp[4] + pv_copy[1,1]*dp[5]
        pv[0,1] = pv_copy[0,1] + dp[2] + pv_copy[0,0]*dp[2] + pv_copy[0,1]*dp[3]
        
        pn = np.linalg.norm(dp)
        iter = iter + 1
        
    img_warp = cv2.warpAffine(img_gray,pv+p1,(img_gray.shape[1],img_gray.shape[0]))
    return img_warp.astype(np.uint8)


if __name__ == "__main__":
    vid = cv2.VideoCapture('train3.mp4')
    length = int(vid.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    fps = int(vid.get(cv2.cv.CV_CAP_PROP_FPS))
    print(length," - ",fps)
    ret, frame = vid.read()

    template = frame
    template_gray = cv2.cvtColor(template,cv2.COLOR_RGB2GRAY)
    fourcc = cv2.cv.CV_FOURCC(*'XVID')


    out = cv2.VideoWriter("out.mp4",fourcc,fps,(template_gray.shape[1],template_gray.shape[0]))
    out.write(np.repeat(template_gray[...,None],3,axis=2))
    frame_count = 1
    while(1):
        ret, frame = vid.read()
        if not ret:
            break
        frame_warp = MLK(frame,template,100,0.001)
        
        frame_count = frame_count + 1
        out.write(np.repeat(frame_warp[...,None],3,axis=2))