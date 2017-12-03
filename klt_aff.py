# The below code performs point tracking.Implementation of Lucas-Kanade algorithm.
# This code is an implementation of "Lucas-Kanade 20 Years On: A Unifying Framework: Part 1" by Simon Baker and Iain Matthews
import numpy as np
import cv2
cap = cv2.VideoCapture('train3.mp4')
ret, old_frame = cap.read()
# Transformation array
p = [0,0]
# Initializing Hessian Matrix
H = np.zeros((2,2))
B = np.zeros((2,1))
refPt = []
cropping = False
# Funtion to get the image co-ordinates of the point chosen 
def click_and_crop(event, x, y, flags, param):
    global refPt, cropping
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
    elif event == cv2.EVENT_LBUTTONUP:
        cropping = False
        cv2.circle(image,(refPt[0][0],refPt[0][1]),10,(255,0,0),-1)
        cv2.imshow("image", image)
image = old_frame.copy()
cv2.namedWindow('image')
cv2.setMouseCallback('image', click_and_crop)
# Select the point to be tracked and press 'a'
while(1):
    cv2.imshow('image',image)
    k = cv2.waitKey(20) & 0xFF
    if k == ord('a'):        
        break
cv2.destroyAllWindows()
old_gray1 = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
# 100*100 template is chosen with the above point as the center
old_gray = old_gray1[(refPt[0][0] - 50):(refPt[0][0] + 50), (refPt[0][1] - 50):(refPt[0][1] + 50)]
color = np.random.randint(0,255,(100,3))
r1,c1 = old_gray.shape
#Point to be tracked
point_track = [[refPt[0][0]],[refPt[0][1]]]
# Circle is made on the same point 
cv2.circle(old_frame,(int(point_track[0][0]),int(point_track[1][0])), 2, (0,0,255), -1)
while(1):
	p1 = [0.06,-0.05]
	ret,frame = cap.read()
	frame_gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame_gray = frame_gray1[(point_track[0][0] - 50):(point_track[0][0] + 50), (point_track[1][0] - 50):(point_track[1][0] + 50)]
	r1,c1 = old_gray.shape
	#Norm of the array
	sq  = abs(p1[0])+abs(p1[1])
	#Gradient along x
	sobelx = cv2.Sobel(frame_gray,cv2.CV_64F,1,0,ksize=5)
	#Gradient along y
	sobely = cv2.Sobel(frame_gray,cv2.CV_64F,0,1,ksize=5)
	# Warp affine
	M = np.float32([[1,0,int(p[0])],[0,1,int(p[1])]])
	dst = cv2.warpAffine(frame_gray,M,(c1,r1))
	# Convergence of p1
	while(sq>0.01):
		M = np.float32([[1,0,int(p[0])],[0,1,int(p[1])]])
		dst = cv2.warpAffine(frame_gray,M,(c1,r1))
		diff = dst - old_gray
		sobel1x = cv2.warpAffine(sobelx,M,(c1,r1))
		sobel1y = cv2.warpAffine(sobely,M,(c1,r1))
		for i in xrange(dst.shape[0]):
			for j in xrange(dst.shape[1]):
				a = sobel1x[i][j]*sobel1x[i][j]
				b = sobel1x[i][j]*sobel1y[i][j]
				c = sobel1x[i][j]*sobel1y[i][j]
				d = sobel1y[i][j]*sobel1y[i][j]
				H = H + np.array([[a,b],[c,d]])
				a1 = sobel1x[i][j]
				b1 = sobel1y[i][j]
				B = B + np.array([[a1],[b1]])*diff[i][j]
		iH = np.linalg.inv(H)
		p1 = np.dot(iH,B)
		p[0] = p[0] + p1[0]*0.09
		p[1] = p[1] + p1[1]*0.09
		sq  = p1[0]*p1[0]+p1[1]*p1[1]
	# point_track update
	point_track[0][0] = point_track[0][0] + p[0]
	point_track[1][0] = point_track[1][0] + p[1]
	# Circle the tracked point
	cv2.circle(frame,(int(point_track[0][0]),int(point_track[1][0])), 10, (0,0,255), -1)
	# Resize of image
	res = cv2.resize(image,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
	res1 = cv2.resize(frame,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
	# Output of first frame and the tracked frame
	cv2.imshow('output',res1)
	cv2.imshow('image',res)
	old_gray = dst.copy()
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break	
cv2.destroyAllWindows()
cap.release()
	

	

	
	
	
