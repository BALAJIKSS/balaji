import numpy as np
import cv2
cap = cv2.VideoCapture('train3.mp4')
ret, old_frame = cap.read()


p = [0,0]
#p1 = [100,100]
H = np.zeros((2,2))
B = np.zeros((2,1))
#sq  = p1[0]*p1[0]+p1[1]*p1[1]
refPt = []
cropping = False

def click_and_crop(event, x, y, flags, param):
    global refPt, cropping
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
    elif event == cv2.EVENT_LBUTTONUP:
        #refPt.append((x, y))
        cropping = False
        cv2.circle(image,(refPt[0][0],refPt[0][1]),10,(255,0,0),-1)
        cv2.imshow("image", image)


image = old_frame.copy()
cv2.namedWindow('image')
cv2.setMouseCallback('image', click_and_crop)

while(1):
    cv2.imshow('image',image)
    k = cv2.waitKey(20) & 0xFF
    if k == ord('a'):        
        
        break
cv2.destroyAllWindows()
old_gray1 = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
old_gray = old_gray1[(refPt[0][0] - 50):(refPt[0][0] + 50), (refPt[0][1] - 50):(refPt[0][1] + 50)]
color = np.random.randint(0,255,(100,3))
#mask = np.zeros_like(old_frame)
r1,c1 = old_gray.shape
#old_gray = old_gray1[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
#old_color = old_frame[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]

point_track = [[refPt[0][0]],[refPt[0][1]]]

cv2.circle(old_frame,(int(point_track[0][0]),int(point_track[1][0])), 2, (0,0,255), -1)
while(1):
	
	p1 = [0.06,-0.05]
	ret,frame = cap.read()
	
	frame_gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame_gray = frame_gray1[(point_track[0][0] - 50):(point_track[0][0] + 50), (point_track[1][0] - 50):(point_track[1][0] + 50)]
	r1,c1 = old_gray.shape
	sq  = abs(p1[0])+abs(p1[1])
	#print r1,c1
	#frame_gray = frame_gray1[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
	#frame_color = frame[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
	sobelx = cv2.Sobel(frame_gray,cv2.CV_64F,1,0,ksize=5)
	sobely = cv2.Sobel(frame_gray,cv2.CV_64F,0,1,ksize=5)
	M = np.float32([[1,0,int(p[0])],[0,1,int(p[1])]])
	#print r1,c1
	dst = cv2.warpAffine(frame_gray,M,(c1,r1))
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
		
		
	
	# print sq
	# print "------------------"
	# print "point_track"
	# print point_track[0][0],point_track[1][0]

	# print "p1[0],p1[1]"
	#print p1[0],p1[1]
	point_track[0][0] = point_track[0][0] + p[0]
	point_track[1][0] = point_track[1][0] + p[1]
	#print point_track[0]
	cv2.circle(frame,(int(point_track[0][0]),int(point_track[1][0])), 10, (0,0,255), -1)
	res = cv2.resize(image,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
	res1 = cv2.resize(frame,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

	cv2.imshow('output',res1)
	cv2.imshow('image',res)
	old_gray = dst.copy()
		#cv2.imshow('input',old_frame)


	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break	

cv2.destroyAllWindows()
cap.release()
	

	

	
	
	
