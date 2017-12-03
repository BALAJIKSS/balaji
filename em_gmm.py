##Below is the code for Background Subtraction 
## Number of Gaussian Distributions = 2
import numpy as np
import cv2
cap = cv2.VideoCapture('train1.webm')
ret, old_frame = cap.read()
gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
fgbg2 = cv2.BackgroundSubtractorMOG2()
fgbg = cv2.BackgroundSubtractorMOG()
r,c = gray.shape
#Initializing means
u1 = np.zeros((r,c))
u2 = np.zeros((r,c))
#Initializing covariances
z1 = np.ones((r,c))*125
z2 = np.ones((r,c))*125
#Initializing weights
w1 = np.ones((r,c))*0.1
w2 = np.ones((r,c))*0.1
alpha = 0.5
T =  np.ones((r,c))*0.7
#Gaussian Function
def mult_normal(x, u, sigma):
    power = x-u
    inv = sigma
    fin_pow = -0.5 * power*inv*power
    exp = np.exp(fin_pow)
    denom = np.power(2 * np.pi, 1) * np.sqrt((sigma))
    return exp/denom
while(1):
	ret,new_frame = cap.read()
	frame_gray1 = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
	fgmask = fgbg.apply(frame_gray1)
	fgmask2= fgbg.apply(frame_gray1)
	#Condition for background elements
	B1 = w1 - T 
	B2 = w2 - T 
	B1[B1 < 0] = 0
	B1[B1 > 0] = 1
	B2[B2 < 0] = 0
	B2[B2 > 0] = 1
	z1_update = z1
	z2_update = z2
	w1_update = w1
	w2_update = w2
	u1_update = u1
	u2_update = u2
	
	diff = frame_gray1-u1
	#Condition for pixel matching
	X1 = (np.multiply(diff,np.multiply(np.reciprocal(z1),diff))) - 2.5*z1
	X2 = (np.multiply(diff,np.multiply(np.reciprocal(z2),diff))) - 2.5*z2
	OR = np.logical_or(X1,X2)*1
	
	X1[X1 > 0] = 0
	X1[X1 < 0] = 1
	X2[X2 > 0] = 0
	X2[X2 < 0] = 1
	
	B1X1 = np.logical_and(B1,X1)
	B2X2 = np.logical_and(B2,X2)
	rho1 = 0.5*mult_normal(frame_gray1,w1,np.reciprocal(z1))*X1
	rho2 = 0.5*mult_normal(frame_gray1,w2,np.reciprocal(z2))*X2
	#weight updates for matched elements
	w1_update = (np.ones((r,c))-alpha*np.ones((r,c)))*w1*X1*OR + alpha*np.ones((r,c))*X1*OR 
	w2_update = (1-alpha)*w2*X2*OR + alpha*np.ones((r,c))*X2*OR 
	#Mean updates for matched elements 
	u1_update = (1-rho1)*w1*X1*OR + rho1*frame_gray1*X1*OR
	u2_update = (1-rho2)*w2*X2*OR + rho2*frame_gray1*X2*OR
	#Covariance update for matched elements
	z1_update = (1- rho1)*z1*X1*OR + rho1*(frame_gray1 - u1_update)*(frame_gray1 - u1_update)*X1*OR
	z2_update = (1- rho2)*z2*X2*OR + rho2*(frame_gray1 - u2_update)*(frame_gray1 - u2_update)*X2*OR
	#weight updates for unmatched elements
	w1_update[X1==0] = (1-alpha)*w1[X1==0]*OR[X1==0]
	w2_update[X1==0] = (1-alpha)*w2[X2==0]*OR[X1==0]
	
	Y = w1 - w2
	#Updates when no match found
	r2,c2 = OR.shape
	if len(OR[OR == 0]) == r2*c2:
		Y[Y > 0] = 0
		Y[Y < 0] = 1
		w1_update = Y*0.01*(np.ones((r,c))- OR)
		w2_update = (np.ones((r,c))-Y)*0.01*(np.ones((r,c))- OR)
		z1_update = Y*10000*(np.ones((r,c))-OR)
		z2_update = (np.ones((r,c))-Y)*10000*(np.ones((r,c))- OR)
		u1_update = frame_gray1*Y*(np.ones((r,c))- OR)
		u2_update = frame_gray1*(np.ones((r,c))-Y)*(np.ones((r,c))- OR)

	r3,c3 = B1X1.shape
	segment = 1-np.multiply((np.logical_or(B1X1,B2X2)),np.ones((r2,c2)))
	cv2.imshow('original',new_frame)
	cv2.imshow('segment',segment)
	w1 = w1_update
	w2 = w2_update
	u1 = u1_update
	u2 = u2_update
	z1 = z1_update
	z2 = z2_update
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break
cv2.destroyAllWindows()
cap.release()
