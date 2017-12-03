function H = affine_rect(Iin,out_string)
#The below code performs affine rectification for a given image and output the affine rectified image
#H = Required Projective transformation for affine rectification
#Iin = imread of the image to be input
#out_string = desired filename of the output image
#locate the pair of parallel lines on the world plane....
figure(1)
Iin = imread('tiles.jpg');
imshow(Iin)
title('Select a pair of parallel lines from the image below by clicking any two points on each line')
xlabel('Press enter once done')
[x,y] = getpts
close Figure 1
# points retrived from the image
p1 = [x(1) y(1) 1];
p2 = [x(2) y(2) 1];
p3 = [x(3) y(3) 1];
p4 = [x(4) y(4) 1];
# 2 pairs of parallel lines obtained from image
l1 = cross(p1,p2);
l2 = cross(p3,p4);
l3 = cross(p2, p3);
l4 = cross(p1, p4);
# To find image line at infinity
a = cross(l1, l2);
a = a/a(1,3); 
b = cross(l3, l4);
b = b/ b(1,3); 
l = cross(a, b); 
# to compute the transformation matrix H
H = [1 0 0; 0 1 0; l(1, 1)/l(1,3) l(1, 2)/l(1,3) 1];
temp = maketform('projective',H')

# to compute the warped image using H.
Iout = imtransform(Iin, temp);
myout = imshow(Iout) 
saveas(myout,out_string)
end
