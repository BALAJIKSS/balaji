function H = affine_rect(Iin,out_string)
figure(1)
Iin = imread('tiles.jpg');
imshow(Iin)
title('Select a pair of parallel lines from the image below by clicking any two points on each line')
xlabel('Press enter once done')
[x,y] = getpts
close Figure 1

p1 = [x(1) y(1) 1];
p2 = [x(2) y(2) 1];
p3 = [x(3) y(3) 1];
p4 = [x(4) y(4) 1];

l1 = cross(p1,p2);
l2 = cross(p3,p4);
l3 = cross(p2, p3);
l4 = cross(p1, p4);

a = cross(l1, l2);
a = a/a(1,3); 
b = cross(l3, l4);
b = b/ b(1,3); 
l = cross(a, b); 

H = [1 0 0; 0 1 0; l(1, 1)/l(1,3) l(1, 2)/l(1,3) 1];
temp = maketform('projective',H')


Iout = imtransform(Iin, temp);
myout = imshow(Iout) 
saveas(myout,out_string)
end