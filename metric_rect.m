function H = metric_rect(Iin,out_string)
Iin = imread('tiles.jpg');
figure(1)
imshow(Iin)
title('Select 2 pairs of orthogonal lines from the image below by using mouse')
xlabel('Press enter once done')
[x,y] = getpts;
close Figure 1

l11 = x(1)
l12 = y(1)
l21 = x(3)
l22 = y(3)
m11 = x(2)
m12 = y(2)
m21 = x(4)
m22 = y(4)


M = [l11*m11 (l11*m12 + l12*m11) ; l21*m21 (l21*m22 + l22*m21)]
b = [-l12*m12;-l22*m22 ]

x = linsolve(M,b)

S = eye(2)
S(1,1) = x(1)
S(1,2) = x(2)
S(2,1) = x(2)

[U,D,V] = svd(S)
sqrtD = sqrt(D)
U_T = transpose(U)
A = U_T*sqrtD
A = A*V
H2 = eye(3)
H2(1,1) = A(1,1)
H2(1,2) = A(1,2)
H2(2,1) = A(2,1)
H2(2,2) = A(2,2)
if H2(1,1) < 0
    H2(1,1) = -H2(1,1)

elseif H2(2,2) < 0
    H2(2,2) = -H2(2,2)
end
H = H2'

temp = maketform('projective',H)
Iout = imtransform(Iin, temp);
myout = imshow(Iout)
saveas(myout,out_string)
end