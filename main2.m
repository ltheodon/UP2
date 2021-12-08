clear all
close all


nmean = 300;
lambda = nmean/(1024*1024);
I_size = 1024;
Rmin = 15;
Rmax = 40;


% [I,am,pm] = generateImage(lambda,I_size,Rmin,Rmax);
% [I,am,pm] = generateImage2(lambda,I_size,Rmin,Rmax);
[I,am,pm] = generateImage3(lambda,I_size,Rmin,Rmax,true);


figure
imshow(I,[]);


n_img = 200;
As = zeros(1,n_img);
Ps = zeros(1,n_img);
Es = zeros(1,n_img);
as = zeros(1,n_img);
ps = zeros(1,n_img);



parfor k=1:n_img
%     [I,am,pm] = generateImage3(lambda,I_size,Rmin,Rmax,true);
    I = imread(strcat('pix300/I_300_',num2str(k),'.png'));
    As(k) = bwarea(I)/(I_size^2);
    Ps(k) = bwarea(bwperim(I,4))/(I_size^2);
    Es(k) = bweuler(I,8)/(I_size^2);
%     as(k) = am;
%     ps(k) = pm;
%     imwrite(I,strcat('I_',num2str(nmean),'_',num2str(k),'.png'));
end

A = mean(As)
P = mean(Ps)
E = mean(Es)
std(Es)

am = 2.176886538804468e+03;
pm = 2.629058124671639e+02;



options = optimset('MaxFunEvals',100000,'MaxIter',100000,'TolFun',1e-5);
fun = @(x) 100*abs( pi*E - (1-A) * (-x*pi -  ( 1/2*P / (1-A) )^2) );
x0 = [1];
x3 = fminsearch(fun,x0,options)

options = optimset('MaxFunEvals',100000,'MaxIter',100000,'TolFun',1e-5);
fun = @(x) abs( P - (1-A)*x*x3 );
x0 = [1];
x2 = fminsearch(fun,x0,options)

options = optimset('MaxFunEvals',100000,'MaxIter',100000,'TolFun',1e-5);
fun = @(x) abs( A-1+exp(-x*x3) );
x0 = [1];
x1 = fminsearch(fun,x0,options)


abs(am-x1)/am*100
abs(pm-x2)/pm*100
abs(lambda-x3)/lambda*100



options = optimset('MaxFunEvals',100000,'MaxIter',100000,'TolFun',1e-5);
fun = @(x) 100*abs( pi*E - (1-A) * (x*x3*pi -  ( 1/2*P / (1-A) )^2) );
x0 = [1];
x4 = fminsearch(fun,x0,options)


x2/2/pi
