function [I,am,pm] = generateImage3(lambda,I_size,Rmin,Rmax,compute_img)
    [X,Y] = meshgrid(-I_size*1.2/2+1:I_size*1.2/2);

    I = true(I_size,I_size);
    n = poissrnd(lambda*size(X,1)*size(X,2));
    
    Am = zeros(1,n);
    Pm = zeros(1,n);

    parfor k=1:n
        pos = rand(1,2)*I_size*1.2-I_size*.6;
        R = rand(1)*(Rmax-Rmin)+Rmin;
        Rs = (.15+rand(1)*.22)*R;
        theta = rand(1)*pi;
        if(compute_img)
            I_tmp = (X-pos(1)).^2 + (Y-pos(2)).^2 > R^2 | (X-pos(1)-R/2*cos(theta)).^2 + (Y-pos(2)-R/2*sin(theta)).^2 < Rs^2 | (X-pos(1)+R/2*cos(theta)).^2 + (Y-pos(2)+R/2*sin(theta)).^2 < Rs^2;
            I = I.*I_tmp(ceil(.1*I_size):ceil(.1*I_size)+I_size-1,ceil(.1*I_size):ceil(.1*I_size)+I_size-1);
        end
        Am(k) = pi*(R^2-2*Rs^2);
        Pm(k) =  2*pi*(R+2*Rs);
    end
    am = mean(Am);
    pm = mean(Pm);
    I = ~I;
end