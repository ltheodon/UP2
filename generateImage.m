function [I,am,pm] = generateImage(lambda,I_size,Rmin,Rmax)
    [X,Y] = meshgrid(-I_size*1.2/2+1:I_size*1.2/2);

    I = true(I_size,I_size);
    n = poissrnd(lambda*size(X,1)*size(X,2));
    
    Rm = zeros(1,n);

    parfor k=1:n
        pos = rand(1,2)*I_size*1.2-I_size*.6;
        R = rand(1)*(Rmax-Rmin)+Rmin;
        I_tmp = (X-pos(1)).^2 + (Y-pos(2)).^2 > R^2;
        I = I.*I_tmp(ceil(.1*I_size):ceil(.1*I_size)+I_size-1,ceil(.1*I_size):ceil(.1*I_size)+I_size-1);
        Rm(k) = R;
    end
    am = pi*mean(Rm.^2);
    pm = 2*pi*mean(Rm);
    I = ~I;
end