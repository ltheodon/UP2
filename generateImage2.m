function [I,am,pm] = generateImage2(lambda,I_size,Rmin,Rmax)
    [X,Y] = meshgrid(-I_size*1.2/2+1:I_size*1.2/2);

    I = true(I_size,I_size);
    n = poissrnd(lambda*size(X,1)*size(X,2));
    
    Rm = zeros(1,n);

    parfor k=1:n
        pos = rand(1,2)*I_size*1.2-I_size*.6;
        R1 = rand(1)*(Rmax-Rmin)+Rmin;
        R2 = rand(1)*(Rmax-Rmin)+Rmin;
        r_theta = rand(1)*pi;
        I_tmp = (((X-pos(1))*cos(r_theta) + (Y-pos(2))*sin(r_theta))/R1).^2 + (((X-pos(1))*sin(r_theta) - (Y-pos(2))*cos(r_theta))/R2).^2 > 1;
        I = I.*I_tmp(ceil(.1*I_size):ceil(.1*I_size)+I_size-1,ceil(.1*I_size):ceil(.1*I_size)+I_size-1);
        Rm(k) = sqrt(R1*R2);
    end
    am = pi*mean(Rm.^2);
    pm = 2*pi*mean(Rm);
    I = ~I;
end