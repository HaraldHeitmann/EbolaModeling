function loss = loss_func(params,data, tspan, N)
%LOSS_FUNC Summary of this function goes here
%   Detailed explanation goes here
    y0 = [N-data(1), 0,data(1),0];
    [t,y] = ode45(@ebola_sys,tspan, y0, [], params, N);
    loss = (y(:,3)+y(:,4)-data);
    params
end

