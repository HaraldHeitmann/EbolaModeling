clear all;
p0 = [500,0.01,50];
lb = [0,0,0];
ub = [inf,inf,inf];
Ng = 11750000;
Ns = 6092000;
Nl = 4294000;
N = Nl;
loader_train();
loader_test();
options = optimoptions(@lsqnonlin,'MaxFunEvals',2500);
[param,resnorm] = lsqnonlin(@(params)loss_func(params, cases, Day1, N), p0,lb,ub, options);
param
resnorm
y0 = [N-cases(1), 0,cases(1),0];
tspan = 0:250;
[t1,y] = ode45(@ebola_sys,tspan, y0, [], param, N);
predicted = y(:,3)+y(:,4);
plot(t1,predicted,Day1,cases,'x',Day_test,cases_test,'*');
MSE = 0;
for i = 1:length(t1)
   for j = 1:length(Day_test)
      if t1(i) == Day_test(j)
          MSE = MSE + (y(j)-predicted(j))^2;
      end
   end
end
MSE = MSE/length(Day_test)