%n = repmat(300, [1 5]);
total = 3200;
k = 4;
%n = repmat(total/k, [1 k]);
p = randfixedsum(8,1,1,0,1);
n = round(total * p);
a = gendatm(n);
X = a.data;
Y = a.nlab;

scatterd(a, 2);
save('synth8_unbalanced.mat', 'X', 'Y');