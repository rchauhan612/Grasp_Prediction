y = gpresall(:, 2:end);
x = gpresall(:, 1);

[~, n] = size(y);
accs = [];
for i = 1:n
    accs = [accs sum(y(:, i) == x)];
end

accs = 100*accs / 297