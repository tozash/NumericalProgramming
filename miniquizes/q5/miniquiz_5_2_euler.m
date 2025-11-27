% MiniQuiz 5.2 - Euler method + linear interpolation

a = 1.0;
b = 2.0;
h = 0.05;
y0 = -1.0;

% Run Euler's method
[t_vals, w_vals] = euler_explicit(a, b, y0, h);

% Print header for grid values
fprintf('%-10s %-15s %-15s %-15s\n', 't_i', 'w_i (Euler)', 'y(t_i) (Exact)', 'Error');
fprintf('%s\n', repmat('-', 1, 55));

% Compute exact and error on the grid and print
for i = 1:length(t_vals)
    t = t_vals(i);
    w = w_vals(i);
    y_true = exact_solution(t);
    err = w - y_true;
    fprintf('%-10.4f %-15.6f %-15.6f %-15.6e\n', t, w, y_true, err);
end

fprintf('\n=======================================================\n\n');

% Interpolation queries
t_queries = [1.052, 1.555, 1.978];

fprintf('%-10s %-20s %-15s %-15s\n', 't_query', 'Approx (Interp)', 'Exact', 'Error');
fprintf('%s\n', repmat('-', 1, 60));

for j = 1:length(t_queries)
    tq = t_queries(j);
    
    % Interpolate
    approx = interp_from_euler(t_vals, w_vals, tq);
    
    % Exact solution
    y_ex = exact_solution(tq);
    
    % Error
    err_interp = approx - y_ex;
    
    fprintf('%-10.3f %-20.6f %-15.6f %-15.6e\n', tq, approx, y_ex, err_interp);
end


%% Local Functions

function val = f(t, y)
    % RHS of the ODE: f(t,y) = 1/t^2 - y/t - y^2
    val = 1./t.^2 - y./t - y.^2;
end

function y = exact_solution(t)
    % Exact solution y(t) = -1./t
    y = -1./t;
end

function [ts, ws] = euler_explicit(a, b, y0, h)
    % Explicit Euler method:
    %   w_{i+1} = w_i + h * f(t_i, w_i)
    
    N = round((b - a) / h);
    ts = zeros(N+1, 1);
    ws = zeros(N+1, 1);
    
    ts(1) = a;
    ws(1) = y0;
    
    for i = 1:N
        ws(i+1) = ws(i) + h * f(ts(i), ws(i));
        ts(i+1) = ts(i) + h;
    end
end

function y_interp = linear_interp(t0, y0, t1, y1, t)
    % Linear interpolation between (t0, y0) and (t1, y1)
    %   y(t) = y0 + (t - t0) * (y1 - y0) / (t1 - t0)
    y_interp = y0 + (t - t0) * (y1 - y0) / (t1 - t0);
end

function yq = interp_from_euler(ts, ws, t_query)
    % Find index i such that ts(i) <= t_query <= ts(i+1)
    % and return linear_interp(ts(i), ws(i), ts(i+1), ws(i+1), t_query)
    
    % Default initialization
    yq = NaN;
    
    % Scan grid (ts is sorted)
    for i = 1:length(ts)-1
        if t_query >= ts(i) && t_query <= ts(i+1)
            yq = linear_interp(ts(i), ws(i), ts(i+1), ws(i+1), t_query);
            return;
        end
    end
    
    % Handle edge case for floating point equality at the very end
    if abs(t_query - ts(end)) < 1e-9
        yq = ws(end);
    end
end

