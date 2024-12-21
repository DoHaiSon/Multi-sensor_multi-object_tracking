% Set random seed
rng(2808);

% Test cases with different sizes and parameters
test_data = struct();

% Test randn
test_data.randn_1 = randn(1, 1);      % Single value
test_data.randn_2 = randn(3, 2);      % Matrix
test_data.randn_3 = randn(5, 1);      % Column vector

% Test rand
test_data.rand_1 = rand(1, 1);        % Single value
test_data.rand_2 = rand(2, 4);        % Matrix
test_data.rand_3 = rand(4, 1);        % Column vector

% Test poisson
lambda_vals = [5, 10, 15];
test_data.poisson = zeros(1, length(lambda_vals));
for i = 1:length(lambda_vals)
    test_data.poisson(i) = poissrnd(lambda_vals(i));
end

% Test normal distribution
test_data.normal_1 = normrnd(0, 1);           % Single value
test_data.normal_2 = normrnd(2, 0.5, 3, 2);   % Matrix with mu=2, sigma=0.5

% Test uniform distribution
test_data.uniform_1 = unifrnd(-1, 1);         % Single value
test_data.uniform_2 = unifrnd(0, 5, 2, 3);    % Matrix

% Test randi
test_data.randi_1 = randi(10);                % Single value up to 10
test_data.randi_2 = randi(100, 3, 3);         % 3x3 matrix up to 100

% Test randperm
test_data.randperm = randperm(10);            % Permutation of 1:10

% Save test data
save('matlab_rng_test_data.mat', 'test_data');