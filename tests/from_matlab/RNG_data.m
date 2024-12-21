% Initialize test data structure
test_data = struct();

% Test randn
rng(2808);  % Reset seed before randn tests
test_data.randn_1 = randn();          % Single value (no args)
test_data.randn_2 = randn(1);         % Single arg -> (n,1) vector 
test_data.randn_3 = randn(3, 2);      % Matrix (m,n)
test_data.randn_4 = randn(5, 1);      % Column vector
test_data.randn_5 = randn(1, 4);      % Row vector

% Test rand
rng(2808);  % Reset seed before rand tests
test_data.rand_1 = rand();            % Single value (no args)
test_data.rand_2 = rand(3);           % Single arg -> (n,1) vector
test_data.rand_3 = rand(2, 4);        % Matrix (m,n)
test_data.rand_4 = rand(4, 1);        % Column vector
test_data.rand_5 = rand(1, 3);        % Row vector

% Test poisson
rng(2808);  % Reset seed before poisson tests
test_data.poisson_1 = poissrnd(5);              % Single value
test_data.poisson_2 = poissrnd(10, 3, 1);       % Column vector
test_data.poisson_3 = poissrnd(15, 1, 4);       % Row vector
test_data.poisson_4 = poissrnd(7, 2, 3);        % Matrix

% Test normal distribution
rng(2808);  % Reset seed before normal tests
test_data.normal_1 = normrnd(0, 1);             % Single value (no size args)
test_data.normal_2 = normrnd(2, 0.5, 4);        % Single arg -> (n,1) vector
test_data.normal_3 = normrnd(2, 0.5, 3, 2);     % Matrix (m,n)
test_data.normal_4 = normrnd(-1, 2, 5, 1);      % Column vector
test_data.normal_5 = normrnd(1, 0.1, 1, 3);     % Row vector

% Test uniform distribution
rng(2808);  % Reset seed before uniform tests
test_data.uniform_1 = unifrnd(-1, 1);           % Single value (no size args)
test_data.uniform_2 = unifrnd(0, 5, 3);         % Single arg -> (n,1) vector
test_data.uniform_3 = unifrnd(0, 5, 2, 3);      % Matrix (m,n)
test_data.uniform_4 = unifrnd(-2, 2, 4, 1);     % Column vector
test_data.uniform_5 = unifrnd(0, 1, 1, 5);      % Row vector

% Test randi
rng(2808);  % Reset seed before randi tests
test_data.randi_1 = randi(10);                  % Single value (no size args)
test_data.randi_2 = randi(10, 5);               % Single arg -> (n,1) vector
test_data.randi_3 = randi(100, 3, 3);           % Matrix (m,n)
test_data.randi_4 = randi(50, 4, 1);            % Column vector
test_data.randi_5 = randi(20, 1, 6);            % Row vector

% Test randperm
rng(2808);  % Reset seed before randperm tests
test_data.randperm_1 = randperm(5);             % Small permutation
test_data.randperm_2 = randperm(10);            % Medium permutation
test_data.randperm_3 = randperm(15);            % Large permutation

% Test multivariate normal distribution
rng(2808);  % Reset seed before multivariate normal tests
test_data.mvnorm_1 = mvnrnd([0; 0], eye(2));                   % Single sample, 2D std normal
test_data.mvnorm_2 = mvnrnd([1; 2], [2 0.5; 0.5 1]);           % Single sample, 2D with correlation
test_data.mvnorm_3 = mvnrnd([0; 0], eye(2), 3);                % Multiple samples, 2D std normal
test_data.mvnorm_4 = mvnrnd([1; 2; 3], eye(3), 4);             % Multiple samples, 3D
test_data.mvnorm_5 = mvnrnd([0], [1], 5);                      % Multiple samples, 1D

% Save test data
save('matlab_rng_test_data.mat', 'test_data');