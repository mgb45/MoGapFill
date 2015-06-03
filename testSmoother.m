%% Load markers
load ('gait-raw.csv');

markers = gait_raw;
clear data;

%% Parameters
tol = 5e-6;
sigmaR = 5e-2;

%% Get nice training data - all sets with full markers
[r,~] = find(isnan(markers));
bins = setdiff(1:length(markers),unique(r));
Train = markers(bins,:);

%% Project Training data into pca space
[U,V,l] = pca(Train(:,:));
mPCA = mean(Train(:,:));

%% Determine required number dimensions for model fitting
d = find(abs(cumsum(l)./sum(l)-1) < tol,1,'first');

%% Find model process noise
sigma_a = var(diff(Train));
Q = U(:,1:d)'*((diag(sigma_a)).^2)*U(:,1:d);

%% Forward stage
Estimate = zeros(length(markers),size(markers,2));
frate = 0;
bin = 1;
state_pred{1} = randn(d,1);
state{1} = randn(d,1);
cov{1} = 1e12*eye(d);
cov_pred{1} = 1e12*eye(d);
for j = 2:length(markers)+1
    tic
    %% Construct measurement matrix
    H = diag(~isnan(markers(j-1,:)));
    H(sum(H,2)==0,:) = [];
    %% Measurement noise
    R = sigmaR*eye(size(H,1));
    Ht = H*U(:,1:d);

    %% Extract valid measurements
    z = markers(j-1,~isnan(markers(j-1,:)))';

    state_pred{j} = state{j-1};
    cov_pred{j} = cov{j-1} + Q;

    K = cov_pred{j}*Ht'*inv(Ht*cov_pred{j}*Ht' + R);

    state{j} = state_pred{j} + K*(z - (Ht*state_pred{j} + H*mPCA'));

    cov{j} = (eye(d) - K*Ht)*cov_pred{j};

    est = U(:,1:d)*state{j} + mPCA';

    frate = frate+toc;
    
    cla
    plot3(est(1:3:end),est(2:3:end),est(3:3:end),'o')
    hold on;
    plot3(markers(j-1,1:3:end),markers(j-1,2:3:end),markers(j-1,3:3:end),'r+')
    axis equal
    axis([min(min(markers(:,1:3:end))) max(max(markers(:,1:3:end))),min(min(markers(:,2:3:end))) max(max(markers(:,2:3:end))), min(min(markers(:,3:3:end))) max(max(markers(:,3:3:end)))])
    grid on
    view(3)
    drawnow;
     
    bin = bin + 1;
    
    fprintf ('Forward pass: Dim %d, Frame %d, Average proc time %0.4f \r',d, j,frate/j)
end

%% Backward stage
state_new = cell(length(state)-1,1);
cov_new = cell(length(cov)-1,1);
state_new{end} = state{end};
cov_new{end} = cov{end};
frate = 0;
for j = length(state)-1:-1:2
    tic

    state_new{j} = state{j} + cov{j}*inv(cov_pred{j})*(state{j+1} - state_pred{j+1});
    cov_new{j} = cov{j} + cov{j}*inv(cov_pred{j})*(cov{j+1} - cov_pred{j+1})*cov{j};

    Estimate(j-1,:) = U(:,1:d)*state_new{j} + mPCA';

    frate = frate + toc;
    
    cla
    plot3(Estimate(j-1,1:3:end),Estimate(j-1,2:3:end),Estimate(j-1,3:3:end),'o')
    hold on;
    plot3(markers(j-1,1:3:end),markers(j-1,2:3:end),markers(j-1,3:3:end),'r+')
    axis equal
    axis([min(min(markers(:,1:3:end))) max(max(markers(:,1:3:end))),min(min(markers(:,2:3:end))) max(max(markers(:,2:3:end))), min(min(markers(:,3:3:end))) max(max(markers(:,3:3:end)))])
    grid on
    view(3)
    drawnow;
    
    fprintf ('Backward pass: Dim %d, Frame %d, Average proc time %0.4f \r',d, j,frate/(length(markers)-j))
end

fprintf ('\nDone\n')
Estimate(end,:) = U(:,1:d)*state_new{end} + mPCA';

