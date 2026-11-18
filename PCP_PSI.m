% PCP-PSI: Principal Component Pursuit for Phase-Shifting Interferometry
% Authors:
% Shouyu Wang @ OptiX+ Laboratory, Wuxi University
% Javier Vargas @ CNB, CSIC
% Wei Yu @ OptiX+ Laboratory, Wuxi University
% Aihui Sun @ Computational Optics Laboratory, Jiangnan University
% Version 1, September, 2026

%% CCC
clear;
close;
clc;

%% 1. PSI Frame Construction
% ----------------------Adjustable Setting Parameters----------------------
Nx = 256;  % Pixel Numbers in X
Ny = 256;  % Pixel Numbers in Y
N = 20;  % PSI Frame Numbers
DeltaStep = pi/3;  % Phase-Shifting Angle Step
DistortedFrameRatio = 0.15;  % Distorted Frame Ratio (0-0.3 Suggested)
DefectivePixelRatio = 0.02;  % Defective Pixel Ratio (0-0.1 Suggested)
SNRGaussian = 15;  % Gaussian SNR (dB, 20-30 Suggested)
% ----------------------Computed Fixed Parameters--------------------------
M = Nx*Ny;  % Total Pixel Numbers
Delta_NoNoise = (0:N-1)*DeltaStep;  % Clean Phase-Shifting Angles
Delta_Noise = rand(1,N)*0.1;  % Phase-Shifting Angle Noise
Delta = Delta_NoNoise+Delta_Noise;  % Phase-Shifting Angles
% ----------------------Background and Modulation Amplitutes---------------
A = 0.5;  % Background Amplitute
B = 0.5;  % Modulation Amplitute
% ----------------------Distorted Frames-----------------------------------
FrameAnomalyIndex = randperm(N,floor(DistortedFrameRatio*N));
FrameAnomalyIndex = sort(FrameAnomalyIndex);
% ----------------------Specimen Phase Distribution------------------------
[Xsize, Ysize] = meshgrid(linspace(-1,1,Ny),linspace(-1,1,Nx));
Phi = 5*pi*(Xsize.^2+Ysize.^2)+2*pi*Xsize.*Ysize;
PhiWrap = angle(exp(1i*Phi));
% ----------------------PSI Frame Matrix-----------------------------------
X = zeros(M,N);  % PSI Frame Matrix
% ----------------------Distorted Frames-----------------------------------
for FrameNumber = 1:N
    if ismember(FrameNumber, FrameAnomalyIndex)
        TiltX = 5*randn();
        TiltY = 5*randn();
        TiltPhase = TiltX*Xsize+TiltY*Ysize;  % Distorted Phase
        I = A+B.*cos(Phi+Delta(FrameNumber)+TiltPhase);  % Distorted Frame
    else
        I = A+B.*cos(Phi+Delta(FrameNumber));  % Stable Frames
    end
% ----------------------Gaussian Noise-------------------------------------
    SignalPower = mean(I(:).^2);
    NoisePower = SignalPower/(10^(SNRGaussian/10));
    Noise = sqrt(NoisePower)*randn(size(I));
    INoisy = I+Noise;
% ----------------------Intensity Range------------------------------------
    INoisy(INoisy < 0) = 0;
    INoisy(INoisy > 1) = 1;
    X(:,FrameNumber) = INoisy(:);  % PSI Frame Matrix
end
% ----------------------Defective Pixels-----------------------------------
for FrameNumber = 1:1:N
    PixelAnomalyNumber = round(DefectivePixelRatio*M);  % Pixel Anomaly Number
    PixelAnomalyIndex = randperm(M,PixelAnomalyNumber);  % Pixel Anomaly Index
    for PixelNumber = 1:PixelAnomalyNumber
        if rand() > 0.5
            X(PixelAnomalyIndex(PixelNumber),FrameNumber) = 1;
        else
            X(PixelAnomalyIndex(PixelNumber),FrameNumber) = 0;
        end
    end
end

%% 2. PCP-PSI Framework (This Work)
% Use Augmented Lagrange Multiplier (ALM) to Solve:
% min||L||_*+λ*||S||_1  s.t. X = L+S
%
% Input： X - PSI Frame Matrix
% Output：L - Low Rank Matrix (Smooth Frame)
%         S - Sparse Matrix (Anomaly)
% ----------------------PCP Parameters-------------------------------------
Lambda = 1/sqrt(max(M,N));  % Theoretical Optimal λ
Mu = 1.25/norm(X,2);  % Initial Penalty Factor
Rho = 1.5;  % Penalty Factor Growth Coefficient
Epsilon = 1e-6;  % Convergence Tolerance
IterationNumberMax = 100;  % Maximum Iteration Number
IterationNumber = 0;  % Iteration
% ----------------------Initialize (Step 1 in This Work)-------------------
L = zeros(M, N);  % Initialize L
S = zeros(M, N);  % Initialize S
Y = X/max(norm(X,2),1);  % Initialize Y
% ----------------------Repeat Update (Step 2 in This Work)----------------
while IterationNumber < IterationNumberMax
    IterationNumber = IterationNumber+1;
% ----------------------Update L (Step 2a in This Work)--------------------
    % Solve: L = argmin||L||_*+(μ/2)*||X-L-S+Y/μ||_F^2
    % Perform Singular Value Soft Threshold Shrinkage on (X-S+Y/μ)
    T_L = X-S+Y/Mu;
    [U,Sigma,V] = svd(T_L,'econ');  % SVD of T_L
    SigmaVector = diag(Sigma);
    % Soft Threshold Shrinkage: Shrink(Sigma,1/μ) = sign(Sigma)*max(|Sigma|-1/μ,0)
    SigmaThreshold = sign(SigmaVector).*max(abs(SigmaVector)-(1/Mu),0);
    L = U*diag(SigmaThreshold)*V';  % Reconstruct L
% ----------------------Update S (Step 2b in This Work)--------------------
    % Solve: S = argmin lambda*||S||_1+(μ/2)*||X-L-S+Y/μ||_F^2
    T_S = X-L+Y/Mu;
    S = sign(T_S).*max(abs(T_S)-Lambda/Mu,0);
% ----------------------Update Y (Step 2c in This Work)--------------------
    Y = Y+Mu*(X-L-S);
% ----------------------Update μ (Step 2d in This Work)--------------------   
    Mu = min(Mu*Rho,1e10);
% ----------------------Check Convergence Conditions-----------------------
    Error = (norm((X-L-S),'fro'))/(norm(X,'fro'));
    if Error < Epsilon
        fprintf('Complete Convergence at Iteration Number of %d\n', IterationNumber);
        break;
    end
end
if IterationNumber == IterationNumberMax
    fprintf('Incomplete Convergence at Maximum Iteration Number of %d, \n', IterationNumberMax);
end
% ----------------------Distorted Frame Recognition------------------------
FrameEnergy = sum(S.^2, 1);  % S Energy of Each Frame
StableFrameIndex = find(~isoutlier(FrameEnergy,'median','ThresholdFactor',3));
DistortedFrameIndex = setdiff(1:N,StableFrameIndex);
fprintf('Setting Distorted Frames: %s',strjoin(string(FrameAnomalyIndex), ','));
fprintf('\n');
fprintf('Recognized Distorted Frames: %s',strjoin(string(DistortedFrameIndex), ','));
fprintf('\n');
% ----------------------Retain Stable Frames for Phase Retrieval-----------
L_clean = L(:, StableFrameIndex);
N_clean = size(StableFrameIndex, 2);

%% 3. Phase Retrieval Algoritm (PCA used here)
% ----------------------Matrix Rearrangement and Centering-----------------
PSI_Noise_Squeeze = L_clean;
PSI_Noise_Squeeze_Mean = mean(PSI_Noise_Squeeze,2);
PSI_Tilde = PSI_Noise_Squeeze-PSI_Noise_Squeeze_Mean;
% ----------------------Covariance Matrix----------------------------------
C = (PSI_Tilde'*PSI_Tilde);
% ----------------------Eigen Decomposition--------------------------------
[U,Lambda] = eig(C);
[~,idx] = sort(diag(Lambda),'descend');
U = U(:,idx);
Sigma = diag(Lambda(idx,idx));
% ----------------------Phase-Shifting Angle Estimation--------------------
U1 = U(:,1);
U2 = U(:,2);
Delta_PCA = atan2(U2,U1);
Delta_PCA = Delta_PCA-Delta_PCA(1);
% ----------------------Phase-Shifting Angle Correction--------------------
if Delta_PCA(2) < 0
    Delta_PCA = -Delta_PCA;
end
Delta_PCA = mod(Delta_PCA,2*pi);
% ----------------------Phase Under Detection Estimation-------------------
Y1 = PSI_Tilde*U1;
Y2 = PSI_Tilde*U2;
Phi_PCA_Squeeze = atan2(-Y2,Y1);
Phi_PCA = reshape(Phi_PCA_Squeeze,Nx,Ny);

%% LSM
PSI_LSM_Tilde = L_clean;
Delta_LSM = Delta_PCA;
% ----------------------LSM Iterations-------------------------------------
Iteration_Number = 5;
for i = 1:Iteration_Number
% ----------------------Phase Under Detection Update-----------------------
    XLSM = [ones(N_clean,1),cos(Delta_LSM),sin(Delta_LSM)];
    P = (XLSM'*XLSM)\(XLSM'*PSI_LSM_Tilde');
    Phi_LSM_Tilde = atan2(-P(3,:)',P(2,:)');
    Phi_LSM = reshape(Phi_LSM_Tilde,Nx,Ny);
% ----------------------Phase-Shifting Angle Update------------------------
    Q = [P(2,:)',P(3,:)'];
    Delta_LSM = zeros(N_clean,1);
    for n = 1:N_clean
        Z = (Q'*Q)\(Q'*(PSI_LSM_Tilde(:,n)-P(1,:)'));
        Delta_LSM_SingleFrame = atan2(Z(2),Z(1));
        Delta_LSM(n) = Delta_LSM_SingleFrame;
    end
    Delta_LSM = mod(Delta_LSM,2*pi);
end
PhiRetrieval = Phi_LSM;

%% 4. Result Display
% ----------------------X, L, and S of a Stable Frame----------------------
subplot(3, 3, 1);
imagesc(reshape(X(:, StableFrameIndex(1,1)), Nx, Ny));
axis equal tight; colorbar; clim([0 1]); colormap gray;
title('X (Stable Frame)');

subplot(3, 3, 2);
imagesc(reshape(L(:, StableFrameIndex(1,1)), Nx, Ny));
axis equal tight; colorbar; clim([0 1]); colormap gray;
title('L (Stable Frame)');

subplot(3, 3, 3);
imagesc(reshape(S(:, StableFrameIndex(1,1)), Nx, Ny));
axis equal tight; colorbar; clim([-1 1]); colormap gray;
title('S (Stable Frame)');
% ----------------------X, L, and S of a Distorted Frame-------------------
if isempty(DistortedFrameIndex)
    
else
    subplot(3, 3, 4);
    imagesc(reshape(X(:, DistortedFrameIndex(1,1)), Nx, Ny));
    axis equal tight; colorbar; clim([0 1]); colormap gray;
    title('X (Distorted Frame)');

    subplot(3, 3, 5);
    imagesc(reshape(L(:, DistortedFrameIndex(1,1)), Nx, Ny));
    axis equal tight; colorbar; clim([0 1]); colormap gray;
    title('L (Distorted Frame)');

    subplot(3, 3, 6);
    imagesc(reshape(S(:, DistortedFrameIndex(1,1)), Nx, Ny));
    axis equal tight; colorbar; clim([-1 1]); colormap gray;
    title('S (Distorted Frame)');
end
% ----------------------Phase Retrieval------------------------------------
subplot(3, 3, 7);
imagesc(PhiWrap);
axis equal tight; colorbar; colormap gray;
title('Setting Wrapped Phase');

subplot(3, 3, 8);
imagesc(PhiRetrieval);
axis equal tight; colorbar; colormap gray;
title('Retrieved Wrapped Phase');
