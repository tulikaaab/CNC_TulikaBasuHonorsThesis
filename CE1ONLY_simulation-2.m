%=======================
% MODEL FITTING SCRIPT (CE1 only, with confidenceType confirmation)
%=======================
function sigmascaleSTAR_2()
    clc; close all;
    fignums = [10 11 12; 13 14 15];
    sigmaScaleRange = 0.05:0.05:10;
    ntrials = 10000;
    true_sigma = 5.0;
    nsubjects = 24;

    recovered_sigmas = zeros(nsubjects, 1);
    recovered_CE1 = zeros(nsubjects, 1);

    for subj = 1:nsubjects
        fprintf('\n--- Simulated Subject %d ---\n', subj);

        % Simulate data ONCE using true sigma
        sigmaScale_true = [true_sigma, true_sigma];
        isHumanM2_generate = 0; % Use 0 to generate with same conf, 1 for different conf
        [modelPChoices_gt, ~, ~, ~, ~, ~] = simulate_model_fit(ntrials, sigmaScale_true, fignums, isHumanM2_generate, subj);

        % Fit sigma by minimizing CE1 between generated data and simulated fits
        CE1_values = zeros(length(sigmaScaleRange), 1);

        for s = 1:length(sigmaScaleRange)
            test_sigma = sigmaScaleRange(s);
            sigmaScale_vec = [test_sigma, test_sigma];

            [modelPChoices_fit, ~, ~, ~, ~, ~] = simulate_model_fit(ntrials, sigmaScale_vec, fignums, isHumanM2_generate, subj);

            CE1_values(s) = crossEntropy_Type1({modelPChoices_gt{1}, modelPChoices_fit{2}});
        end

        [min_CE1, best_idx] = min(CE1_values);
        best_sigmascale = sigmaScaleRange(best_idx);

        recovered_sigmas(subj) = best_sigmascale;
        recovered_CE1(subj) = min_CE1;

        figure(subj); clf;
        plot(sigmaScaleRange, CE1_values, '-o', 'LineWidth', 2);
        hold on;
        plot(best_sigmascale, min_CE1, 'r*', 'MarkerSize', 12);
        xlabel('Sigma Scale'); ylabel('Type 1 Cross-Entropy (CE1)');
        title(sprintf('Subject %d: Recovered \\sigma = %.2f | True \\sigma = %.2f', subj, best_sigmascale, true_sigma));
        grid on;
    end

    fprintf('\n==== SUMMARY OVER %d SUBJECTS ====\n', nsubjects);
    fprintf('True sigma used to generate: %.2f\n', true_sigma);
    fprintf('Mean recovered sigma: %.4f (SE = %.4f)\n', mean(recovered_sigmas), std(recovered_sigmas)/sqrt(nsubjects));
    fprintf('Mean CE1: %.4f\n', mean(recovered_CE1));
end

function [modelPChoices, meanConf, allConfidence, allChoices, confidenceDists, pCorrect] = simulate_model_fit(ntrials, sigmaScale_vec, fignums, isHumanM2, subj)
    modelPChoices = cell(1, 2);
    meanConf = zeros(14, 2);
    allConfidence = zeros(14, ntrials, 2);
    allChoices = zeros(14, ntrials, 2);
    confidenceDists = cell(1, 2);
    pCorrect = zeros(14, 2);

    for ds = 1:2
        confidenceType = determineConfidenceType(ds, isHumanM2);
        fprintf('Subject %d | Dataset %d using Confidence Type: %d\n', subj, ds, confidenceType);

        sigma = eye(3) * sigmaScale_vec(ds);
        [conds, muA, muB, muC, priors] = defineDistributions();

        [modelPChoices{ds}, meanConf(:,ds), allConfidence(:,:,ds), allChoices(:,:,ds), confDist, pCorrect(:,ds)] = ...
            simulateConditions(ds, ntrials, sigma, conds, muA, muB, muC, priors, confidenceType, fignums);

        confidenceDists{ds} = confDist;
    end
end

function [pChoice, meanConf, allConfidence, allChoices, confidenceDists, pCorrect] = simulateConditions(ds, ntrials, sigma, conds, muA, muB, muC, priors, confidenceType, fignums)
    allChoices = zeros(size(conds,1), ntrials);
    allConfidence = zeros(size(conds,1), ntrials);
    pChoice = zeros(size(conds,1), 3);
    meanConf = zeros(size(conds,1), 1);
    pCorrect = zeros(size(conds,1), 1);
    confidenceDists = zeros(3, 20, size(conds,1));

    for c = 1:size(conds,1)
        data = mvnrnd(conds(c,:), sigma, ntrials);
        pA = mvnpdf(data, muA, sigma) * priors(1);
        pB = mvnpdf(data, muB, sigma) * priors(2);
        pC = mvnpdf(data, muC, sigma) * priors(3);
        totalP = pA + pB + pC;
        posteriors = [pA./totalP, pB./totalP, pC./totalP];

        switch confidenceType
            case 1
                [conf, choice] = max(posteriors, [], 2);
            case 2
                [conf, choice] = max(data, [], 2);
        end

        allChoices(c,:) = choice;
        allConfidence(c,:) = conf;
        pCorrect(c) = mean(choice == 1);
        meanConf(c) = mean(conf);
        pChoice(c,:) = [mean(choice==1), mean(choice==2), mean(choice==3)];
    end
end

function [conds, muA, muB, muC, priors] = defineDistributions()
    conds = [10 9 8; 10 9 1; 10 3 1; 9 8 7; 9 8 1; 9 2 1;
             7 6 5; 7 6 2; 7 3 1; 6 2 1; 5 4 3; 5 4 1;
             5 2 1; 4 3 2];
    muA = [10 0 0];
    muB = [0 10 0];
    muC = [0 0 10];
    priors = [1/3, 1/3, 1/3];
end

function confidenceType = determineConfidenceType(ds, isHumanM2)
    if ds == 1 || isHumanM2 == 0
        confidenceType = 2;
    else
        confidenceType = 1;
    end
end

function CE = crossEntropy_Type1(modelPChoices)
    for i = 1:2
        modelPChoices{i}(modelPChoices{i} == 0) = eps;
    end
    CE = -sum(modelPChoices{1}(:) .* log(modelPChoices{2}(:)));
end

function CE2= crossEntropy_Type2(confidenceDists)
    A = confidenceDists{1};
    B = confidenceDists{2};
    A(A == 0) = eps;
    B(B == 0) = eps;
    CE2 = -sum(A(:) .* log(B(:)));
end
