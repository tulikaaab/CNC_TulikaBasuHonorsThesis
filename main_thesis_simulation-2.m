%=======================
# Behavioral Analysis of Confidence Models in 3AFC Tasks

This repository contains MATLAB code for simulating and analyzing behavioral data in a 3-alternative forced-choice (3AFC) task using two models of confidence: the **Posterior Probability Model of Confidence (PPMC)** and the **Magnitude of Evidence Model of Confidence (MEMC)**.

## Overview

The main script, `main_thesis_simulation.m`, simulates trials under different experimental conditions and confidence models. It generates:

- Choice probabilities (`pChoice`)
- Confidence ratings (`meanConf`)
- Condition-specific confidence histograms
- Plots for visual comparison
- Type 1 and Type 2 cross-entropy losses (CE1 and CE2) to evaluate model fit

This simulation supports model-based analysis of both simulated and real human decision behavior, particularly for confidence judgments.

%=======================
function main_thesis_simulation()
    clc; close all; clear;
    fignums = [1 2 3; 4 5 6];
    sigmaScale = [10, 10];
    ntrials = 10000;
    isHumanM2 = 1;

    for ds = 1:2
        confidenceType = determineConfidenceType(ds, isHumanM2);
        sigma = scaleSigma(ds, sigmaScale);
        [conds, muA, muB, muC, priors] = defineDistributions();

        [modelPChoices{ds}, meanConf(:,ds), allConfidence(:,:,ds), allChoices(:,:,ds), confidenceDists{ds}, pCorrect(:,ds)] = simulateConditions(ds, ntrials, sigma, conds, muA, muB, muC, priors, confidenceType, fignums);

        confidenceDists{ds} = generateHistograms(ds, allConfidence(:,:,ds), allChoices(:,:,ds), conds, ntrials, confidenceDists{ds}, fignums);

        figure(fignums(ds,3))
        plot(pCorrect(:,ds), meanConf(:,ds), 'k.', 'markersize', 30)
        set(gca, 'FontSize', 20)  
        xlabel('pCorrect', 'FontSize', 30); ylabel('mean conf', 'FontSize', 30)
    end

    % Cross-entropy losses
    CE1 = crossEntropy_Type1(modelPChoices);
    CE2 = crossEntropy_Type2(confidenceDists);

    fprintf('Type 1 Cross-Entropy Loss: %.4f\n', CE1);
    fprintf('Type 2 Cross-Entropy Loss: %.4f\n', CE2);
    disp('Smaller cross entropy is better!')
end

function confidenceType = determineConfidenceType(ds, isHumanM2)
    if ds == 1 || isHumanM2 == 0
        confidenceType = 1;
    else
        confidenceType = 2;
    end
end

function sigma = scaleSigma(ds, sigmaScale)
    sigma = eye(3) * sigmaScale(ds);
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


        % Normalize confidence to [0, 1] per choice within condition
        conf_norm = NaN(size(conf));  % Preallocate
        
        % Normalize confidence to [-1, 1] per choice within condition
        for ch = 1:3
            idx = (choice == ch);
            if sum(idx) > 1
                min_c = min(conf(idx));
                max_c = max(conf(idx));
                range_c = max_c - min_c;
                if range_c > 0
                    conf_norm(idx) = 2 * ((conf(idx) - min_c) / range_c) - 1;
                else
                    conf_norm(idx) = 0;  % Flat value if no range
                end
            elseif sum(idx) == 1
                conf_norm(idx) = 0;  % Define singleton as 0
            end
        end
        
        allChoices(c,:) = choice;
        allConfidence(c,:) = conf_norm;



        pCorrect(c) = mean(choice == 1);
        meanConf(c) = mean(conf);
        pChoice(c,:) = [mean(choice==1), mean(choice==2), mean(choice==3)];
        ConfByChoice = [mean(conf(choice==1)), mean(conf(choice==2)), mean(conf(choice==3))];

        figure(fignums(ds,1))
        subplot(2,14,c); bar(pChoice(c,:)); ylim([0 1]); title(num2str(conds(c,:)), 'FontSize', 15)
        set(gca, 'FontSize', 15) 
        if c==1, ylabel('pChoice', 'FontSize', 30); end
        subplot(2,14,c+14); bar(ConfByChoice);
        if ds == 2 && confidenceType == 2
            ylim([0 15]);
        else
            ylim([0 1]);
        end
        set(gca, 'FontSize', 15)
        if c==1, ylabel('mean conf', 'FontSize', 30); xlabel('choice', 'FontSize', 30); end
    end
end

function confidenceDists = generateHistograms(ds, allConfidence, allChoices, conds, ntrials, confidenceDists, fignums)

    maxConf = max(allConfidence(:)); % Find the maximum of all the confidence scores
    if maxConf > 1
        % Normalize the confidence to between 0 and 1
        allConfidence = allConfidence ./ maxConf;
    end

    figure(fignums(ds,2));
    binEdges = 0:0.05:1;

    for c = 1:size(conds,1)
        conf = allConfidence(c,:);
        choice = allChoices(c,:);
        countsA = histcounts(conf(choice==1), binEdges) ./ ntrials;
        countsB = histcounts(conf(choice==2), binEdges) ./ ntrials;
        countsC = histcounts(conf(choice==3), binEdges) ./ ntrials;
        confidenceDists(1,:,c) = countsA;
        confidenceDists(2,:,c) = countsB;
        confidenceDists(3,:,c) = countsC;
        binCenters = binEdges(1:end-1) + diff(binEdges)/2;

        subplot(4,4,c); hold on
        plot(binCenters, countsA, '-r', 'LineWidth', 2);
        plot(binCenters, countsB, '-g', 'LineWidth', 2);
        plot(binCenters, countsC, '-b', 'LineWidth', 2);
        hold off
        maxY = max([countsA countsB countsC]);
        if maxY > 0
            ylim([0 maxY*1.1]);
        else
            ylim([0 1]);
        end
        title(['Condition: ', num2str(conds(c,:))])
        if c == 1, legend('A','B','C'); end
        xlabel('(Normalized) Confidence'); ylabel('Probability')
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
