function [idealM, monkeyM, optimalStim] = monkeyRLWhat(choice, rew, maxF)
%
% Input:
% choice:     image choice, 0/1
% rew:        binary reward vector
% maxF:       max trial on which a reversal could occur (typically i trial > the number of trials in a block)

%Output:
% monkey behavior posteriors:
% monkeyM.mscheduleP   probability over schedules
% monkeyM.mreverseP    probability over reversal point in block
% monkeyM.mtargetP     probability over target that is high reward at start
% of block
% idealM are the ideal observer versions of the same variables.

optimalStim = zeros(maxF, 1);

for trial = 1:maxF-1

    [idealM, monkeyM] = runPosterior(choice(1:trial), rew(1:trial), maxF);
        
    optimalStim(trial) = idealM.targetP(1);

end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [idealM, monkeyM] = runPosterior(choice, outcome, maxF)

trials = length(choice);

%%% start analysis from trial 0
minF = 0;

pLevels = 0.51 : 0.1 : 0.99;
    
%%% enumerate hypotheses    

%%% hypothesis space is N*K*2
%%% dim1: N = points where reversal could occur
%%% dim2: is schedule
%%% dim3: is whether target 1 is high reward first or second
ilikelihood = ones(trials+1, length(pLevels), 2);
mlikelihood = ones(trials+1, length(pLevels), 2);


%%% where does reversal occur
for reverse = (minF : 1 : maxF) %+1

    %%% schedule
    for pi = 1 : length(pLevels)

        p = pLevels(pi);

        %%% target 1 high prob
        for t1 = 1 : 2

            sequencep  = zeros(trials, 1);
            msequencep = zeros(trials, 1);
            
            for ti = 1 : maxF - 1

                if (t1 == 1 & ti < reverse) | (t1 == 2 & ti >= reverse) 
                    q = p;
                else
                    q = 1-p;
                end

                if ti <= length(choice)

                    if choice(ti) == 1 & outcome(ti) == 1
                        sequencep(ti) = q;
                    elseif choice(ti) == 1 & outcome(ti) == 0
                        sequencep(ti) = 1-q;
                    elseif choice(ti) == 0 & outcome(ti) == 1
                        sequencep(ti) = 1-q;
                    else 
                        sequencep(ti) = q;
                    end

                    if choice(ti) == 1
                        msequencep(ti) = q;
                    else
                        msequencep(ti) = 1 - q;
                    end

                else

                    sequence(ti) = 1;

                end

            end       

            ilikelihood(reverse+1, pi, t1) = prod(sequencep);
            mlikelihood(reverse+1, pi, t1) = prod(msequencep);

        end %%% target that is high prob (left or image 1)      
    end %%% schedule
end   %%% reversal trial



%%% hypothesis space is N*K*2
%%% dim1: N = points where reversal could occur
%%% dim2: is schedule
%%% dim3: is whether target 1 is high reward first or second

iZ = sum(sum(sum(sum(ilikelihood))));

ilikelihood = ilikelihood/iZ;
idealM.scheduleP   = squeeze(sum(sum(ilikelihood, 3), 1));
idealM.reverseP    = sum(sum(ilikelihood, 3), 2);
idealM.targetP     = squeeze(sum(squeeze(sum(ilikelihood, 1)), 1));

mZ = sum(sum(sum(sum(mlikelihood))));

mlikelihood  = mlikelihood/mZ;
monkeyM.mscheduleP   = squeeze(sum(sum(mlikelihood, 3), 1));
monkeyM.mreverseP    = sum(sum(mlikelihood, 3), 2);
monkeyM.mtargetP     = squeeze(sum(squeeze(sum(mlikelihood, 1)), 1));



