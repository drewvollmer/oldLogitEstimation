% calc_trans_loglik.m
% A function calculating the log likelihood of observed transactions given candidate preference
% parameters
% Drew Vollmer 2019-05-23

function sumLoglik = calc_trans_loglik(params, data)

% Unpack parameters
numQ = length(unique(data.quality_group));
alphas = params(1:numQ);
%alphas = params(1)*ones(numQ, 1);
betas = [0; params((numQ + 1):end)];
%betas = [0; params(2:end)];

% Get the log-likelihood of each observed purchase
blockTable = table(cell(50*length(unique(data.opponent)), 1), ...
                   zeros(50*length(unique(data.opponent)), 1));
blockTable.Properties.VariableNames = {'opponent', 'time'};
idx = 1;
for opp = unique(data.opponent)'
    for t = 1:50
        blockTable.time(idx) = t;
        blockTable.opponent{idx} = opp{1};
        idx = idx + 1;
    end
end

loglik = zeros(height(blockTable), 1);
tot_trans_count = sum(data.num_trans > 0);
for row = 1:height(blockTable)

    % Find the relevant observations
    sample = (data.time_to_event == blockTable.time(row)) & ...
              strcmp(data.opponent, blockTable.opponent{row});

    % If there are no transactions in the sample, move on
    if sum(data.num_trans(sample)) == 0
        continue
    end

    % Calculate inclusive values and number of transactions for each number of seats
    maxSeats = max(data.num_trans(sample));
    incVals = zeros(maxSeats / 2, 1);
    transCounts = zeros(maxSeats / 2, 1);
    for i = 1:length(incVals)
        seatSample = sample & (data.num_seats >= 2*i);
        incVals(i) = sum(exp( -alphas(data.quality_group(seatSample)).*data.price(seatSample) + ...
                              betas(data.quality_group(seatSample)) ));

        transCounts(i) = sum( data.num_trans(sample) == 2*i );
    end

    % The log-likelihood is the sum of log(prob) for each transaction
    for i = find(sample)'
        % Skip if there are no transactions
        if data.num_trans(i) == 0
            continue
        end

        % Otherwise, find the purchase probability and add its log to log-likelihood
        loglik(row) = loglik(row) + log( exp(-alphas(data.quality_group(i)).*data.price(i) + ...
                                             betas(data.quality_group(i)) ) / ...
                                         incVals(data.num_trans(i) / 2) );

    end

end


sumLoglik = sum(loglik);

end
