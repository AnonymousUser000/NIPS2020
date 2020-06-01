function [trainSet, testSet] = splitTrainTest(data, ratio)
m = size(data, 1);
d = [1:m]';

trainSet = [];
comm1 = find(data(:, 3) < 0);
sizeComm1 = length(comm1);
rp = randperm(sizeComm1);
rpRatio = rp(1:floor(sizeComm1 * ratio));
ind = comm1(rpRatio);
trainSet = [trainSet; data(ind, :)];
d = setdiff(d, ind);

comm2 = find(data(:, 3) > 0);
sizeComm2 = length(comm2);
rp = randperm(sizeComm2);
rpRatio = rp(1:floor(sizeComm2 * ratio));
ind = comm2(rpRatio);
trainSet = [trainSet; data(ind, :)];
d = setdiff(d, ind);

testSet = data(d, :);
end

