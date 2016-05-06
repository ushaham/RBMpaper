function [] = writeDatasetInCubamFormat(dataset)
%% Save to txt file
% Assumes f,y exists, and creates a the text files
%dataset='magic/magic_block_5.mat';
load(dataset)
f(f==-1)=0;
y(y==-1)=0;
dataset = dataset(1:strfind(dataset,'.')-1);
m = size(f,1);
n = size(f,2);
str = strcat(dataset,'_labels.txt');
str = strrep(str, 'simulated', 'cubam');
str = strrep(str, 'real', 'cubam');
outfile = fopen(str,'w');
fprintf(outfile, '%d %d %d\n', n,m,numel(f));
for classifier=0:(m-1)
    for instance=0:(n-1)
        fprintf(outfile, '%d %d %d\n', instance, classifier, f(classifier+1,instance+1));
    end;
end;
fclose(outfile);
str = strcat(dataset,'_gt.txt');
str = strrep(str, 'simulated', 'cubam');
str = strrep(str, 'real', 'cubam');
outfile = fopen(str,'w');
for instance=0:(n-2)
    fprintf(outfile, '%d,', y(instance+1));
end;
fprintf(outfile, '%d', y(n));
fclose(outfile);