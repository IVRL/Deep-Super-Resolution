% original data folder
hrdir = '/scratch/mfr/TestSets/Manga109/';
% configuration
savedir = '/scratch/mfr/GeneratedSets/';
name = 'Manga109/';

folders = {'hr', 'lr2', 'lr3', 'lr4', 'lhr', 'llr2', 'llr3', 'llr4'};
for i=1:size(folders, 2)
    subfolder = [savedir name char(folders(i))];
    if ~exist(subfolder, 'dir')
      mkdir(subfolder);
    end
end

fnum = 1;
files = dir([hrdir '/' '*.png']);

for k = 1:length(files)
    file = files(k).name;
    
    hrpatch = imread([hrdir '/' file]);
    
    if ndims(hrpatch) > 2
        lr2patch = imresize(imresize(hrpatch, 1/2), 2);
        lr3patch = imresize(imresize(hrpatch, 1/3), 3);
        lr4patch = imresize(imresize(hrpatch, 1/4), 4);

        imwrite(hrpatch,  [savedir name char(folders(1)) '/' file]);
        imwrite(lr2patch, [savedir name char(folders(2)) '/' file])
        imwrite(lr3patch, [savedir name char(folders(3)) '/' file])
        imwrite(lr4patch, [savedir name char(folders(4)) '/' file])
    
        hrpatch = rgb2gray(hrpatch);
    end
    
    llr2patch = imresize(imresize(hrpatch, 1/2), 2);
    llr3patch = imresize(imresize(hrpatch, 1/3), 3);
    llr4patch = imresize(imresize(hrpatch, 1/4), 4);
    
    imwrite(hrpatch,  [savedir name char(folders(5)) '/' file])
    imwrite(llr2patch, [savedir name char(folders(6)) '/' file])
    imwrite(llr3patch, [savedir name char(folders(7)) '/' file])
    imwrite(llr4patch, [savedir name char(folders(8)) '/' file])

    fnum = fnum + 1;
        
    if (mod(fnum, 10) == 0)
        sprintf('already generated %d patches\n', fnum)
    end
end
sprintf('generated %d files\n', fnum)


