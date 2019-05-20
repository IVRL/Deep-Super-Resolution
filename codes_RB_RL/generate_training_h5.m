% original data folder
hrdir = '/scratch/mfr/DIV2K/DIV2K_valid_HR/';
% configuration
p = 144;

% stride
stride = 32;


fnum = 1;
files = dir([hrdir '/' '*.png']);
savedir = 'valid';

% 3 channels
h5create([savedir '/hr.h5'],  '/data', [p p 3 Inf], 'Datatype', 'single', 'ChunkSize', [p p 3 1]);
h5create([savedir '/lr2.h5'], '/data', [p p 3 Inf], 'Datatype', 'single', 'ChunkSize', [p p 3 1]);
h5create([savedir '/lr3.h5'], '/data', [p p 3 Inf], 'Datatype', 'single', 'ChunkSize', [p p 3 1]);
h5create([savedir '/lr4.h5'], '/data', [p p 3 Inf], 'Datatype', 'single', 'ChunkSize', [p p 3 1]);

% 1 channel
h5create([savedir '/lhr.h5'],  '/data', [p p 1 Inf], 'Datatype', 'single', 'ChunkSize', [p p 1 1]);
h5create([savedir '/llr2.h5'], '/data', [p p 1 Inf], 'Datatype', 'single', 'ChunkSize', [p p 1 1]);
h5create([savedir '/llr3.h5'], '/data', [p p 1 Inf], 'Datatype', 'single', 'ChunkSize', [p p 1 1]);
h5create([savedir '/llr4.h5'], '/data', [p p 1 Inf], 'Datatype', 'single', 'ChunkSize', [p p 1 1]);

for k = 1:length(files)
    file = files(k).name;
    image = imread([hrdir '/' file]);
    
    [w h c] = size(image);
    i = 1;
    j = 1;
    
    while ((j + p <= h))
        hrpatch = image(i:i+p-1, j:j+p-1, :);
        lr2patch = imresize(imresize(hrpatch, 1/2), 2);
        lr3patch = imresize(imresize(hrpatch, 1/3), 3);
        lr4patch = imresize(imresize(hrpatch, 1/4), 4);
        
        h5write([savedir '/hr.h5'],  '/data', single(hrpatch),  [1 1 1 fnum], [p p 3 1]);
        h5write([savedir '/lr2.h5'], '/data', single(lr2patch), [1 1 1 fnum], [p p 3 1]);
        h5write([savedir '/lr3.h5'], '/data', single(lr3patch), [1 1 1 fnum], [p p 3 1]);
        h5write([savedir '/lr4.h5'], '/data', single(lr4patch), [1 1 1 fnum], [p p 3 1]);
        
        
        hrgpatch = rgb2gray(hrpatch);
        lr2gpatch = imresize(imresize(hrgpatch, 1/2), 2);
        lr3gpatch = imresize(imresize(hrgpatch, 1/3), 3);
        lr4gpatch = imresize(imresize(hrgpatch, 1/4), 4);
        
        h5write([savedir '/lhr.h5'],  '/data', single(hrgpatch),  [1 1 1 fnum], [p p 1 1]);
        h5write([savedir '/llr2.h5'], '/data', single(lr2gpatch), [1 1 1 fnum], [p p 1 1]);
        h5write([savedir '/llr3.h5'], '/data', single(lr3gpatch), [1 1 1 fnum], [p p 1 1]);
        h5write([savedir '/llr4.h5'], '/data', single(lr4gpatch), [1 1 1 fnum], [p p 1 1]);
        
        fnum = fnum + 1;
        
        if (mod(fnum, 100) == 0)
            sprintf('already generated %d patches\n', fnum)
        end
        
        j = j + stride;
        if (i + p > w)
            i = 1;
            j = j + stride;
        end
    end
end
sprintf('generated %d patches\n', fnum)
