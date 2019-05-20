function img = mod_crop(img, scale)
if size(img, 3) == 1
    sz = size(img);
    sz = sz - mod(sz, scale);
    img = img(1:sz(1), 1:sz(2));
else
    tmpsz = size(img);
    sz = tmpsz(1:2);
    sz = sz - mod(sz, scale);
    img = img(1:sz(1), 1:sz(2),:);
end