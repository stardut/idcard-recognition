clc,clear
close all
img_path = 'img/test3';
colorImage = imread([img_path '.jpg']);
colorImage  = imresize(colorImage,[300 450]);
I = rgb2gray(colorImage);
figure,imshow(I),title('GrayImage')
% Detect MSER regions.
[mserRegions, mserConnComp] = detectMSERFeatures(I, ...
    'RegionAreaRange',[20 5000],'ThresholdDelta',0.8);
figure,imshow(I),hold on
plot(mserRegions, 'showPixelList', true,'showEllipses',false)
title('MSER regions'),hold off

mserStats = regionprops(mserConnComp, 'BoundingBox', 'Eccentricity', ...
    'Solidity', 'Extent', 'Euler', 'Image');

%% 
strokeWidthThreshold = 0.4;
% Process the remaining regions
for j = 1:numel(mserStats)
    regionImage = mserStats(j).Image;
    regionImage = padarray(regionImage, [1 1], 0);
    distanceImage = bwdist(~regionImage);
    skeletonImage = bwmorph(regionImage, 'thin', inf);
    strokeWidthValues = distanceImage(skeletonImage);
    strokeWidthMetric = std(strokeWidthValues)/mean(strokeWidthValues);
    strokeWidthFilterIdx(j) = strokeWidthMetric > strokeWidthThreshold;
end
% Remove regions based on the stroke width variation
mserRegions(strokeWidthFilterIdx) = [];
mserStats(strokeWidthFilterIdx) = [];
% Show remaining regions
figure,imshow(I),hold on
plot(mserRegions, 'showPixelList', true,'showEllipses',false)
title('After Removing Non-Text Regions Based On Stroke Width Variation'),hold off

%%
% Get bounding boxes for all the regions
bboxes = vertcat(mserStats.BoundingBox);
% Convert from the [x y width height] bounding box format to the [xmin ymin
% xmax ymax] format for convenience.
xmin = bboxes(:,1);
ymin = bboxes(:,2);
xmax = xmin + bboxes(:,3) - 1;
ymax = ymin + bboxes(:,4) - 1;

% Expand the bounding boxes by a small amount.
expansionAmount = 0.03;
xmin = (1-expansionAmount) * xmin;
ymin = (1-expansionAmount) * ymin;
xmax = (1+expansionAmount) * xmax;
ymax = (1+expansionAmount) * ymax;
% Clip the bounding boxes to be within the image bounds
xmin = max(xmin, 1);
ymin = max(ymin, 1);
xmax = min(xmax, size(I,2));
ymax = min(ymax, size(I,1));
% Show the expanded bounding boxes
% Show the expanded bounding boxes
expandedBBoxes = [xmin ymin xmax-xmin+1 ymax-ymin+1];
boxes = [xmin ymin xmax ymax];
pick = nms(boxes,0.7);
IExpandedBBoxes = insertShape(colorImage,'Rectangle',expandedBBoxes,'LineWidth',3);
figure,imshow(IExpandedBBoxes),title('Expanded Bounding Boxes Text')
IExpandedBBoxes = insertShape(colorImage,'Rectangle',expandedBBoxes(pick,:),'LineWidth',3);
figure,imshow(IExpandedBBoxes),title('Expanded Bounding Boxes Text')

%% Save samples
fid = fopen([img_path '.txt'], 'w');
for n=1:length(pick)
    fprintf(fid, '%0.3f %0.3f %0.3f %0.3f\n', boxes(pick(n),:));
end
fclose(fid);

% for n =1:length(pick)
%     reg = boxes(pick(n),:);
%     SubImg = colorImage(reg(2):reg(4),reg(1):reg(3));
%     imwrite(SubImg,['img/samples/' num2str(84+n) '.jpg']);
% end

%% mergeboxes
% boxes = boxes(pick,:);
% [val,I] = sort(boxes(:,2));
% y_delta = val(2:end)-val(1:end-1);
% y={};
% ymin=[];
% y_th = 10;
% for n=1:length(I)-1
%     if(y_delta(n)<y_th)
%         ymin = [ymin; boxes(I(n),:)];
%         if(n == length(I)-1)
%             ymin = [ymin; boxes(I(end),:)];
%             y = [y; ymin]; 
%         end
%     else
%         ymin = [ymin; boxes(I(n),:)];
%         y = [y; ymin];
%         ymin = [];
%         if(n == length(I)-1)
%             y = [y; boxes(I(end),:)];
%         end 
%     end
% end
% 
% [clu ~] = size(y);
% k=[];
% for n=1:clu
%    [m ~] = size(y{n,1}); 
%    k = [k; m];
% end
% [D,id] = sort(k,'descend');


%y = y{id(1:8)};

% bbb = vertcat(y);
% IExpandedBBoxes = insertShape(colorImage,'Rectangle',[bbb(:,1) bbb(:,2) bbb(:,3)-bbb(:,1) bbb(:,4)-bbb(:,2)],'LineWidth',3);
% figure,imshow(IExpandedBBoxes),title('Expanded Bounding Boxes Text')
