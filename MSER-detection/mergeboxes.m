function out_boxes = mergeboxes( boxes )

[val,I] = sort(boxes(:,2));
y_delta = val(2:end,2)-val(1:end-1,2);
y={};
y_th = 15;
for n=1:length(I)
    if(y_delta(n)<y_th)
        ymin = [ymin; boxes(I(n),:)];
    else
       y = [y; ymin]; 
       ymin = boxes(I(n),:);
    end
end

out_boxes = 0;
end

