function pick = nms(boxes, overlap)  
  
% pick = nms(boxes, overlap)     输入： boxes为所有候选框的左下角和右上角x，y坐标。overlap为事先人为设定的阈值。
% Non-maximum suppression.       输出： pick，即所有保留下来的框
% Greedily select high-scoring detections and skip detections  
% that are significantly covered by a previously selected detection.  
  
if isempty(boxes)  
  pick = [];  
else  
  x1 = boxes(:,1);          %所有候选框的左下角顶点x   
  y1 = boxes(:,2);          %所有候选框的左下角顶点y   
  x2 = boxes(:,3);          %所有候选框的右上角顶点x   
  y2 = boxes(:,4);          %所有候选框的右上角顶点y  
 % s = boxes(:,end);         %所有候选框的置信度，可以包含1列或者多列，用于表示不同准则的置信度  
  area = (x2-x1+1) .* (y2-y1+1);   %所有候选框的面积  
  
  [vals, I] = sort(area);      %将所有候选框进行从小到大排序，vals为排序后结果，I为排序后标签  
  pick = [];  
  while ~isempty(I)  
    last = length(I);       %last代表标签I的长度，即最后一个元素的位置，（matlab矩阵从1开始计数）  
    i = I(last);            %所有候选框的中置信度最高的那个的标签赋值给i  
    pick = [pick; i];       %将i存入pick中，pick保存输出的NMS处理后的box的序号，即保存最后结果
    suppress = [last];      %将I中最大置信度的标签在I中位置赋值给suppress，suppress作用为类似打标志，  
    for pos = 1:last-1      %从1到倒数第二个进行循环  
      j = I(pos);           %得到pos位置的标签，赋值给j  
      xx1 = max(x1(i), x1(j));%左上角最大的x（求两个方框的公共区域）  
      yy1 = max(y1(i), y1(j));%左上角最大的y  
      xx2 = min(x2(i), x2(j));%右下角最小的x  
      yy2 = min(y2(i), y2(j));%右下角最小的y  
      w = xx2-xx1+1;          %公共区域的宽度  
      h = yy2-yy1+1;          %公共区域的高度  
      if w > 0 && h > 0     %w,h全部>0，证明2个候选框相交  
        o = w * h / area(j);%计算overlap比值，即交集占候选框j的面积比例  
        if o > overlap      %如果大于设置的阈值就去掉候选框j，因为候选框i的置信度最高  
          suppress = [suppress; pos];   %大于规定阈值就加入到suppress，即待删除的框
        end  
      end  
    end  
    I(suppress) = [];%将待删除的框删除，I中剩余未处理的框。当I为空结束循环  
  end    
end