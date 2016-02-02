function res = softThresholding(v,t)
res =  sign(v) .* max(abs(v)-t,zeros(size(v)));