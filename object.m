[trD, trLb, valD, valLb, trRegs, valRegs] = HW4_Utils.getPosAndRandomNeg();
c=10;
warning('off','all')
[alpha,fval,w,b]=svm(trD,trLb,c);
HW4_Utils.genRsltFile(w,b,"val","file_3_4_1");
[ap,prec,rec]=HW4_Utils.cmpAP("file_3_4_1","val");
ap_list=zeros(10,1);
objectives=zeros(10,1);
disp(ap);
disp(fval*-1)
load(sprintf('%s/%sAnno.mat', HW4_Utils.dataDir, 'train'), 'ubAnno');
%% 
imagefiles = dir('trainIms/*.jpg');
number_files = length(imagefiles);
disp(number_files); 

%% 
for i =1:10
    disp("iteration : ");
    disp(i);
    posD = [];
    negD = [];
    for j = 1:size(trLb, 1)
       if trLb(j) == 1
           posD = [posD trD(:, j)];
       else
           if alpha(j)-0.001>0
               negD = [negD trD(:, j)];
           end
       end
    end
    hardest_negative_samples=[];
    for l =1:number_files
       ubs = ubAnno{l};
       im =imread(sprintf('%s/%sIms/%04d.jpg',HW4_Utils.dataDir, 'train', l));
       rects = HW4_Utils.detect(im,w,b,0);
       %disp(size(rects))
       %disp(rects)
       num_positives = sum(rects(end, :)>0);
       rects = rects(:,1:num_positives+5);
%        disp("num_positives")
%        disp(num_positives)
%        disp("rects")
%        disp(size(rects))
       [imH, imW,~] = size(im);
       badIdxs = or(rects(3,:) > imW, rects(4,:) > imH);
       rects = rects(:,~badIdxs);
      
   
       for m=1:size(ubs,2)
            overlap = HW4_Utils.rectOverlap(rects, ubs(:,m));                    
            rects = rects(:, overlap < 0.2);
            if isempty(rects)
                break;
            end
       end
       
       for k = 1:size(rects, 2)
                imReg = im(rects(2,k):rects(4,k), rects(1,k):rects(3,k),:);
                imReg = imresize(imReg, HW4_Utils.normImSz);
                feature = HW4_Utils.cmpFeat(rgb2gray(imReg));
                feature = feature / norm(feature);
                hardest_negative_samples = [hardest_negative_samples feature];
       end
            
       if size(hardest_negative_samples, 2) > 1000
             break;
       end    
    end
    negD=[negD hardest_negative_samples];
    trD = [posD negD];
    trLb = ones(size(posD, 2), 1);
    trLb= [trLb; -1*ones(size(negD,2),1)];
    [alpha,fval,w,b] =svm(trD,trLb,c);
    objectives(i) = -1*fval;
    disp(fval);
    HW4_Utils.genRsltFile(w, b, 'val', "3_3_val.mat");
    [ap_new, ~, ~] = HW4_Utils.cmpAP("3_3_val.mat", 'val');
    ap_list(i) = ap_new;
    %disp(ap_new)
end
%% 
disp("objective values")
disp(objectives);
disp("ap values")
disp(ap_list);
% 
numbers = linspace(1, 10, 10);
subplot(2,1,1);

plot(numbers, objectives);
ylabel('objective values')
subplot(2,1, 2);

plot(numbers, ap_list);
ylabel('ap values')
%HW4_Utils.genRsltFile(w, b, "test", "submissiontest2.mat");
