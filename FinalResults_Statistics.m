f1Score = [
    0.738067019	0.785532534	0.847840987	0.877584509	0.887618713	0.86146013	0.867518494	0.87511816	0.981417625	0.8795529;
    0.722240103	0.809825146	0.853050306	0.872432606	0.83577265	0.89901438	0.861843107	0.872380553	0.978764479	0.870859979;
   0.79158361	0.814543977	0.856096668	0.892269148	0.870698036	0.875843961	0.854054054	0.885014138	0.974672858	0.857893584;
     0.800926901	0.783113245	0.86509716	0.881675788	0.850657109	0.882796688	0.873280763	0.877840909	0.979854621	0.883416866
    ];

youdenJ = [
    0.560393105	0.581434163	0.703721702	0.757223961	0.784814282	0.70095919	0.723791488	0.758578816	0.962700018	0.764439842;
    0.531839149	0.590082277	0.716051371	0.741575123	0.691942917	0.775797007	0.665855947	0.752825768	0.957614027	0.74505265
    0.6376033	0.659783929	0.724523601	0.794274152	0.761826959	0.764724849	0.715106777	0.787480097	0.950777295	0.74007551;
    0.63884838	0.596172246	0.745945693	0.782209151	0.731056862	0.771164709	0.747745696	0.775063133	0.961091032	0.78521914;
    ];

%%
clc
for idx=1:size(f1Score,1)

    [a_h, a_p, a_w] = shapiroTest(f1Score(idx,:), 0.05);
    disp("SW P-Value for F1 Score for " + num2str(idx) + " is " + a_p + " -- Hypothesis: " + a_h);

    [a_h, a_p, a_w] = shapiroTest(youdenJ(idx,:), 0.05);
    disp("SW P-Value for J Youden for " + num2str(idx) + " is " + a_p + " -- Hypothesis: " + a_h);
    disp("-------------------------")
end


%%
anova1(f1Score')
xlabel("Experimento")
ylabel("F1-Score [%]")

%%
anova1(youdenJ')
xlabel("Experimento")
ylabel("J de Youden [-]")