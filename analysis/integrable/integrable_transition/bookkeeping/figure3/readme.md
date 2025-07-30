# Figure 3 Book Keeping


## First sub figure
csr_heatmap_blurred_L=x.txt contains a 200x200 table of scalar values, comprising the CSR density heatmap (with gaussian blur) for X layer integrable circuit. This corresponds to the first subfigure of figure 3.

In addition, csr_raw_L=x.txt contains the raw values from the CSRs, should you want to generate the heatmap from scratch.

## Second subfigure

scatterplot_mean.txt contains \<r\> -\<-cos(theta)\> values for L=5, 10, 20, 30, 40, 50. First column is \<r\>, second is \<-cos(theta)\>. Each row correspond to the different L.

scatterplot_std.txt contains average standard deviation calculated from bootstrap resampling and refitting of the models

scatterplot_theory.txt contains theoretic values for \<r\> -\<-cos(theta)\>. First row is \<r\> and \<-cos(theta)\> for AI (chaotic). Second row is \<r\> and \<-cos(theta)\> for FF (integrable). A mild gaussian smoothing has been applied to make the plot look better.


## Third Subfigure

x = 5, 20, 50

radial_L=x.txt contains radial density of CSR for X layer integrable circuit. First column is central bin values for the density. Second column is the density corresponding to each bin.

radial_std_L=x.txt contains a single column of bootstrap errors corresponding to the values in radial_L=x.txt.


radial_L=x.txt contains angular density of CSR for X layer integrable circuit. First column is central bin values for the density. Second column is the density corresponding to each bin.

radial_std_L=x.txt contains a single column of bootstrap errors corresponding to the values in angular_L=x.txt.


radial_FF_theory.txt contains radial density of CSR for FF4,1 Haar. First column is central bin values for the density. Second column is the density corresponding to each bin.

radial_AI_theory.txt contains radial density of CSR for AI4,1 Haar. First column is central bin values for the density. Second column is the density corresponding to each bin.

angular_FF_theory.txt and angular_AI_theory.txt follow same convention.



