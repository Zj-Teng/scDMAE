#part2 
rm(list = ls())
library(data.table)
#-----------------------------------------------------------Clustering-------------------------------------------------------------------
re_exp <- read.table("D:/scDMAE/output/gene_exp.csv",header = TRUE,sep=",",row.names = 1)
exp <- read.table("D:/scDMAE/Data/Clustering/Pollen_HVG.csv",header = TRUE,sep=",",row.names = 1)
colnames(re_exp) <- colnames(exp)
rownames(re_exp) <- rownames(exp)
for (i in 1:nrow(re_exp)){
  for (j in 1:ncol(re_exp)){
    if(re_exp[i,j]<0){
      re_exp[i,j]<-0
    }
  }
}
my_exp <- 0.9*(10**(re_exp)-1)+0.1*exp
my_exp <- round(my_exp,3)
write.csv("D:/scDMAE/Data/Clustering/Pollen_scDMAE.csv",row.names = T)

