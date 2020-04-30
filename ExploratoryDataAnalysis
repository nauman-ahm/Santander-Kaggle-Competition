library(ggplot2)
library(tidyverse)
train_data<-read.csv("/kaggle/input/train.csv")
test_data<-read.csv("/kaggle/input/test.csv")
test_data<- mutate(test_data, target = NA)
aggregate = rbind(train_data, test_data)

## EDA
variables<-colnames(train_data)[3:202]
for(i in variables)
{
plot_1<-ggplot(data = aggregate, aes_string(x = i)) + geom_density(data = subset(aggregate, is.na(target)==1),color="green") + 
geom_density(data = subset(aggregate, is.na(target)==0), color="red")
plot(plot_1)
}
