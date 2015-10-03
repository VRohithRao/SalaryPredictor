args <- commandArgs()
setwd("/home/yaxin/Documents/CS6220/Project/census-income")
# Find the columns with null values: incase the 
find.null.data <- read.table("census-income.data", 
                             sep = ",",strip.white = T, colClasses = rep("factor", 15))
cat("Total number of rows")
print(nrow(find.null.data))
cat("Columns with missing values:")
for (j in names(find.null.data)) {
    if(any(find.null.data[,j] == "?") ) {
        print(j)
    }
}
## output:
#[1] "V2"
#[1] "V7"
#[1] "V14"

# This piece needs a while to run
row.with.miss.values <- 0
for (i in seq(1, nrow(find.null.data))) {    
    if(any(find.null.data[i,] == "?")) {
        row.with.miss.values <- row.with.miss.values + 1
    }
}
cat("Number of rows containing missing values:")
print(row.with.miss.values)
# > cat("Number of rows containing missing values:")
# Number of rows containing missing values:
# > print(row.with.miss.values)
# [1] 2399

train.data <- read.table("census-income.data", sep=",",strip.white = T)
names(train.data) <- c("age", "workclass", "fnlwgt", "education", "education-num", 
                       "marital-stat","occupation", "relationship", "race",
                       "sex", "capital-gain", "capital-loss", "hours-per-week", 
                       "native-contry", "label")
cat("Summary of the training data:")
summary(train.data)
cat("Summary of the label column:")
summary(train.data[, "label"])
