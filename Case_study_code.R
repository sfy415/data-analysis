#####interview case study#####
###@Time : 2019/11/18
###@Author: Feiyi Su
###@Email: feiyi.su@aalto.fi

###https://www.business-science.io/business/2019/03/11/ab-testing-machine-learning.html


###  load packages ###
# Core packages
library(tidyverse)
library(tidyquant)
library(lattice)

# Modeling packages
library(parsnip)
library(recipes)
library(rsample)
library(yardstick)
library(broom)

# Connector packages
library(rpart)
library(rpart.plot)
library(xgboost)



###############   data cleaning   #######################
      ###  a . missing value cleaning ###

## set working directory
setwd("C:\\Users\\feiyi\\Desktop\\3inteview\\Results\\Results")

raw.da<-readxl::read_xlsx("Home Experiment_raw.xlsx")

## deal with missing values
raw.da %>%
  map_df(~ sum(is.na(.))) %>%
  gather(key = "feature", value = "missing_count") %>%
  arrange(desc(missing_count))
## see which values are missing
raw.da %>%
  filter(is.na(action))
## we don't have action information at these four timestamp.
## we will remove these observations.

da <- raw.da %>%
  filter(!is.na(action))

          ### b. date format conversion ###

library(lubridate)
set.seed(1)
# encode group as "1:experiment; 0: group"
data_formatted <- da %>%mutate(Experiment = case_when(group == "control" ~ 0 , group == "experiment" ~1))
# Create date from timestamp
data_formatted <- data_formatted %>% mutate(date =as.Date(timestamp))



# Create a Day of Week
data_formatted <- data_formatted %>% mutate(DOW=wday(as.Date(timestamp))) %>% mutate(Pageviews= case_when(action == "view" ~ 1 , action == "click" ~0) ) %>% mutate(Clicks = case_when(action == "view" ~ 0, action == "click" ~1) )

## write csv to directory and use pivot table of excel to do some cleaning and then read cleaned data into clean_da
write.csv(data_formatted,"clean.csv")
clean_da<-readxl::read_xlsx("cleaned.xlsx")

# Create a Day of Week
clean_da2 <- clean_da %>% mutate(DOW=wday(Date)) %>%
  # Add row id
  mutate(row_id = row_number()) %>%
  # Shuffle the data (note that set.seed is used to make reproducible)
  sample_frac(size = 1)

clean_da2 %>% glimpse()

## With the data formatted properly for analysis, we can now separate
## into training and testing sets using an 80% / 20% ratio.

###########  Modeling ##################
    #####  Linear Regression  ######

set.seed(1)
split_obj <- clean_da2 %>%
  initial_split(prop = 0.8, strata = "Experiment")

train_tbl <- training(split_obj)
test_tbl  <- testing(split_obj)

## We can take a quick glimpse of the training data.
train_tbl %>% glimpse()
#  188 observations randomly selected.

## we can take a quick glimpse of the testing data.
test_tbl %>% glimpse()
#  the remaining 46 observations.

                 #### Modeling #####
            #####  Linear Regression ####

## https://www.twblogs.net/a/5cc4b2c6bd9eee3971146497/zh-cn


model_01_lm <- linear_reg("regression") %>%
  set_engine("lm") %>%
  fit(Clicks ~ ., data = train_tbl %>% select(-row_id))
# knitr::kable() used for pretty tables
model_01_lm %>%
  predict(new_data = test_tbl) %>%
  bind_cols(test_tbl %>% select(Clicks)) %>%
  metrics(truth = Clicks, estimate = .pred) %>%
  knitr::kable()

## plot data
model_01_lm %>%
  # Format Data
  predict(test_tbl) %>%
  bind_cols(test_tbl %>% select(Clicks)) %>%
  mutate(observation = row_number() %>% as.character()) %>%
  gather(key = "key", value = "value", -observation, factor_key = TRUE) %>%

  # Visualize
  ggplot(aes(x = observation, y = value, color = key)) +
  geom_point() +
  expand_limits(y = 0) +
  theme_tq() +
  scale_color_tq() +
  labs(title = "Clicks: Prediction vs Actual",
    subtitle = "Model 01: Linear Regression (Baseline)")

linear_regression_model_terms_tbl <- model_01_lm$fit %>%
  tidy() %>%
  arrange(p.value) %>%
  mutate(term = as_factor(term) %>% fct_rev())

# knitr::kable() used for pretty tables
linear_regression_model_terms_tbl %>% knitr::kable()

# We can visualize the importance separating "p.values" of 0.05 with a red dotted line.
linear_regression_model_terms_tbl %>%
  ggplot(aes(x = p.value, y = term)) +
  geom_point(color = "#2C3E50") +
  geom_vline(xintercept = 0.05, linetype = 2, color = "red") +
  theme_tq() +
  labs(title = "Feature Importance",
    subtitle = "Model 01: Linear Regression (Baseline)")

## As we can see in the graph, Pageviews are judged strong predictors with a p-value less than 0.05. Also, We note that the coefficient of Experiment is 0.8641552, and because the term is binary (0 or 1) this can be interpreted as increasing clicks by  0.8641552 per day when the Experiment is run.

           #######  Decision Trees #####


# Decision Trees can do something about non-linearities and compliment linear models by providing a different way of viewing the problem.

model_02_decision_tree <- decision_tree(
  mode = "regression",
  cost_complexity = 0.001,
  tree_depth = 5,
  min_n = 4) %>%
  set_engine("rpart") %>%
  fit(Clicks ~ ., data = train_tbl %>% select(-row_id))

###
model_02_decision_tree %>%
  predict(new_data = test_tbl) %>%
  bind_cols(test_tbl %>% select(Clicks)) %>%
  metrics(truth = Clicks, estimate = .pred) %>%
  knitr::kable()

### The MAE of the predictions is approximately the same as the linear model at 1.9 Clicks per day.

model_02_decision_tree %>%
  # Format Data
  predict(test_tbl) %>%
  bind_cols(test_tbl %>% select(Clicks)) %>%
  mutate(observation = row_number() %>% as.character()) %>%
  gather(key = "key", value = "value", -observation, factor_key = TRUE) %>%

  # Visualize
  ggplot(aes(x = observation, y = value, color = key)) +
  geom_point() +
  expand_limits(y = 0) +
  theme_tq() +
  scale_color_tq() +
  labs(title = "Clicks: Prediction vs Actual",
    subtitle = "Model 02: Decision Tree")

model_02_decision_tree$fit %>%
  rpart.plot(
    roundint = FALSE,
    cex = 0.8,
    fallen.leaves = TRUE,
    extra = 101,
    main = "Model 02: Decision Tree")

######### Interpretation ########
# Each decision is a rule, and Yes is to the left, No is to the right. The top features are the most important to the model ("Pageviews"). The decision tree shows that "Experiment" is involved in the decision rules. The rules indicate a when Experiment >= 0.5, there is a drop in clicks.



           ##########  KPI 1 ##########

#### Test if view-to-click conversion rate has improved from control
## to experiment

## set working directory
setwd("C:\\Users\\feiyi\\Desktop\\3inteview\\Results\\Results")

## read data into "da"
da<-readxl::read_xlsx("Home Experiment_raw.xlsx")

## load library
library(tidyverse)

conversion<-da %>% filter(!is.na(action)) %>% group_by(group) %>% summarise(views= sum(action=="view"),clicks=sum(action=="click"), conversion.rate=sum(action=="click")/sum(action=="view"))

pa<-conversion$conversion.rate[1]
pb<-conversion$conversion.rate[2]
na<-conversion$views[1]
nb<-conversion$views[2]

## calculate z
z<-(pb-pa)/(sqrt(pb*(1-pb)/na+pa*(1-pa)/na))
z
1-dnorm(z)

dnorm(z)


            #########  KPI 2 ##########

### Test if average time on page shortened because of the GIF;

## set working directory
setwd("C:\\Users\\feiyi\\Desktop\\3inteview\\Results\\Results")

## read data into "da"
da<-readxl::read_xlsx("Home Experiment_raw.xlsx")

## load library
library(tidyverse)

da<- da%>% filter(!is.na(action))
## convert duration to seconds
Res <- as.POSIXlt(da$duration)
options(digits.secs = 2)
Duration<-Res$min*60+Res$sec-1

## add a new column "Duration" to da
da<-cbind(da,Duration)
da$Duration

## calculate average & total time spent between viewing and clicking on webpage
da %>% group_by(group) %>% summarise(mean.time=mean(as.numeric(Duration),na.rm=TRUE),sum.time=sum(as.numeric(Duration),na.rm=TRUE))

### It appears that experiment group and control group doesn't differ because people spent as much time on page before clicking and being directied to subcategories.
### To further prove the hypothesis that experiment group doesn't differ control group in terms of time spend, here I use T-test

## before T-test, test of homoscedasticity is needed (i.e.are the variances homogenous)
CG <- subset(da, group == "control")
TG <- subset(da, group == "experiment")
var.test(CG$Duration, TG$Duration)

## Since p > 0.05,we can assume that the variances of both samples are homogenous.We can run classic Student's two-sample t-test
test<- t.test(CG$Duration, TG$Duration, var.equal = TRUE)

### Since the p-value of 0.822 is greater than the significance level of 0.05, we fail to reject the H0. That is to say, time spent on page doesn't differ between groups.



     ##########   KPI 3 ##############


#####  Draw funnel plots
# install.packages("plotly")
library(plotly)

## calculate total views and clicks of both control and experiment groups
tbl<-da %>% group_by(group) %>% summarise(sum.view=sum(action=="view"),sum.click=sum(action=="click"))
tbl$sum.view[1]

## draw control group conversion funnel
control_funnel<-plot_ly() %>%
  add_trace(
    type = "funnel",
    y = c("view","click"),
    x = c(tbl$sum.view[1],tbl$sum.click[1]),
    marker=list(color=c("deepskyblue","lightsalmon")))%>%layout(yaxis = list(categoryarray = c("View", "Click")))
control_funnel

## draw experiment group conversion funnel
experiment_funnel<-plot_ly() %>%
  add_trace(
    type = "funnel",
    y = c("view","click"),
    x = c(tbl$sum.view[2],tbl$sum.click[2]),
    marker=list(color=c("deepskyblue","lightsalmon")))%>%layout(yaxis = list(categoryarray = c("View", "Click")))
experiment_funnel


    ##########   Supplementary analysis ##############

   ###further explore the click rate by time period between groups

## set working directory
setwd("C:\\Users\\feiyi\\Desktop\\3inteview\\Results\\Results")

## Different from above analysis,before read data, because now I want to focus on click rate, 
## So combine the data with both click and view into one row.then, we got 6326 data
##1 and 7 =  weekend day, 2-6 = work day 

## read data into "train_data"
train_data = read.csv('pred_test.csv',na.strings=c("", '#N/A', '[]'))

train_data$he <- 1
train_data$group_rate <- 0
train_data$hour_rate <- 0
train_data$year_rate <- 0
train_data$month_rate <- 0
train_data$weekday_rate <- 0
train_data$day_rate <- 0

## calculate click hour rate/year rate/month rate/weekday rate/day rate/group rate
for (i in 1:6326){
  train_data$hour_rate[i]<-sum(train_data$action.click.1.[train_data$hour==train_data$hour[i]&train_data$group==train_data$group[i]])/ sum(train_data$he[train_data$hour==train_data$hour[i]&train_data$group==train_data$group[i]])
  
  train_data$year_rate[i]<-sum(train_data$action.click.1.[train_data$year==train_data$year[i]&train_data$group==train_data$group[i]])/ sum(train_data$he[train_data$year==train_data$year[i]&train_data$group==train_data$group[i]])
  
  train_data$month_rate[i]<-sum(train_data$action.click.1.[train_data$month==train_data$month[i]&train_data$group==train_data$group[i]])/ sum(train_data$he[train_data$month==train_data$month[i]&train_data$group==train_data$group[i]])
  
  train_data$weekday_rate[i]<-sum(train_data$action.click.1.[train_data$weekday==train_data$weekday[i]&train_data$group==train_data$group[i]])/ sum(train_data$he[train_data$weekday==train_data$weekday[i]&train_data$group==train_data$group[i]])
  
  train_data$day_rate[i]<-sum(train_data$action.click.1.[train_data$day==train_data$day[i]&train_data$group==train_data$group[i]])/ sum(train_data$he[train_data$day==train_data$day[i]&train_data$group==train_data$group[i]])  
  
  train_data$group_rate[i]<-sum(train_data$action.click.1.[train_data$group==train_data$group[i]])/ sum(train_data$he[train_data$group==train_data$group[i]]) 
}

## 1 and 7 =  weekend day, 2-6 = work day 
## I set worktime is 8:00am - 18:00pm and relax time is other time. 

train_data$work_relax_time <- 0
train_data$work_relax_rate <- 0
train_data$weekend_time <- 0
train_data$weekend_rate <- 0

for (i in 1:6326){
  if (train_data$hour[i]>7 & train_data$hour[i]<19) {
    train_data$work_relax_time[i]='worktime'
  }
  else {
    train_data$work_relax_time[i]='relaxtime'
  }
  
}
for (i in 1:6326){
  train_data$work_relax_rate[i]<-sum(train_data$action.click.1.[train_data$work_relax_time==train_data$work_relax_time[i]&train_data$group==train_data$group[i]])/ sum(train_data$he[train_data$work_relax_time==train_data$work_relax_time[i]&train_data$group==train_data$group[i]])
  
}
train_data$work_relax_time<-as.factor(train_data$work_relax_time)


for (i in 1:6326){
  if (train_data$weekday[i]>1 & train_data$weekday[i]<7) {
    train_data$weekend_time[i]='workday'
  }
  else {
    train_data$weekend_time[i]='weekend'
  }
  
}
for (i in 1:6326){
  train_data$weekend_rate[i]<-sum(train_data$action.click.1.[train_data$weekend_time==train_data$weekend_time[i]&train_data$group==train_data$group[i]])/ sum(train_data$he[train_data$weekend_time==train_data$weekend_time[i]&train_data$group==train_data$group[i]])
  
}

train_data$weekend_time<-as.factor(train_data$weekend_time)

train= train_data[,c(3:9,11:20)]
train<-train %>% mutate_if(is.character, as.factor)

#########Draw click rate plots

ggplot(train, aes(x = hour, y = hour_rate, colour = factor(group))) + 
  geom_point(size = 5)

ggplot(train, aes(x = as.factor(month), y = month_rate, colour = factor(group))) + 
  geom_point(size = 5)

ggplot(train, aes(x = weekday, y = weekday_rate, colour = factor(group))) + 
  geom_point(size = 5)

ggplot(train, aes(x = day, y = day_rate, colour = factor(group))) + 
  geom_point(size = 5)


##########LR analysis#######
lm1 <- lm(train$weekday_rate~group+weekday,data=train)
summary(lm1)

lm2 <- lm(train$hour_rate~group+hour,data=train)
summary(lm2)


##
lm3 <- lm(train$month_rate~group+month,data=train)
summary(lm3)

lm4 <- lm(train$work_relax_rate~group+work_relax_time,data=train)
summary(lm4)

lm5 <- lm(train$weekend_rate~group+weekend_time,data=train)
summary(lm5)

