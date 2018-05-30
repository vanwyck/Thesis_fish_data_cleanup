git_path <- 'C:/Users/Thoma/Documents/Github/Thesis_fish_data_cleanup/'

#load required packages - install if necessary
library(zoo)
library(TMB) 

source('supportFuncs_automized_bi.R')
compile("yaps_bi.cpp")

#read in hydrophones
hydro_data<- read.csv(paste(git_path, 'Site_data/Receiver_location.csv', sep = ''),row.names = 1)
receivers <- rownames(hydro_data)

IDS <- 103

for (ID in IDS){

  # read in the generated TOA matrix
  data <- read.csv(paste(git_path,'Example_data/TOA_data_',toString(ID),'.csv',sep = ''),check.names = FALSE)
  passings <- unique(data$groups_pas)
  
  mean_BI = unique(data$mean_BI)
  
  #initialize output dataframe
  YAPS_data <- data.frame()
  time_taken <- array(dim=length(passings))

  for (i in 1:length(passings)){
    toa <- data[data$groups_pas ==passings[i],colnames(data)%in% receivers]
    ss <- data$soundspeed[data$groups_pas == passings[i]]
    DATETIME <- data$synced_time[data$groups_pas == passings[i]]
    output <- NULL
    attempt <-1
    # if YAPS fails to calculate atrack, it is retried five times
    while(is.null(output) && attempt <=5){
      attempt <- attempt+1
      try({
        output <- runYAPS_bi(hydro_data,toa,ss,mean_BI,max_it = 500000)
        output$pas_nr = passings[i]
        output$synced_time <- DATETIME
        YAPS_data <- rbind(YAPS_data,output)
      })
    }
  }
  
  write.csv(YAPS_data,paste(git_path,'Example_data/YAPS_track_',toString(ID),'.csv',sep = ''))

}

