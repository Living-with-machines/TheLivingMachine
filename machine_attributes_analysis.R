# Initialization

## Load libraries and data, and set initial parameters

library(dplyr)
library(knitr)
library(plyr)
library(pheatmap)
library("Hmisc")
library(gdata)
library(knitr)
library(tseries)
library(pander)
library(lexicon)
library(stringr)
library(textclean)
library(tm)
library(stringr)



#Directory names:
  
path_in =  file.path("/Users", "bmcgillivray", "Documents", "OneDrive", "The Alan Turing Institute", "The Alan Turing Institute", "Mariona Coll Ardanuy - LivingMachines2", fsep = "/")
path_data = paste(path_in, "older/", sep = "/")
path_plots = paste(path_in, "plots", sep = "/")
path_out = file.path("/Users", "bmcgillivray", "Documents", "OneDrive", "The Alan Turing Institute", "OneDrive - The Alan Turing Institute", "Research", "2021", "LwM", "Living machines2", "Machine attributes analysis")

#Corpus names:
  
corpora = c( "jsa", "blb", "hmd", "rsc")

# Read files

datasets <- list()
count = 0
for (corpus in corpora) {
  print(corpus)
  count = count + 1
  file_name = paste(corpus, "_withmere.tsv", sep = "")
  dataset = read.csv(paste(path_data, file_name, sep = "/"), header = T, sep = "\t", colClasses=c("synt"="character"))
  print(dim(dataset))
  # only keep relevant columns:
  if (corpus == "blb"){
    cols = c("identifier", "date", "currentSentence", "synt")
  }
  else if (corpus == "hmd"){
    cols = c("item_code", "year", "currentSentence", "synt")
  }
  else if (corpus == "jsa"){
    cols= c("filename", "year", "currentSentence", "synt")
  }
  else {
    cols= c("volume", "year", "currentSentence", "synt")
  }
  dataset = dataset[,cols]
  if (corpus == "blb"){
    colnames(dataset)[2] <- "year"
  }
  # add to list of datasets:
  datasets[[count]] = dataset
}



# Overview of the datasets:

print("Names of datasets")
for (dataset in datasets){
  print(names(dataset))
}

# Summaries:

for (dataset in datasets){
#print(summary(dataset$year))
  hist(dataset$year)
}

# Extract modifiers of "machine"

machine_dep = data.frame(matrix(ncol = 6, nrow = 1)) # list of dependants of machine (tokens, pos, synt_role)
#machine_dep_tokens = data.frame(matrix(ncol = 4, nrow = 1)) # list of dependants of machine (tokens)
#machine_dep_pos = data.frame(matrix(ncol = 4, nrow = 1)) # list of dependants of machine (pos)
#machine_dep_syntrole = data.frame(matrix(ncol = 4, nrow = 1)) # list of dependants of machine (synt_role)
colnames(machine_dep) = c("corpus", "year", "currentSentence", "dependant", "pos", "synt_role")


# Second corpus: blb:

print("Second corpus: blb")
c = 2
    name = corpora[c]
    dataset = datasets[[c]]
    
    for (i in 1:nrow(dataset)){
    syntactic_analysis = dataset[i,]$synt
    syntactic_analysis_list = as.list(strsplit(syntactic_analysis, ')'))
        this_year = dataset[i,]$year
        sentence = dataset[i,]$currentSentence
        print(paste(name, i, "out of", nrow(dataset), sep = " "))
        fields = strsplit(syntactic_analysis_list[[1]], ',')
        for (j in 1:length(fields)){
            #print(j)
            #print(fields[[j]])
            #print(class(fields[[j]]))
            #print(length(fields[[j]]))
            if (length(fields[[j]]) > 1){
                synt_head = fields[[j]][length(fields[[j]])]
                #print("synt_head:")
                #print(synt_head)
                synt_role = fields[[j]][length(fields[[j]])-1]
                if (length(fields[[j]]) == 4){
                    token = fields[[j]][1]
                    pos = fields[[j]][2]
                }
                else {
                    token = fields[[j]][2]
                    pos = fields[[j]][3]
                }
                # I clean:
                list = c(synt_head, synt_role, token, pos)
                for(index in 1:length(list)){
                    list[index] <- gsub("[()]", "", gsub("\\[|\\]", "", gsub("'", '', gsub("\\s", "", list[index]))))
                }
                synt_head = list[1]
                synt_role = list[2]
                token = list[3]
                pos = list[4]
                if (grepl('machine', tolower(synt_head))){
                    #print("head is machine")
                    machine_dependants = c(name, this_year, sentence, token, pos, synt_role) 
                    #print(machine_dependants)
                    machine_dep = rbind(machine_dep, machine_dependants)
                }
            }
       }
    }



machine_dep
machine_dep = machine_dep[2:nrow(machine_dep),]
#machine_dep <- data.frame(lapply(machine_dep, as.factor))

#Save the file:

write.csv(machine_dep, paste(path_out, paste('machine_dep_', name, '.csv', sep = ""), sep = "/"), row.names = F)

