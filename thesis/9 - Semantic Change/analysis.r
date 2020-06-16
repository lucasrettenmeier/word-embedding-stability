# Packages
require(lme4)
require(MuMIn)
library(reticulate)

# NumPy
np <- import("numpy")

# Read Data
path <- "/home/lucas/data/experiments/coha/analysis/"
files <- list.files(path)

# NP Array for Results
results <- array( data = rep(0,3*2*6*2), dim =  c(3,2,6,2))

i <- 1
for (model in list("word2vec", "glove", "fasttext")){
	j <- 1
	for (data_type in list("genuine", "random")){
		k <- 1
		for (size in list("0001", "0002", "0004", "0008", "0016", "0032")){
			file_name <- paste(model, data_type, size, sep = "_")
			if (file_name %in% files){
				npz <- np$load(paste(path,file_name, sep = ""))

				words = npz$f[["words"]]
				freq = npz$f[["frequencies"]]
				disp = npz$f[["displacements"]]

				d <- data.frame("words" = words, "freq" = freq, "disp" = disp)

				fm <- lmer(disp ~ freq + (1 | words), data = d)

				print(file_name)
				print(coef(summary(fm))[2])
				print(r.squaredGLMM(fm)[1])

				results[i,j,k,1] <- coef(summary(fm))[2]
				results[i,j,k,2] <- r.squaredGLMM(fm)[1]

			}
			k <- k + 1
		}
		j <- j + 1
	} 
	i <- i + 1
}

np$save("analysis_results", results)





