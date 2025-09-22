# 9) 500 Randomly Sampled Articles for Manual Classification

library(readr)
library(dplyr)

in_file   <- "1.csv"                # your combined file
out_samp  <- "sampled_500.csv"      # sampled output
out_rest  <- "remaining.csv"        # remainder output
set.seed(67)                        # reproducibility

df <- read_csv(in_file, show_col_types = FALSE)

sampled_idx <- sample(nrow(df), size = min(500, nrow(df)))
sampled <- df[sampled_idx, ]
remaining <- df[-sampled_idx, ]

write_csv(sampled, out_samp)
write_csv(remaining, out_rest)

cat("Sampled", nrow(sampled), "rows →", out_samp, "\n")
cat("Remaining", nrow(remaining), "rows →", out_rest, "\n")
