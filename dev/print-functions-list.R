suppressPackageStartupMessages({
  library(roxygen2)
  library(devtools)
  library(tidyverse)
})

documented_functions <- function(path = ".") {
  blocks <- roxygen2::parse_package(path)

  rows <- lapply(blocks, function(block) {
    # Function name
    name <- block$object$alias
    name %||% return(NULL)
    # Extract @title
    title_tag <- Filter(function(x) x$tag == "title", block$tags)
    title <- if (length(title_tag)) title_tag[[1]]$val else NA_character_
    data.frame(
      name = name,
      description = title,
      stringsAsFactors = FALSE
    )
  })

  out <- do.call(rbind, Filter(Negate(is.null), rows))
  out[order(out$name), ]
}

devtools::load_all(quiet = TRUE)

docf <- documented_functions()
docf$documented <- TRUE

allf <- as.character(lsf.str(envir = getNamespace("Rmlx")))
allf <- data.frame(name = allf)

allf <- left_join(allf, docf, by = "name")
allf$exported <- allf$name %in% getNamespaceExports("Rmlx")
allf <- allf[! grepl("^cpp_", allf$name), ]

write_delim(allf, "dev/mlx-functions.txt")
