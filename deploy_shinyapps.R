# deploy_shinyapps.R
#
# The shinyapps.io account must already be connected through rsconnect.

required_packages <- c("rsconnect", "fs")

missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0) {
  install.packages(missing_packages)
}

library(rsconnect)
library(fs)

# ------------------------------------------------------------
# 1. Resolve and validate project root
# ------------------------------------------------------------

app_dir <- normalizePath(
  ".",
  winslash = "/",
  mustWork = TRUE
)

required_core_files <- c(
  "app.R",
  "R/plots.R"
)

missing_core_files <- required_core_files[
  !file.exists(file.path(app_dir, required_core_files))
]

if (length(missing_core_files) > 0) {
  stop(
    paste0(
      "Missing required application files:\n- ",
      paste(missing_core_files, collapse = "\n- ")
    )
  )
}

# ------------------------------------------------------------
# 2. Collect only files required by the dashboard
# ------------------------------------------------------------

processed_files <- list.files(
  path = file.path(app_dir, "data", "processed"),
  pattern = "\\.(csv|parquet)$",
  recursive = TRUE,
  full.names = FALSE,
  ignore.case = TRUE
)

processed_files <- file.path(
  "data",
  "processed",
  processed_files
)

table_files <- list.files(
  path = file.path(app_dir, "outputs", "tables"),
  pattern = "\\.(csv|parquet|txt)$",
  recursive = TRUE,
  full.names = FALSE,
  ignore.case = TRUE
)

table_files <- file.path(
  "outputs",
  "tables",
  table_files
)

app_files <- unique(
  c(
    "app.R",
    "R/plots.R",
    processed_files,
    table_files
  )
)

# Convert Windows backslashes to deployment-safe forward slashes.
app_files <- gsub(
  "\\\\",
  "/",
  app_files
)

# Remove any paths that do not currently exist.
app_files <- app_files[
  file.exists(file.path(app_dir, app_files))
]

# ------------------------------------------------------------
# 3. Validate essential analytical files
# ------------------------------------------------------------

essential_data <- c(
  "data/processed/master_dataset.parquet",
  "data/processed/song_embeddings.parquet",
  "data/processed/song_similarity_pairs.parquet",
  "data/processed/song_network_nodes.parquet",
  "data/processed/song_network_edges.parquet",
  "data/processed/song_network_metrics.parquet",
  "outputs/tables/era_comparison.csv",
  "outputs/tables/network_mst_edges.csv",
  "outputs/tables/temporal_song_distances.csv",
  "outputs/tables/top_anger_songs.csv",
  "outputs/tables/top_negative_songs.csv",
  "outputs/tables/top_positive_songs.csv",
  "outputs/tables/top_sadness_songs.csv"
)

missing_essential <- essential_data[
  !file.exists(file.path(app_dir, essential_data))
]

if (length(missing_essential) > 0) {
  warning(
    paste0(
      "Some expected dashboard files are missing:\n- ",
      paste(missing_essential, collapse = "\n- "),
      "\n\nDeployment will continue, but related panels may use fallbacks ",
      "or appear unavailable."
    )
  )
}

# ------------------------------------------------------------
# 4. Display deployment bundle
# ------------------------------------------------------------

cat("\nApplication directory:\n")
cat(app_dir, "\n")

cat("\nFiles selected for deployment:", length(app_files), "\n\n")
cat(paste0("  ", app_files, collapse = "\n"))
cat("\n\n")

bundle_size <- sum(
  file.info(file.path(app_dir, app_files))$size,
  na.rm = TRUE
)

cat(
  "Approximate bundle size:",
  round(bundle_size / 1024^2, 2),
  "MB\n\n"
)

# Detect R package dependencies before deployment.
dependencies <- rsconnect::appDependencies(
  appDir = app_dir,
  appFiles = app_files
)

cat("Detected R package dependencies:\n")
print(dependencies[, intersect(
  c("Package", "Version", "Source"),
  names(dependencies)
)])

# ------------------------------------------------------------
# 5. Deploy
# ------------------------------------------------------------

rsconnect::deployApp(
  appDir = app_dir,
  appFiles = app_files,
  appPrimaryDoc = "app.R",
  appName = "linkin-park-evolution",
  appTitle = "Linkin Park Evolution",
  forceUpdate = TRUE,
  launch.browser = TRUE
)