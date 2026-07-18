# ============================================================
# app.R
# Linkin Park Evolution — Shiny Dashboard
# ============================================================

library(shiny)
library(shinydashboard)
library(plotly)
library(DT)
library(dplyr)
library(tidyr)
library(readr)
library(scales)
library(stringr)
library(htmltools)

source("R/plots.R")

# ------------------------------------------------------------
# Defensive readers
# ------------------------------------------------------------

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0 || all(is.na(x))) y else x
}

safe_read_csv <- function(path) {
  if (is.null(path) || length(path) == 0 || is.na(path) || !file.exists(path)) {
    return(tibble())
  }

  readr::read_csv(
    path,
    show_col_types = FALSE,
    progress = FALSE
  )
}

safe_read_parquet <- function(path) {
  if (is.null(path) || length(path) == 0 || is.na(path) || !file.exists(path)) {
    return(tibble())
  }

  if (!requireNamespace("arrow", quietly = TRUE)) {
    warning(
      "Package 'arrow' is not installed; parquet input skipped: ",
      path
    )
    return(tibble())
  }

  arrow::read_parquet(path) |>
    as_tibble()
}

candidate_project_roots <- function() {
  current <- normalizePath(
    getwd(),
    winslash = "/",
    mustWork = FALSE
  )

  parent <- normalizePath(
    file.path(current, ".."),
    winslash = "/",
    mustWork = FALSE
  )

  unique(c(
    current,
    file.path(parent, "Linkin-Park-Evolution"),
    file.path(parent, "Linkin Park Evolution"),
    file.path(parent, "Linkin-Park-Evolution-main"),
    parent
  ))
}

resolve_project_file <- function(relative_path) {
  candidates <- file.path(
    candidate_project_roots(),
    relative_path
  )

  existing <- candidates[file.exists(candidates)]

  if (length(existing) == 0) {
    return(NA_character_)
  }

  normalizePath(
    existing[[1]],
    winslash = "/",
    mustWork = FALSE
  )
}

first_available <- function(paths, reader = safe_read_csv) {
  for (path in paths) {
    obj <- reader(path)

    if (nrow(obj) > 0 || ncol(obj) > 0) {
      attr(obj, "source_path") <- path
      return(obj)
    }
  }

  tibble()
}

read_processed_table <- function(stem) {
  parquet_path <- resolve_project_file(
    file.path("data", "processed", paste0(stem, ".parquet"))
  )

  csv_path <- resolve_project_file(
    file.path("data", "processed", paste0(stem, ".csv"))
  )

  first_available(
    c(parquet_path, csv_path),
    reader = function(path) {
      if (is.na(path)) {
        return(tibble())
      }

      if (grepl("\\.parquet$", path, ignore.case = TRUE)) {
        safe_read_parquet(path)
      } else {
        safe_read_csv(path)
      }
    }
  )
}

read_output_csv <- function(relative_path) {
  safe_read_csv(
    resolve_project_file(relative_path)
  )
}

read_first_output_csv <- function(relative_paths) {
  for (relative_path in relative_paths) {
    path <- resolve_project_file(relative_path)

    if (!is.na(path) && file.exists(path)) {
      obj <- safe_read_csv(path)

      if (nrow(obj) > 0 || ncol(obj) > 0) {
        attr(obj, "source_path") <- path
        return(obj)
      }
    }
  }

  tibble()
}

choose_join_keys <- function(left, right) {
  preferred <- list(
    c("master_track_id"),
    c("recording_id"),
    c("track_id"),
    c("album", "canonical_title"),
    c("album", "track_title"),
    c("album", "track_position")
  )

  for (keys in preferred) {
    if (
      all(keys %in% names(left))
      && all(keys %in% names(right))
    ) {
      return(keys)
    }
  }

  character(0)
}

append_missing_features <- function(base, extra) {
  if (nrow(base) == 0 || nrow(extra) == 0) {
    return(base)
  }

  keys <- choose_join_keys(base, extra)

  if (length(keys) == 0) {
    warning(
      "Could not find a safe join key for a processed feature table."
    )
    return(base)
  }

  missing_columns <- setdiff(
    names(extra),
    c(keys, names(base))
  )

  if (length(missing_columns) == 0) {
    return(base)
  }

  extra_reduced <- extra |>
    select(
      all_of(keys),
      all_of(missing_columns)
    ) |>
    distinct(
      across(all_of(keys)),
      .keep_all = TRUE
    )

  base |>
    left_join(
      extra_reduced,
      by = keys
    )
}

# ------------------------------------------------------------
# Load analytical outputs
# ------------------------------------------------------------

master_dataset <- read_processed_table("master_dataset")

# Defensive enrichment:
# The dashboard may live in a sibling folder while the analytical repo
# contains the processed tables. These joins also recover feature columns
# if an older master dataset was copied into the Shiny project.
lyrics_features <- read_processed_table("lyrics_features")
audio_features <- read_processed_table("audio_features_acousticbrainz")

message(
  "Lyrics feature table: ",
  nrow(lyrics_features),
  " rows | ",
  ncol(lyrics_features),
  " columns"
)

message(
  "Audio feature table: ",
  nrow(audio_features),
  " rows | ",
  ncol(audio_features),
  " columns"
)

master_dataset <- master_dataset |>
  append_missing_features(lyrics_features) |>
  append_missing_features(audio_features)

song_embeddings <- read_processed_table("song_embeddings")
song_similarity <- read_processed_table("song_similarity_pairs")
song_network_metrics <- read_processed_table("song_network_metrics")
song_clusters <- read_processed_table("song_clusters")

album_evolution <- read_output_csv(
  file.path("outputs", "tables", "album_evolution_full.csv")
)

temporal_transitions <- read_output_csv(
  file.path("outputs", "tables", "temporal_album_transitions.csv")
)

temporal_era_transition <- read_first_output_csv(
  c(
    file.path("outputs", "tables", "temporal_era_transition.csv"),
    file.path("outputs", "tables", "era_transition.csv"),
    file.path("outputs", "tables", "temporal_era_transitions.csv")
  )
)


era_comparison <- read_first_output_csv(
  c(
    file.path("outputs", "tables", "era_comparison.csv"),
    file.path("outputs", "tables", "temporal_era_comparison.csv")
  )
)

temporal_song_distances <- read_first_output_csv(
  c(
    file.path("outputs", "tables", "temporal_song_distances.csv"),
    file.path("outputs", "tables", "temporal_song_distance.csv"),
    file.path("outputs", "tables", "song_distance_distribution.csv"),
    file.path("outputs", "tables", "within_album_song_dispersion.csv")
  )
)


top_anger_songs <- read_output_csv(
  file.path("outputs", "tables", "top_anger_songs.csv")
)

top_negative_songs <- read_output_csv(
  file.path("outputs", "tables", "top_negative_songs.csv")
)

top_positive_songs <- read_output_csv(
  file.path("outputs", "tables", "top_positive_songs.csv")
)

top_sadness_songs <- read_output_csv(
  file.path("outputs", "tables", "top_sadness_songs.csv")
)

network_mst_edges <- read_first_output_csv(
  c(
    file.path("outputs", "tables", "network_mst_edges.csv"),
    file.path("outputs", "tables", "song_network_mst_edges.csv"),
    file.path("outputs", "tables", "mst_edges.csv")
  )
)

album_embedding_story <- read_output_csv(
  file.path("outputs", "tables", "album_embedding_story.csv")
)

numb_similarity <- read_output_csv(
  file.path("outputs", "tables", "numb_similarity.csv")
)

network_hubs <- read_output_csv(
  file.path("outputs", "tables", "network_hub_songs.csv")
)

network_bridges <- read_output_csv(
  file.path("outputs", "tables", "network_bridge_songs.csv")
)

network_communities <- read_output_csv(
  file.path("outputs", "tables", "network_communities.csv")
)

prediction_summary <- read_output_csv(
  file.path("outputs", "tables", "album_prediction_model_summary.csv")
)

prediction_errors <- read_output_csv(
  file.path("outputs", "tables", "album_prediction_errors.csv")
)

global_importance <- read_output_csv(
  file.path(
    "outputs",
    "tables",
    "interpretable_album_rules_global_importance.csv"
  )
)

album_signatures <- read_output_csv(
  file.path(
    "outputs",
    "tables",
    "interpretable_album_rules_album_signatures.csv"
  )
)

artist_fingerprint <- read_output_csv(
  file.path("outputs", "tables", "artist_fingerprint.csv")
)

artist_album_deviation <- read_output_csv(
  file.path("outputs", "tables", "artist_album_deviation.csv")
)

artist_emotional_balance <- read_output_csv(
  file.path("outputs", "tables", "artist_emotional_balance.csv")
)

message(
  "Dashboard master dataset: ",
  attr(master_dataset, "source_path") %||% "resolved/merged processed tables"
)

message(
  "Master rows: ",
  nrow(master_dataset),
  " | columns: ",
  ncol(master_dataset)
)

# ------------------------------------------------------------
# Dashboard palettes
# ------------------------------------------------------------

dashboard_palettes <- list(
  "Hybrid Theory Red" = list(
    primary = "#A61B1B",
    secondary = "#191919",
    accent = "#D9D9D9",
    bg = "#F4F4F4",
    text = "#111111"
  ),
  "Meteora Steel" = list(
    primary = "#3E5968",
    secondary = "#18252D",
    accent = "#B7C4CC",
    bg = "#EEF2F4",
    text = "#101820"
  ),
  "Midnight Signal" = list(
    primary = "#2D4059",
    secondary = "#111827",
    accent = "#D9A441",
    bg = "#F6F7F9",
    text = "#111827"
  ),
  "A Thousand Suns" = list(
    primary = "#B86B18",
    secondary = "#372A1B",
    accent = "#F2C14E",
    bg = "#FFF8EA",
    text = "#2B2118"
  ),
  "Living Things Circuit" = list(
    primary = "#2F6B5F",
    secondary = "#163832",
    accent = "#9BC5B5",
    bg = "#F0F7F5",
    text = "#102A26"
  ),
  "The Hunting Party" = list(
    primary = "#7C1F24",
    secondary = "#1A1A1A",
    accent = "#C9A227",
    bg = "#F6F3EE",
    text = "#151515"
  ),
  "One More Light" = list(
    primary = "#6C63A8",
    secondary = "#24213A",
    accent = "#C9C1ED",
    bg = "#F6F4FB",
    text = "#211F2D"
  ),
  "From Zero Neon" = list(
    primary = "#00A7A5",
    secondary = "#171A1D",
    accent = "#B7F171",
    bg = "#F1F7F6",
    text = "#121719"
  )
)

get_palette <- function(name) {
  dashboard_palettes[[name]] %||% dashboard_palettes[["Meteora Steel"]]
}

palette_css <- function(palette_name = "Meteora Steel") {
  p <- get_palette(palette_name)

  tags$style(HTML(sprintf("
    body, .content-wrapper, .right-side {
      background-color: %s !important;
      color: %s;
      overflow-x: hidden;
    }

    .main-header .logo,
    .main-header .navbar,
    .skin-blue .main-header .logo,
    .skin-blue .main-header .navbar,
    .skin-blue .main-header .logo:hover {
      background-color: %s !important;
    }

    .main-header .logo {
      font-weight: 800;
      letter-spacing: 0.02em;
      white-space: nowrap !important;
      width: 560px !important;
      text-align: left !important;
      padding-left: 56px !important;
      font-size: 18px !important;
    }

    .main-header .navbar {
      margin-left: 560px !important;
    }

    .main-header .navbar .sidebar-toggle,
    .main-header .sidebar-toggle {
      position: fixed !important;
      left: 8px !important;
      top: 8px !important;
      width: 42px !important;
      height: 42px !important;
      z-index: 20000 !important;
      background: transparent !important;
      text-align: center !important;
    }

    .skin-blue .main-sidebar,
    .skin-blue .left-side,
    .main-sidebar {
      background-color: %s !important;
    }

    .skin-blue .sidebar-menu > li > a {
      color: #FFFFFF !important;
      border-left: 4px solid transparent;
    }

    .skin-blue .sidebar-menu > li:hover > a,
    .skin-blue .sidebar-menu > li.active > a {
      background-color: rgba(255,255,255,0.09) !important;
      color: #FFFFFF !important;
      border-left-color: %s !important;
    }

    .box {
      border-radius: 10px;
      border-top: 3px solid %s;
      box-shadow: 0 2px 8px rgba(0,0,0,0.09);
      background: #FFFFFF;
    }

    .box.box-primary {
      border-top-color: %s;
    }

    .box.box-solid.box-primary > .box-header {
      background-color: %s;
      border-color: %s;
    }

    .source-banner,
    .explain-box {
      background: #FFFFFF;
      border-left: 5px solid %s;
      border-radius: 8px;
      padding: 12px 16px;
      margin-bottom: 15px;
      box-shadow: 0 1px 5px rgba(0,0,0,0.07);
      font-size: 14px;
      line-height: 1.5;
    }

    .source-banner strong,
    .explain-box strong {
      color: %s;
    }

    .value-box {
      border-radius: 10px;
      overflow: hidden;
      box-shadow: 0 2px 8px rgba(0,0,0,0.10);
    }

    .small-box.bg-blue,
    .small-box.bg-purple,
    .small-box.bg-green,
    .small-box.bg-yellow,
    .small-box.bg-red {
      background-color: %s !important;
    }

    .sidebar-control-block {
      padding: 12px 15px;
      border-top: 1px solid rgba(255,255,255,0.12);
    }

    .sidebar-control-block .shiny-input-container,
    .sidebar-control-block .selectize-control,
    .sidebar-control-block .selectize-input {
      width: 100%% !important;
      max-width: 100%% !important;
      box-sizing: border-box;
    }

    .observations-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }

    .observation-card {
      background: #FFFFFF;
      border-left: 5px solid %s;
      border-radius: 10px;
      padding: 14px 16px;
      box-shadow: 0 1px 5px rgba(0,0,0,0.08);
      min-height: 158px;
    }

    .observation-theme {
      color: %s;
      font-weight: 800;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 8px;
    }

    .observation-title {
      font-size: 15px;
      font-weight: 700;
      line-height: 1.35;
      margin-bottom: 10px;
    }

    .observation-why {
      font-size: 13px;
      color: #4B5563;
      line-height: 1.4;
    }

    .metric-chip {
      display: inline-block;
      margin: 4px 6px 4px 0;
      padding: 5px 9px;
      border-radius: 999px;
      background: %s;
      color: #FFFFFF;
      font-size: 12px;
      font-weight: 700;
    }

    .dataTables_wrapper {
      font-size: 13px;
    }

    @media (max-width: 1100px) {
      .observations-grid { grid-template-columns: 1fr; }
    }
  ",
  p$bg, p$text, p$primary, p$secondary, p$accent,
  p$primary, p$primary, p$primary, p$primary,
  p$primary, p$primary, p$primary, p$accent,
  p$primary, p$primary, p$secondary, p$accent, p$primary
  )))
}

# ------------------------------------------------------------
# UI helpers
# ------------------------------------------------------------

format_num <- function(x) {
  scales::comma(as.numeric(x), accuracy = 1)
}

value_box <- function(value, subtitle, icon_name, color = "blue") {
  shinydashboard::valueBox(
    value = value,
    subtitle = subtitle,
    icon = icon(icon_name),
    color = color,
    width = 3
  )
}

how_to_box <- function(title, ..., width = 12) {
  box(
    width = width,
    title = title,
    status = "primary",
    solidHeader = TRUE,
    class = "compact-explain",
    div(class = "explain-box", ...)
  )
}

info_banner <- function() {
  div(
    class = "source-banner",
    strong("Linkin Park Evolution"), br(),
    "Exploring almost three decades of lyrics, emotions, themes, audio characteristics and musical evolution with data science.",
    br(),
    "Coverage: eight studio albums, 97 catalogue tracks, 96 lyric-bearing tracks and 86 tracks with AcousticBrainz features."
  )
}

observations_df <- function() {
  tibble::tibble(
    theme = c(
      "Return to origin",
      "Closest album pair",
      "Largest rupture",
      "Fastest evolution",
      "Album cohesion",
      "Numb similarity",
      "Predictive signal",
      "Network structure",
      "Artist fingerprint",
      "Emotional volatility",
      "Early-era aggression",
      "Late-era relaxation",
      "Experimental isolation",
      "Era transition",
      "Methodological caution"
    ),
    observation = c(
      "From Zero is closest to Hybrid Theory in the two-dimensional hybrid PCA album space.",
      "Living Things and Minutes to Midnight form the closest album-centroid pair in the embedding story.",
      "The Hunting Party to One More Light is the largest absolute album-profile shift.",
      "A Thousand Suns to Living Things has the highest estimated evolution velocity.",
      "One More Light is the most cohesive album in PCA space, while Meteora is the most internally diverse.",
      "By emotion, By Myself is the closest neighbour to Numb; by audio, Faint is the closest match. Figure.09 and Lost in the Echo also appear among its strongest audio neighbours.",
      "Audio and hybrid models identify albums far better than lyrics alone: macro F1 is about 0.50 versus about 0.23.",
      "Heavy Is the Crown and Two Faced are major hubs; Drawbar is the strongest bridge across similarity communities.",
      "Writing structure is more stable than emotional content. Lexical density, vocabulary structure and repetition form the strongest internal artist fingerprint.",
      "Emotional features vary much more across albums than lexical structure, especially VADER compound and conflict-related themes.",
      "Hybrid Theory and Meteora occupy the catalogue's most aggressive audio region, with aggression declining sharply in later albums.",
      "The Hunting Party is unexpectedly high in relaxed audio probability despite its heavy surface style, suggesting classifier mood and perceived genre are not identical.",
      "A Thousand Suns is the most internally diverse album and separates strongly in the nonlinear embeddings, consistent with its experimental construction.",
      "The Emily-era profile shows substantially longer lyrics and much lower Flesch reading ease, while most emotion-ratio changes are comparatively modest.",
      "From Zero lacks AcousticBrainz coverage, so lyrics-led conclusions are stronger than audio-led comparisons for the Emily era."
    ),
    why_it_matters = c(
      "The new era appears to reconnect with the band's earliest linguistic and emotional profile, although the audio side is unavailable for direct confirmation.",
      "It suggests that Living Things may represent a return toward the melodic and personal profile of Minutes to Midnight.",
      "The data supports the common perception that One More Light was a major stylistic departure.",
      "This transition combines a large profile change with only a two-year release gap.",
      "Cohesion and diversity are method-dependent, but they reveal how tightly each album occupies its own feature region.",
      "Different feature spaces answer different questions: lyrical emotion points toward By Myself, while sound characteristics point toward Faint.",
      "The sound of an album is more diagnostic than lyrical wording alone, while lyrics still provide meaningful thematic signal.",
      "Hub songs represent the catalogue's central sound; bridge songs connect otherwise distinct album or style communities.",
      "Linkin Park changes what it expresses more than how it structures songs and lyrics.",
      "The band's identity appears structurally stable but emotionally adaptive across releases.",
      "The early records have a distinctive audio signature that remains easy to recognize even when lyrical themes overlap with later work.",
      "Audio classifiers capture production and timbral cues, not simply heaviness, so results should be interpreted as model-based mood profiles.",
      "Its diversity is visible both within the album and in its position relative to neighbouring releases.",
      "The strongest era differences are stylistic and structural rather than a wholesale reversal of emotional vocabulary.",
      "Missingness is not neutral. It must remain visible in the dashboard and in any public interpretation."
    )
  )
}

observation_cards <- function() {
  df <- observations_df()

  tagList(
    lapply(seq_len(nrow(df)), function(i) {
      div(
        class = "observation-card",
        div(class = "observation-theme", df$theme[i]),
        div(class = "observation-title", df$observation[i]),
        div(
          class = "observation-why",
          strong("Why it matters: "),
          df$why_it_matters[i]
        )
      )
    })
  )
}

# ------------------------------------------------------------
# Dashboard UI
# ------------------------------------------------------------

ui <- dashboardPage(
  skin = "blue",

  dashboardHeader(
    title = "Linkin Park Evolution",
    titleWidth = 560
  ),

  dashboardSidebar(
    sidebarMenu(
      id = "tabs",
      menuItem("Evolution Brief", tabName = "brief", icon = icon("bolt")),
      menuItem("Album Evolution", tabName = "albums", icon = icon("compact-disc")),
      menuItem("Emotional Evolution", tabName = "emotions", icon = icon("heartbeat")),
      menuItem("Lyrics & Sentiment", tabName = "lyrics_sentiment", icon = icon("align-left")),
      menuItem("Song Explorer", tabName = "songs", icon = icon("music")),
      menuItem("Similarity & Network", tabName = "network", icon = icon("project-diagram")),
      menuItem("Embedding Atlas", tabName = "embedding_atlas", icon = icon("braille")),
      menuItem("Temporal Dynamics", tabName = "temporal_dynamics", icon = icon("stream")),
      menuItem("Top Emotion Songs", tabName = "top_emotion_songs", icon = icon("chart-bar")),
      menuItem("Network MST", tabName = "network_mst", icon = icon("sitemap")),
      menuItem("Artist DNA", tabName = "fingerprint", icon = icon("fingerprint")),
      menuItem("Fingerprint Lab", tabName = "fingerprint_lab", icon = icon("wave-square")),
      menuItem("Observations", tabName = "observations", icon = icon("lightbulb"))
    ),

    div(
      class = "sidebar-control-block",
      selectInput(
        "palette_choice",
        "Visual identity",
        choices = names(dashboard_palettes),
        selected = "Meteora Steel"
      )
    )
  ),

  dashboardBody(
    uiOutput("palette_css"),

    tabItems(
      tabItem(
        tabName = "brief",

        info_banner(),

        fluidRow(
          value_box(ifelse(nrow(master_dataset) > 0, format_num(nrow(master_dataset)), "97"), "Catalogue tracks", "music", "blue"),
          value_box(ifelse(nrow(album_evolution) > 0, format_num(nrow(album_evolution)), "8"), "Studio albums", "compact-disc", "purple"),
          value_box(ifelse(nrow(song_network_metrics) > 0, format_num(nrow(song_network_metrics)), "97"), "Network nodes", "project-diagram", "green"),
          value_box(ifelse(nrow(artist_fingerprint) > 0, format_num(nrow(artist_fingerprint)), "42"), "Fingerprint features", "fingerprint", "yellow")
        ),

        fluidRow(
          box(
            width = 7,
            title = "Chronological Album Trajectory",
            status = "primary",
            solidHeader = TRUE,
            plotlyOutput("brief_trajectory", height = "440px")
          ),
          box(
            width = 5,
            title = "Album Distance from Artist DNA",
            status = "primary",
            solidHeader = TRUE,
            plotlyOutput("brief_album_distance", height = "440px")
          )
        ),

        fluidRow(
          box(
            width = 6,
            title = "Largest Album-to-Album Shifts",
            status = "primary",
            solidHeader = TRUE,
            plotlyOutput("brief_shifts", height = "400px")
          ),
          box(
            width = 6,
            title = "Model Performance",
            status = "primary",
            solidHeader = TRUE,
            plotlyOutput("brief_models", height = "400px")
          )
        )
      ),

      tabItem(
        tabName = "albums",

        fluidRow(
          how_to_box(
            "How to read album evolution",
            p("Each album is represented by the average profile of its songs."),
            p("Distances show how much the combined lyrical, emotional and audio profile changed between releases."),
            p("Evolution velocity divides profile distance by years elapsed, highlighting rapid stylistic turns.")
          )
        ),

        fluidRow(
          box(
            width = 8,
            title = "Album Evolution Velocity",
            status = "primary",
            solidHeader = TRUE,
            plotlyOutput("album_velocity", height = "450px")
          ),
          box(
            width = 4,
            title = "Album Controls",
            status = "primary",
            solidHeader = TRUE,
            selectInput(
              "album_choice",
              "Select album",
              choices = if ("album" %in% names(master_dataset)) unique(master_dataset$album) else character(0)
            ),
            uiOutput("album_profile_chips")
          )
        ),

        fluidRow(
          box(
            width = 7,
            title = "Album Embedding Story",
            status = "primary",
            solidHeader = TRUE,
            plotlyOutput("album_embedding_plot", height = "500px")
          ),
          box(
            width = 5,
            title = "Album Summary",
            status = "primary",
            solidHeader = TRUE,
            DTOutput("album_story_table")
          )
        )
      ),

      tabItem(
        tabName = "emotions",

        fluidRow(
          how_to_box(
            "How to read emotional evolution",
            p("Album values are averages across eligible songs in each studio album."),
            p("NRC measures the share of emotion-associated words; AcousticBrainz moods are classifier probabilities derived from audio."),
            p("From Zero has no AcousticBrainz coverage, so the audio-mood chart ends with One More Light while lyrics emotions include all eight albums.")
          )
        ),

        fluidRow(
          box(
            width = 12,
            title = "AcousticBrainz Mood Evolution",
            status = "primary",
            solidHeader = TRUE,
            plotlyOutput("audio_mood_evolution", height = "520px")
          )
        ),

        fluidRow(
          box(
            width = 12,
            title = "NRC Emotion Evolution",
            status = "primary",
            solidHeader = TRUE,
            plotlyOutput("nrc_emotion_evolution", height = "540px")
          )
        )
      ),

      tabItem(
        tabName = "lyrics_sentiment",

        fluidRow(
          how_to_box(
            "How to read lyrics and sentiment",
            p("Lyrics-style metrics describe structure rather than topic: lexical diversity, repetition, sentence length and negation."),
            p("VADER compound ranges from negative to positive; positive and negative scores are separate proportions and need not sum to one with compound."),
            p("Album averages reveal broad catalogue movement, not the emotional meaning of every individual song.")
          )
        ),

        fluidRow(
          box(
            width = 12,
            title = "Lyrics-Style Evolution",
            status = "primary",
            solidHeader = TRUE,
            plotlyOutput("lyrics_style_evolution", height = "540px")
          )
        ),

        fluidRow(
          box(
            width = 12,
            title = "Sentiment Evolution",
            status = "primary",
            solidHeader = TRUE,
            plotlyOutput("sentiment_evolution", height = "520px")
          )
        )
      ),

      tabItem(
        tabName = "songs",

        fluidRow(
          column(
            width = 3,
            box(
              width = 12,
              title = "Song selector",
              status = "primary",
              solidHeader = TRUE,
              selectizeInput(
                "song_choice",
                "Song",
                choices = if ("track_title" %in% names(master_dataset)) sort(unique(master_dataset$track_title)) else character(0),
                options = list(placeholder = "Choose a song")
              ),
              selectInput(
                "similarity_model",
                "Similarity model",
                choices = c(
                  "Hybrid" = "hybrid",
                  "Emotion" = "emotion",
                  "Lyrics style" = "lyrics_style",
                  "Audio" = "audio"
                ),
                selected = "hybrid"
              ),
              sliderInput(
                "top_n",
                "Nearest songs",
                min = 3,
                max = 15,
                value = 8,
                step = 1
              )
            ),
            how_to_box(
              "How to read the song explorer",
              p("Similarity is calculated from standardized feature vectors."),
              p("A high score means two songs occupy similar positions in the selected feature space."),
              p("Audio is unavailable for From Zero, so hybrid results for that album are lyrics-led.")
            )
          ),

          column(
            width = 9,
            box(
              width = 12,
              title = "Nearest Songs",
              status = "primary",
              solidHeader = TRUE,
              plotlyOutput("song_similarity_plot", height = "480px")
            )
          )
        ),

        fluidRow(
          box(
            width = 6,
            title = "Selected Song Profile",
            status = "primary",
            solidHeader = TRUE,
            DTOutput("song_profile_table")
          ),
          box(
            width = 6,
            title = "Nearest-Neighbour Table",
            status = "primary",
            solidHeader = TRUE,
            DTOutput("song_similarity_table")
          )
        )
      ),

      tabItem(
        tabName = "network",

        fluidRow(
          how_to_box(
            "How to read the similarity network",
            p("Each node is a song and each edge represents a strong similarity relationship."),
            p("Hub score captures catalogue centrality. Bridge score identifies songs linking otherwise separate communities."),
            p("Community detection groups songs by network structure rather than by album labels.")
          )
        ),

        fluidRow(
          box(
            width = 7,
            title = "Song Similarity Network",
            status = "primary",
            solidHeader = TRUE,
            plotlyOutput("network_scatter", height = "540px")
          ),
          box(
            width = 5,
            title = "Network Leaders",
            status = "primary",
            solidHeader = TRUE,
            tabBox(
              width = 12,
              tabPanel("Hubs", DTOutput("hub_table")),
              tabPanel("Bridges", DTOutput("bridge_table")),
              tabPanel("Communities", DTOutput("community_table"))
            )
          )
        )
      ),

      tabItem(
        tabName = "embedding_atlas",

        fluidRow(
          how_to_box(
            "How to read the embedding atlas",
            p("Each point is a song projected from the high-dimensional hybrid feature space into two dimensions."),
            p("t-SNE emphasizes local neighbourhoods; UMAP preserves a broader mixture of local and global structure."),
            p("The album and era views use identical coordinates but different colour groupings, making continuity and separation easier to compare.")
          )
        ),

        fluidRow(
          box(
            width = 6,
            title = "t-SNE by Album",
            status = "primary",
            solidHeader = TRUE,
            plotlyOutput("tsne_by_album", height = "520px")
          ),
          box(
            width = 6,
            title = "t-SNE by Era",
            status = "primary",
            solidHeader = TRUE,
            plotlyOutput("tsne_by_era", height = "520px")
          )
        ),

        fluidRow(
          box(
            width = 6,
            title = "UMAP by Album",
            status = "primary",
            solidHeader = TRUE,
            plotlyOutput("umap_by_album", height = "520px")
          ),
          box(
            width = 6,
            title = "UMAP by Era",
            status = "primary",
            solidHeader = TRUE,
            plotlyOutput("umap_by_era", height = "520px")
          )
        )
      ),

      tabItem(
        tabName = "temporal_dynamics",

        fluidRow(
          how_to_box(
            "How to read temporal dynamics",
            p("Era transition compares the average Chester-era and Emily-era feature profiles."),
            p("Positive values indicate higher Emily-era averages; negative values indicate higher Chester-era averages."),
            p("Song dispersion measures each song's distance from its album centroid, revealing internal cohesion and outliers.")
          )
        ),

        fluidRow(
          box(
            width = 12,
            title = "Chester Era to Emily Era Transition",
            status = "primary",
            solidHeader = TRUE,
            plotlyOutput("era_transition_plot", height = "570px")
          )
        ),

        fluidRow(
          box(
            width = 12,
            title = "Chester Era vs Emily Era Comparison",
            status = "primary",
            solidHeader = TRUE,
            plotlyOutput("era_comparison_plot", height = "560px")
          )
        ),

        fluidRow(
          box(
            width = 12,
            title = "Within-Album Song Dispersion",
            status = "primary",
            solidHeader = TRUE,
            plotlyOutput("song_dispersion_plot", height = "560px")
          )
        )
      ),

      tabItem(
        tabName = "top_emotion_songs",

        fluidRow(
          how_to_box(
            "How to read top emotion songs",
            p("These rankings show the songs with the strongest values in four separate lyrical-emotion dimensions."),
            p("Anger and sadness come from NRC emotion ratios; positive and negative rankings come from VADER sentiment."),
            p("Scores are not interchangeable across panels, so compare rank and magnitude within each chart rather than across metrics.")
          )
        ),

        fluidRow(
          box(
            width = 6,
            title = "Highest Anger Scores",
            status = "primary",
            solidHeader = TRUE,
            plotlyOutput("top_anger_plot", height = "470px")
          ),
          box(
            width = 6,
            title = "Most Negative Songs",
            status = "primary",
            solidHeader = TRUE,
            plotlyOutput("top_negative_plot", height = "470px")
          )
        ),

        fluidRow(
          box(
            width = 6,
            title = "Most Positive Songs",
            status = "primary",
            solidHeader = TRUE,
            plotlyOutput("top_positive_plot", height = "470px")
          ),
          box(
            width = 6,
            title = "Highest Sadness Scores",
            status = "primary",
            solidHeader = TRUE,
            plotlyOutput("top_sadness_plot", height = "470px")
          )
        )
      ),

      tabItem(
        tabName = "network_mst",

        fluidRow(
          how_to_box(
            "How to read the minimum spanning tree",
            p("The minimum spanning tree connects all songs using only the essential similarity links and contains no cycles."),
            p("Long branches identify stylistically unusual songs; junctions reveal songs that connect multiple parts of the catalogue."),
            p("Node colours represent albums, while hover text exposes song and edge details.")
          )
        ),

        fluidRow(
          box(
            width = 12,
            title = "Linkin Park Similarity Minimum Spanning Tree",
            status = "primary",
            solidHeader = TRUE,
            plotlyOutput("network_mst_plot", height = "980px")
          )
        )
      ),

      tabItem(
        tabName = "fingerprint",

        fluidRow(
          how_to_box(
            "How to read Artist DNA",
            p("The fingerprint identifies features that are both present and stable across the studio catalogue."),
            p("This is an internal catalogue fingerprint, not proof that the traits are unique to Linkin Park."),
            p("Album distance shows how far each release moves from the band's overall internal centroid.")
          )
        ),

        fluidRow(
          box(
            width = 7,
            title = "Core Artist Fingerprint",
            status = "primary",
            solidHeader = TRUE,
            plotlyOutput("fingerprint_plot", height = "500px")
          ),
          box(
            width = 5,
            title = "Emotional Balance",
            status = "primary",
            solidHeader = TRUE,
            plotlyOutput("emotional_balance_plot", height = "500px")
          )
        ),

        fluidRow(
          box(
            width = 7,
            title = "Fingerprint Features",
            status = "primary",
            solidHeader = TRUE,
            DTOutput("fingerprint_table")
          ),
          box(
            width = 5,
            title = "Album Deviation",
            status = "primary",
            solidHeader = TRUE,
            DTOutput("album_deviation_table")
          )
        )
      ),

      tabItem(
        tabName = "fingerprint_lab",

        fluidRow(
          how_to_box(
            "How to read the fingerprint lab",
            p("The radar summarizes selected high-scoring internal fingerprint dimensions on a common 0-1 scale."),
            p("Feature stability measures how consistently a trait appears across albums; high stability does not imply uniqueness versus other artists."),
            p("Use the bar chart together with the confidence and fingerprint tables on the Artist DNA page.")
          )
        ),

        fluidRow(
          box(
            width = 6,
            title = "Artist Fingerprint Radar",
            status = "primary",
            solidHeader = TRUE,
            plotlyOutput("fingerprint_radar", height = "560px")
          ),
          box(
            width = 6,
            title = "Feature Stability",
            status = "primary",
            solidHeader = TRUE,
            plotlyOutput("feature_stability_plot", height = "560px")
          )
        ),

        fluidRow(
          box(
            width = 12,
            title = "Stable and Variable Feature Detail",
            status = "primary",
            solidHeader = TRUE,
            DTOutput("fingerprint_lab_table")
          )
        )
      ),

      tabItem(
        tabName = "observations",

        fluidRow(
          how_to_box(
            "Fifteen findings from the analysis",
            p("These observations summarize the strongest and most interpretable results from the current analytical pipeline."),
            p("They are descriptive findings from a small catalogue dataset and should not be presented as causal claims."),
            p("Audio coverage excludes From Zero, which is explicitly noted where relevant.")
          )
        ),

        fluidRow(
          box(
            width = 12,
            title = "Key Observations",
            status = "primary",
            solidHeader = TRUE,
            div(class = "observations-grid", observation_cards())
          )
        )
      )
    )
  )
)

# ------------------------------------------------------------
# Server
# ------------------------------------------------------------

server <- function(input, output, session) {

  active_palette <- reactive(
    get_palette(input$palette_choice)
  )

  output$palette_css <- renderUI(
    palette_css(input$palette_choice)
  )

  output$brief_trajectory <- renderPlotly({
    plot_album_trajectory(
      album_embedding_story,
      active_palette()
    )
  })

  output$brief_album_distance <- renderPlotly({
    plot_album_distance(
      artist_album_deviation,
      active_palette()
    )
  })

  output$brief_shifts <- renderPlotly({
    plot_temporal_shifts(
      temporal_transitions,
      active_palette(),
      n = 7
    )
  })

  output$brief_models <- renderPlotly({
    plot_model_performance(
      prediction_summary,
      active_palette()
    )
  })

  output$album_velocity <- renderPlotly({
    plot_evolution_velocity(
      temporal_transitions,
      active_palette()
    )
  })

  output$album_embedding_plot <- renderPlotly({
    plot_album_trajectory(
      album_embedding_story,
      active_palette(),
      show_arrows = TRUE
    )
  })

  output$album_story_table <- renderDT({
    cols <- intersect(
      c(
        "album_order",
        "album",
        "release_year",
        "era",
        "nearest_album",
        "nearest_album_distance",
        "distance_from_previous",
        "velocity_from_previous",
        "pca_dispersion"
      ),
      names(album_embedding_story)
    )

    make_dt(
      album_embedding_story |> select(all_of(cols)),
      page_length = 8
    )
  }, server = FALSE)

  output$album_profile_chips <- renderUI({
    req(input$album_choice)

    row <- album_embedding_story |>
      filter(album == input$album_choice) |>
      slice_head(n = 1)

    if (nrow(row) == 0) return(NULL)

    values <- list(
      `Release year` = row$release_year %||% NA,
      `Era` = row$era %||% NA,
      `Nearest album` = row$nearest_album %||% NA,
      `PCA dispersion` = round(row$pca_dispersion %||% NA, 3)
    )

    tagList(
      lapply(names(values), function(name) {
        div(
          class = "metric-chip",
          paste0(name, ": ", values[[name]])
        )
      })
    )
  })

  output$audio_mood_evolution <- renderPlotly({
    plot_audio_mood_evolution(
      audio_features,
      active_palette()
    )
  })

  output$nrc_emotion_evolution <- renderPlotly({
    plot_nrc_emotion_evolution(
      lyrics_features,
      active_palette()
    )
  })

  output$lyrics_style_evolution <- renderPlotly({
    plot_lyrics_style_evolution(
      lyrics_features,
      active_palette()
    )
  })

  output$sentiment_evolution <- renderPlotly({
    plot_sentiment_evolution(
      lyrics_features,
      active_palette()
    )
  })

  selected_song <- reactive({
    req(input$song_choice)
    master_dataset |>
      filter(track_title == input$song_choice) |>
      slice_head(n = 1)
  })

  similarity_rows <- reactive({
    req(input$song_choice)

    resolve_similarity_rows(
      similarity_df = song_similarity,
      selected_song = input$song_choice,
      model = input$similarity_model,
      top_n = input$top_n
    )
  })

  output$song_similarity_plot <- renderPlotly({
    plot_song_similarity(
      similarity_rows(),
      selected_song = input$song_choice,
      palette = active_palette()
    )
  })

  output$song_similarity_table <- renderDT({
    make_dt(
      similarity_rows(),
      page_length = input$top_n
    )
  }, server = FALSE)

  output$song_profile_table <- renderDT({
    row <- selected_song()

    preferred <- intersect(
      c(
        "track_title",
        "album",
        "release_year",
        "era",
        "track_position",
        "track_length_ms",
        "lyrics_word_count",
        "lyrics_vader_compound",
        "lyrics_nrc_anger_ratio",
        "lyrics_nrc_joy_ratio",
        "lyrics_nrc_fear_ratio",
        "lyrics_nrc_sadness_ratio",
        "lyrics_nrc_trust_ratio",
        "lyrics_theme_pain_ratio",
        "lyrics_theme_hope_ratio",
        "lyrics_theme_isolation_ratio"
      ),
      names(row)
    )

    profile <- tibble(
      metric = preferred,
      value = vapply(
        preferred,
        function(column) as.character(row[[column]][1]),
        character(1)
      )
    )

    make_dt(profile, page_length = 20)
  }, server = FALSE)

  output$network_scatter <- renderPlotly({
    plot_network_embedding(
      embeddings = song_embeddings,
      metrics = song_network_metrics,
      palette = active_palette()
    )
  })

  output$hub_table <- renderDT({
    make_dt(network_hubs, page_length = 15)
  }, server = FALSE)

  output$bridge_table <- renderDT({
    make_dt(network_bridges, page_length = 15)
  }, server = FALSE)

  output$community_table <- renderDT({
    make_dt(network_communities, page_length = 10)
  }, server = FALSE)

  output$tsne_by_album <- renderPlotly({
    plot_embedding_space(
      embeddings = song_embeddings,
      method = "tsne",
      group_by = "album",
      palette = active_palette()
    )
  })

  output$tsne_by_era <- renderPlotly({
    plot_embedding_space(
      embeddings = song_embeddings,
      method = "tsne",
      group_by = "era",
      palette = active_palette()
    )
  })

  output$umap_by_album <- renderPlotly({
    plot_embedding_space(
      embeddings = song_embeddings,
      method = "umap",
      group_by = "album",
      palette = active_palette()
    )
  })

  output$umap_by_era <- renderPlotly({
    plot_embedding_space(
      embeddings = song_embeddings,
      method = "umap",
      group_by = "era",
      palette = active_palette()
    )
  })

  output$era_transition_plot <- renderPlotly({
    plot_era_transition(
      era_transition = temporal_era_transition,
      palette = active_palette(),
      n = 15,
      fallback_data = master_dataset
    )
  })

  output$era_comparison_plot <- renderPlotly({
    plot_era_comparison(
      era_comparison = era_comparison,
      palette = active_palette(),
      n = 16,
      fallback_data = master_dataset
    )
  })

  output$song_dispersion_plot <- renderPlotly({
    plot_song_dispersion(
      song_distances = temporal_song_distances,
      palette = active_palette(),
      fallback_embeddings = song_embeddings
    )
  })

  output$top_anger_plot <- renderPlotly({
    plot_top_song_ranking(
      top_anger_songs,
      title = "Top Anger Songs",
      metric_label = "NRC anger ratio",
      palette = active_palette(),
      n = 12
    )
  })

  output$top_negative_plot <- renderPlotly({
    plot_top_song_ranking(
      top_negative_songs,
      title = "Most Negative Songs",
      metric_label = "VADER negative score",
      palette = active_palette(),
      n = 12
    )
  })

  output$top_positive_plot <- renderPlotly({
    plot_top_song_ranking(
      top_positive_songs,
      title = "Most Positive Songs",
      metric_label = "VADER positive score",
      palette = active_palette(),
      n = 12
    )
  })

  output$top_sadness_plot <- renderPlotly({
    plot_top_song_ranking(
      top_sadness_songs,
      title = "Top Sadness Songs",
      metric_label = "NRC sadness ratio",
      palette = active_palette(),
      n = 12
    )
  })

  output$network_mst_plot <- renderPlotly({
    plot_similarity_mst(
      mst_edges = network_mst_edges,
      embeddings = song_embeddings,
      palette = active_palette(),
      similarity_df = song_similarity
    )
  })

  output$fingerprint_plot <- renderPlotly({
    plot_artist_fingerprint(
      artist_fingerprint,
      active_palette(),
      n = 15
    )
  })

  output$emotional_balance_plot <- renderPlotly({
    plot_emotional_balance(
      artist_emotional_balance,
      active_palette()
    )
  })

  output$fingerprint_table <- renderDT({
    preferred <- intersect(
      c(
        "feature",
        "catalogue_mean",
        "stability_score",
        "confidence_score",
        "fingerprint_score",
        "feature_role"
      ),
      names(artist_fingerprint)
    )

    make_dt(
      artist_fingerprint |>
        select(all_of(preferred)) |>
        arrange(desc(fingerprint_score)),
      page_length = 15
    )
  }, server = FALSE)

  output$fingerprint_radar <- renderPlotly({
    plot_fingerprint_radar(
      artist_fingerprint,
      active_palette(),
      n = 10
    )
  })

  output$feature_stability_plot <- renderPlotly({
    plot_feature_stability_detail(
      artist_fingerprint,
      active_palette(),
      n_stable = 12,
      n_variable = 8
    )
  })

  output$fingerprint_lab_table <- renderDT({
    preferred <- intersect(
      c(
        "feature",
        "catalogue_mean",
        "album_std",
        "stability_score",
        "confidence_score",
        "fingerprint_score",
        "feature_role"
      ),
      names(artist_fingerprint)
    )

    make_dt(
      artist_fingerprint |>
        select(all_of(preferred)) |>
        arrange(desc(stability_score)),
      page_length = 20
    )
  }, server = FALSE)

  output$album_deviation_table <- renderDT({
    preferred <- intersect(
      c(
        "album_order",
        "album",
        "release_year",
        "era",
        "weighted_distance_from_artist_fingerprint",
        "largest_feature_deviations"
      ),
      names(artist_album_deviation)
    )

    make_dt(
      artist_album_deviation |>
        select(all_of(preferred)),
      page_length = 8
    )
  }, server = FALSE)
}

shinyApp(ui, server)
