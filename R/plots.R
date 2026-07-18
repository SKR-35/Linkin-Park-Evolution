# ============================================================
# R/plots.R
# Plot and table helpers for Linkin Park Evolution dashboard
# ============================================================

library(plotly)
library(DT)
library(dplyr)
library(tidyr)
library(stringr)

# ------------------------------------------------------------
# Generic helpers
# ------------------------------------------------------------

make_dt <- function(
    df,
    page_length = 10,
    order_col = NULL,
    order_dir = "desc") {

  if (is.null(df) || ncol(df) == 0) {
    df <- tibble::tibble(
      message = "Required output is not available yet."
    )
  }

  options <- list(
    pageLength = page_length,
    scrollX = TRUE,
    autoWidth = TRUE,
    dom = "Blfrtip",
    buttons = list(
      list(
        extend = "csv",
        text = "Export CSV",
        filename = "linkin_park_evolution_export",
        exportOptions = list(
          modifier = list(
            page = "all",
            search = "applied",
            order = "applied"
          )
        )
      )
    )
  )

  if (!is.null(order_col)) {
    options$order <- list(
      list(order_col, order_dir)
    )
  }

  DT::datatable(
    df,
    rownames = FALSE,
    filter = "top",
    extensions = "Buttons",
    options = options
  )
}

plotly_config <- function(p) {
  plotly::config(
    p,
    displaylogo = FALSE,
    responsive = TRUE
  )
}

empty_plot <- function(title, message) {
  plotly::plot_ly() |>
    plotly::layout(
      title = title,
      xaxis = list(visible = FALSE),
      yaxis = list(visible = FALSE),
      annotations = list(
        list(
          x = 0.5,
          y = 0.5,
          xref = "paper",
          yref = "paper",
          text = message,
          showarrow = FALSE
        )
      )
    ) |>
    plotly_config()
}

safe_numeric <- function(x) {
  suppressWarnings(as.numeric(x))
}

pick_first <- function(df, candidates) {
  intersect(candidates, names(df))[1] %||% NA_character_
}

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0 || all(is.na(x))) y else x
}

# ------------------------------------------------------------
# Album views
# ------------------------------------------------------------

plot_album_trajectory <- function(
    album_story,
    palette,
    show_arrows = TRUE) {

  required <- c("album", "pca_x", "pca_y")
  if (!all(required %in% names(album_story)) || nrow(album_story) == 0) {
    return(
      empty_plot(
        "Chronological Album Trajectory",
        "album_embedding_story.csv is unavailable."
      )
    )
  }

  df <- album_story |>
    arrange(album_order) |>
    mutate(
      label = paste0(
        album_order,
        ". ",
        album,
        " (",
        release_year,
        ")"
      ),
      hover = paste0(
        "<b>", album, "</b>",
        "<br>Year: ", release_year,
        "<br>Era: ", era,
        "<br>Nearest album: ", nearest_album,
        "<br>Distance: ", round(nearest_album_distance, 3)
      )
    )

  p <- plotly::plot_ly(
    df,
    x = ~pca_x,
    y = ~pca_y,
    type = "scatter",
    mode = "lines+markers+text",
    text = ~label,
    textposition = "top center",
    hovertext = ~hover,
    hoverinfo = "text",
    line = list(
      color = palette$secondary,
      width = 2
    ),
    marker = list(
      color = palette$primary,
      size = 12,
      line = list(
        color = palette$accent,
        width = 1.5
      )
    )
  )

  if (show_arrows && nrow(df) > 1) {
    arrows <- lapply(
      2:nrow(df),
      function(i) {
        list(
          x = df$pca_x[i],
          y = df$pca_y[i],
          ax = df$pca_x[i - 1],
          ay = df$pca_y[i - 1],
          xref = "x",
          yref = "y",
          axref = "x",
          ayref = "y",
          showarrow = TRUE,
          arrowhead = 3,
          arrowsize = 1,
          arrowwidth = 1.3,
          arrowcolor = palette$accent
        )
      }
    )
  } else {
    arrows <- list()
  }

  p |>
    plotly::layout(
      title = "Chronological Album Trajectory — Hybrid PCA",
      xaxis = list(title = "PCA dimension 1"),
      yaxis = list(title = "PCA dimension 2"),
      annotations = arrows,
      margin = list(l = 70, r = 30, t = 65, b = 65)
    ) |>
    plotly_config()
}

plot_evolution_velocity <- function(
    transitions,
    palette) {

  required <- c(
    "from_album",
    "to_album",
    "evolution_velocity"
  )

  if (!all(required %in% names(transitions)) || nrow(transitions) == 0) {
    return(
      empty_plot(
        "Album Evolution Velocity",
        "temporal_album_transitions.csv is unavailable."
      )
    )
  }

  df <- transitions |>
    mutate(
      transition = paste0(
        from_album,
        " → ",
        to_album
      ),
      evolution_velocity = safe_numeric(
        evolution_velocity
      )
    ) |>
    arrange(evolution_velocity)

  plotly::plot_ly(
    df,
    x = ~evolution_velocity,
    y = ~reorder(
      transition,
      evolution_velocity
    ),
    type = "bar",
    orientation = "h",
    marker = list(
      color = palette$primary
    ),
    text = ~paste0(
      "Distance: ",
      round(profile_distance, 3),
      "<br>Years elapsed: ",
      years_elapsed,
      "<br>Velocity: ",
      round(evolution_velocity, 3)
    ),
    hoverinfo = "text"
  ) |>
    plotly::layout(
      title = "Album-to-Album Evolution Velocity",
      xaxis = list(
        title = "Standardized profile distance per year"
      ),
      yaxis = list(title = ""),
      margin = list(l = 220)
    ) |>
    plotly_config()
}

plot_temporal_shifts <- function(
    transitions,
    palette,
    n = 7) {

  required <- c(
    "from_album",
    "to_album",
    "profile_distance"
  )

  if (!all(required %in% names(transitions)) || nrow(transitions) == 0) {
    return(
      empty_plot(
        "Largest Album Shifts",
        "temporal_album_transitions.csv is unavailable."
      )
    )
  }

  df <- transitions |>
    mutate(
      transition = paste0(
        from_album,
        " → ",
        to_album
      ),
      profile_distance = safe_numeric(
        profile_distance
      )
    ) |>
    slice_max(
      profile_distance,
      n = n,
      with_ties = FALSE
    ) |>
    arrange(profile_distance)

  plotly::plot_ly(
    df,
    x = ~profile_distance,
    y = ~reorder(
      transition,
      profile_distance
    ),
    type = "bar",
    orientation = "h",
    marker = list(
      color = palette$primary
    ),
    hovertemplate = paste0(
      "Transition: %{y}",
      "<br>Profile distance: %{x:.3f}",
      "<extra></extra>"
    )
  ) |>
    plotly::layout(
      title = "Largest Absolute Album-Profile Shifts",
      xaxis = list(
        title = "Standardized profile distance"
      ),
      yaxis = list(title = ""),
      margin = list(l = 220)
    ) |>
    plotly_config()
}

plot_album_distance <- function(
    album_deviation,
    palette) {

  distance_col <- pick_first(
    album_deviation,
    c(
      "weighted_distance_from_artist_fingerprint",
      "unweighted_distance_from_artist_centroid"
    )
  )

  if (
    nrow(album_deviation) == 0
    || is.na(distance_col)
    || !"album" %in% names(album_deviation)
  ) {
    return(
      empty_plot(
        "Album Distance from Artist DNA",
        "artist_album_deviation.csv is unavailable."
      )
    )
  }

  df <- album_deviation |>
    mutate(
      distance = safe_numeric(
        .data[[distance_col]]
      )
    ) |>
    arrange(distance)

  plotly::plot_ly(
    df,
    x = ~distance,
    y = ~reorder(album, distance),
    type = "bar",
    orientation = "h",
    marker = list(
      color = palette$primary
    ),
    hovertemplate = paste0(
      "Album: %{y}",
      "<br>Distance: %{x:.3f}",
      "<extra></extra>"
    )
  ) |>
    plotly::layout(
      title = "Distance from the Internal Artist Fingerprint",
      xaxis = list(
        title = "Weighted standardized distance"
      ),
      yaxis = list(title = ""),
      margin = list(l = 180)
    ) |>
    plotly_config()
}

# ------------------------------------------------------------
# Prediction views
# ------------------------------------------------------------

plot_model_performance <- function(
    model_summary,
    palette) {

  required <- c(
    "task",
    "model",
    "mean_macro_f1"
  )

  if (!all(required %in% names(model_summary)) || nrow(model_summary) == 0) {
    return(
      empty_plot(
        "Album Prediction Performance",
        "album_prediction_model_summary.csv is unavailable."
      )
    )
  }

  df <- model_summary |>
    mutate(
      mean_macro_f1 = safe_numeric(
        mean_macro_f1
      ),
      label = paste0(
        task,
        " / ",
        model
      )
    ) |>
    group_by(task) |>
    slice_max(
      mean_macro_f1,
      n = 1,
      with_ties = FALSE
    ) |>
    ungroup() |>
    arrange(mean_macro_f1)

  plotly::plot_ly(
    df,
    x = ~mean_macro_f1,
    y = ~reorder(label, mean_macro_f1),
    type = "bar",
    orientation = "h",
    marker = list(
      color = palette$primary
    ),
    text = ~paste0(
      "Tracks: ", tracks,
      "<br>Classes: ", classes,
      "<br>Features: ", feature_count,
      "<br>Macro F1: ", round(mean_macro_f1, 3)
    ),
    hoverinfo = "text"
  ) |>
    plotly::layout(
      title = "Best Model by Prediction Task",
      xaxis = list(
        title = "Repeated-CV macro F1",
        range = c(0, 1)
      ),
      yaxis = list(title = ""),
      margin = list(l = 220)
    ) |>
    plotly_config()
}

# ------------------------------------------------------------
# Song similarity
# ------------------------------------------------------------

resolve_similarity_rows <- function(
    similarity_df,
    selected_song,
    model = "hybrid",
    top_n = 8) {

  if (nrow(similarity_df) == 0) {
    return(
      tibble::tibble(
        message = "song_similarity_pairs output is unavailable."
      )
    )
  }

  source_col <- pick_first(
    similarity_df,
    c(
      "source_track_title",
      "track_title_a",
      "song_a",
      "source_song",
      "track_1"
    )
  )

  target_col <- pick_first(
    similarity_df,
    c(
      "target_track_title",
      "track_title_b",
      "song_b",
      "target_song",
      "track_2"
    )
  )

  album_col <- pick_first(
    similarity_df,
    c(
      "target_album",
      "album_b",
      "similar_album",
      "album_2"
    )
  )

  score_candidates <- switch(
    model,
    emotion = c(
      "emotion_similarity",
      "similarity_emotion"
    ),
    lyrics_style = c(
      "lyrics_style_similarity",
      "similarity_lyrics_style"
    ),
    audio = c(
      "audio_similarity",
      "similarity_audio"
    ),
    c(
      "hybrid_similarity",
      "similarity_hybrid",
      "similarity"
    )
  )

  score_col <- pick_first(
    similarity_df,
    score_candidates
  )

  if (
    any(is.na(c(source_col, target_col, score_col)))
  ) {
    return(
      tibble::tibble(
        message = "Similarity column names could not be resolved."
      )
    )
  }

  forward <- similarity_df |>
    filter(.data[[source_col]] == selected_song) |>
    transmute(
      song = .data[[target_col]],
      album = if (!is.na(album_col)) .data[[album_col]] else NA_character_,
      similarity = safe_numeric(
        .data[[score_col]]
      )
    )

  reverse <- similarity_df |>
    filter(.data[[target_col]] == selected_song) |>
    transmute(
      song = .data[[source_col]],
      album = NA_character_,
      similarity = safe_numeric(
        .data[[score_col]]
      )
    )

  bind_rows(
    forward,
    reverse
  ) |>
    filter(song != selected_song) |>
    group_by(song) |>
    summarise(
      album = first(na.omit(album)) %||% NA_character_,
      similarity = max(
        similarity,
        na.rm = TRUE
      ),
      .groups = "drop"
    ) |>
    arrange(desc(similarity)) |>
    slice_head(n = top_n)
}

plot_song_similarity <- function(
    similarity_rows,
    selected_song,
    palette) {

  if (
    nrow(similarity_rows) == 0
    || !"similarity" %in% names(similarity_rows)
  ) {
    return(
      empty_plot(
        paste0("Songs Similar to ", selected_song),
        "No compatible similarity output was found."
      )
    )
  }

  df <- similarity_rows |>
    mutate(
      label = ifelse(
        is.na(album),
        song,
        paste0(song, " — ", album)
      )
    ) |>
    arrange(similarity)

  plotly::plot_ly(
    df,
    x = ~similarity,
    y = ~reorder(label, similarity),
    type = "bar",
    orientation = "h",
    marker = list(
      color = palette$primary
    ),
    hovertemplate = paste0(
      "Song: %{y}",
      "<br>Similarity: %{x:.3f}",
      "<extra></extra>"
    )
  ) |>
    plotly::layout(
      title = paste0(
        "Nearest Songs to ",
        selected_song
      ),
      xaxis = list(
        title = "Cosine similarity"
      ),
      yaxis = list(title = ""),
      margin = list(l = 220)
    ) |>
    plotly_config()
}

# ------------------------------------------------------------
# Network view
# ------------------------------------------------------------

plot_network_embedding <- function(
    embeddings,
    metrics,
    palette) {

  if (
    nrow(embeddings) == 0
    || !all(
      c(
        "hybrid_umap_x",
        "hybrid_umap_y"
      ) %in% names(embeddings)
    )
  ) {
    return(
      empty_plot(
        "Song Similarity Network",
        "song_embeddings.parquet with hybrid UMAP coordinates is unavailable."
      )
    )
  }

  join_col <- intersect(
    c(
      "master_track_id",
      "track_id",
      "canonical_title"
    ),
    intersect(
      names(embeddings),
      names(metrics)
    )
  )[1]

  df <- embeddings

  if (!is.na(join_col) && nrow(metrics) > 0) {
    df <- df |>
      left_join(
        metrics,
        by = join_col,
        suffix = c("", "_network")
      )
  }

  title_col <- pick_first(
    df,
    c(
      "track_title",
      "canonical_title",
      "recording_title"
    )
  )

  community_col <- pick_first(
    df,
    c(
      "community",
      "louvain_community",
      "cluster"
    )
  )

  size_col <- pick_first(
    df,
    c(
      "hub_score",
      "pagerank",
      "weighted_degree"
    )
  )

  df <- df |>
    mutate(
      song_label = if (!is.na(title_col)) .data[[title_col]] else "Song",
      community_value = if (!is.na(community_col)) as.factor(.data[[community_col]]) else factor("1"),
      marker_size = if (!is.na(size_col)) {
        x <- safe_numeric(.data[[size_col]])
        7 + 22 * (x - min(x, na.rm = TRUE)) / (
          max(x, na.rm = TRUE) - min(x, na.rm = TRUE) + 1e-9
        )
      } else {
        10
      },
      hover = paste0(
        "<b>", song_label, "</b>",
        if ("album" %in% names(df)) paste0("<br>Album: ", album) else "",
        if (!is.na(community_col)) paste0("<br>Community: ", .data[[community_col]]) else "",
        if ("pagerank" %in% names(df)) paste0("<br>PageRank: ", round(pagerank, 4)) else "",
        if ("hub_score" %in% names(df)) paste0("<br>Hub score: ", round(hub_score, 3)) else "",
        if ("bridge_score" %in% names(df)) paste0("<br>Bridge score: ", round(bridge_score, 3)) else ""
      )
    )

  plotly::plot_ly(
    df,
    x = ~hybrid_umap_x,
    y = ~hybrid_umap_y,
    type = "scatter",
    mode = "markers",
    color = ~community_value,
    marker = list(
      size = ~marker_size,
      opacity = 0.78,
      line = list(
        color = palette$secondary,
        width = 0.6
      )
    ),
    hovertext = ~hover,
    hoverinfo = "text"
  ) |>
    plotly::layout(
      title = "Song Communities in Hybrid UMAP Space",
      xaxis = list(title = "UMAP 1"),
      yaxis = list(title = "UMAP 2"),
      legend = list(
        title = list(text = "Community")
      )
    ) |>
    plotly_config()
}

# ------------------------------------------------------------
# Artist fingerprint
# ------------------------------------------------------------

plot_artist_fingerprint <- function(
    fingerprint,
    palette,
    n = 15) {

  required <- c(
    "feature",
    "fingerprint_score"
  )

  if (!all(required %in% names(fingerprint)) || nrow(fingerprint) == 0) {
    return(
      empty_plot(
        "Artist Fingerprint",
        "artist_fingerprint.csv is unavailable."
      )
    )
  }

  df <- fingerprint |>
    mutate(
      fingerprint_score = safe_numeric(
        fingerprint_score
      )
    ) |>
    slice_max(
      fingerprint_score,
      n = n,
      with_ties = FALSE
    ) |>
    arrange(fingerprint_score) |>
    mutate(
      label = feature |>
        str_remove("^lyrics_") |>
        str_replace_all("_", " ")
    )

  plotly::plot_ly(
    df,
    x = ~fingerprint_score,
    y = ~reorder(label, fingerprint_score),
    type = "bar",
    orientation = "h",
    marker = list(
      color = palette$primary
    ),
    text = ~paste0(
      "Feature: ", feature,
      "<br>Fingerprint: ", round(fingerprint_score, 3),
      "<br>Stability: ", round(stability_score, 3),
      "<br>Mean: ", round(catalogue_mean, 4),
      "<br>Role: ", feature_role
    ),
    hoverinfo = "text"
  ) |>
    plotly::layout(
      title = "Strongest Internal Artist-Fingerprint Features",
      xaxis = list(
        title = "Fingerprint score",
        range = c(0, 1)
      ),
      yaxis = list(title = ""),
      margin = list(l = 220)
    ) |>
    plotly_config()
}

plot_emotional_balance <- function(
    emotional_balance,
    palette) {

  required <- c(
    "balance",
    "normalized_balance"
  )

  if (
    !all(required %in% names(emotional_balance))
    || nrow(emotional_balance) == 0
  ) {
    return(
      empty_plot(
        "Emotional Balance",
        "artist_emotional_balance.csv is unavailable."
      )
    )
  }

  df <- emotional_balance |>
    mutate(
      normalized_balance = safe_numeric(
        normalized_balance
      ),
      label = balance |>
        str_replace_all("_", " ")
    ) |>
    arrange(normalized_balance)

  plotly::plot_ly(
    df,
    x = ~normalized_balance,
    y = ~reorder(label, normalized_balance),
    type = "bar",
    orientation = "h",
    marker = list(
      color = ifelse(
        df$normalized_balance >= 0,
        palette$primary,
        palette$secondary
      )
    ),
    text = ~paste0(
      "Balance: ", label,
      "<br>Score: ", round(normalized_balance, 3),
      "<br>Dominant side: ", dominant_side
    ),
    hoverinfo = "text"
  ) |>
    plotly::layout(
      title = "Catalogue Emotional Balance",
      xaxis = list(
        title = "Negative side ← balance → Positive side",
        range = c(-1, 1),
        zeroline = TRUE
      ),
      yaxis = list(title = ""),
      margin = list(l = 180)
    ) |>
    plotly_config()
}


# ------------------------------------------------------------
# Emotional and lyrical evolution
# ------------------------------------------------------------

# Resolve feature columns across pipeline versions.
normalize_feature_name <- function(x) {
  x |>
    tolower() |>
    str_replace_all("[^a-z0-9]+", "_") |>
    str_replace_all("^_+|_+$", "")
}

resolve_feature_column <- function(df, candidates) {
  if (ncol(df) == 0) {
    return(NA_character_)
  }

  columns <- names(df)
  normalized_columns <- normalize_feature_name(columns)
  normalized_candidates <- normalize_feature_name(candidates)

  # 1. Exact original-name match.
  exact <- intersect(candidates, columns)
  if (length(exact) > 0) {
    return(exact[[1]])
  }

  # 2. Exact normalized-name match.
  for (candidate in normalized_candidates) {
    hit <- which(normalized_columns == candidate)
    if (length(hit) > 0) {
      return(columns[[hit[[1]]]])
    }
  }

  # 3. Suffix match, useful for lyrics_/audio_ab_ prefixes and .x/.y joins.
  for (candidate in normalized_candidates) {
    hit <- which(
      endsWith(normalized_columns, candidate)
      | endsWith(normalized_columns, paste0(candidate, "_x"))
      | endsWith(normalized_columns, paste0(candidate, "_y"))
    )

    if (length(hit) > 0) {
      return(columns[[hit[[1]]]])
    }
  }

  # 4. Token containment fallback.
  for (candidate in normalized_candidates) {
    tokens <- unlist(str_split(candidate, "_"))
    tokens <- tokens[nchar(tokens) >= 3]

    if (length(tokens) == 0) {
      next
    }

    scores <- vapply(
      normalized_columns,
      function(column) sum(tokens %in% unlist(str_split(column, "_"))),
      numeric(1)
    )

    best <- which.max(scores)

    if (length(best) > 0 && scores[[best]] == length(tokens)) {
      return(columns[[best]])
    }
  }

  NA_character_
}

ensure_album_metadata <- function(df) {
  if (nrow(df) == 0) {
    return(df)
  }

  album_candidates <- c(
    "album",
    "album_name",
    "release_group_title"
  )

  album_col <- resolve_feature_column(
    df,
    album_candidates
  )

  if (is.na(album_col)) {
    return(df)
  }

  if (!"album" %in% names(df)) {
    df$album <- df[[album_col]]
  }

  album_levels <- c(
    "Hybrid Theory",
    "Meteora",
    "Minutes to Midnight",
    "A Thousand Suns",
    "Living Things",
    "The Hunting Party",
    "One More Light",
    "From Zero"
  )

  if (!"album_order" %in% names(df)) {
    df$album_order <- match(
      as.character(df$album),
      album_levels
    )
  }

  if (!"release_year" %in% names(df)) {
    release_years <- c(
      "Hybrid Theory" = 2000,
      "Meteora" = 2003,
      "Minutes to Midnight" = 2007,
      "A Thousand Suns" = 2010,
      "Living Things" = 2012,
      "The Hunting Party" = 2014,
      "One More Light" = 2017,
      "From Zero" = 2024
    )

    df$release_year <- unname(
      release_years[as.character(df$album)]
    )
  }

  df
}

album_ordered_summary <- function(
    df,
    feature_specs) {

  df <- ensure_album_metadata(df)

  if (
    nrow(df) == 0
    || !"album" %in% names(df)
  ) {
    return(tibble::tibble())
  }

  resolved <- vapply(
    feature_specs,
    function(candidates) {
      resolve_feature_column(
        df,
        candidates
      )
    },
    character(1)
  )

  available <- !is.na(resolved)

  if (!any(available)) {
    result <- tibble::tibble()
    attr(result, "loaded_columns") <- names(df)
    attr(result, "requested_features") <- names(feature_specs)
    return(result)
  }

  resolved <- resolved[available]
  labels <- names(resolved)

  working <- df |>
    mutate(
      across(
        all_of(unname(resolved)),
        safe_numeric
      )
    )

  id_cols <- intersect(
    c(
      "album_order",
      "album",
      "release_year",
      "era"
    ),
    names(working)
  )

  result <- working |>
    group_by(
      across(all_of(id_cols))
    ) |>
    summarise(
      across(
        all_of(unname(resolved)),
        ~ {
          values <- .x[is.finite(.x)]
          if (length(values) == 0) {
            NA_real_
          } else {
            mean(values)
          }
        }
      ),
      .groups = "drop"
    ) |>
    arrange(album_order)

  # Rename resolved source columns to stable chart keys.
  for (i in seq_along(resolved)) {
    names(result)[names(result) == resolved[[i]]] <- labels[[i]]
  }

  attr(result, "loaded_columns") <- names(df)
  attr(result, "resolved_columns") <- resolved

  result
}

plot_multi_line_album <- function(
    summary_df,
    feature_labels,
    title,
    y_title,
    palette,
    zero_line = FALSE) {

  available <- intersect(names(feature_labels), names(summary_df))

  if (nrow(summary_df) == 0 || length(available) == 0) {
    return(
      empty_plot(
        title,
        paste0(
          "Required album-level features are unavailable. ",
          "Available source columns did not match the requested metrics."
        )
      )
    )
  }

  long <- summary_df |>
    select(any_of(c("album_order", "album", "release_year")), all_of(available)) |>
    pivot_longer(
      cols = all_of(available),
      names_to = "feature",
      values_to = "value"
    ) |>
    mutate(
      series = unname(feature_labels[feature]),
      album = factor(album, levels = summary_df$album)
    )

  p <- plotly::plot_ly(
    long,
    x = ~album,
    y = ~value,
    color = ~series,
    colors = c(
      palette$primary,
      palette$secondary,
      palette$accent,
      "#8B5CF6",
      "#B45309",
      "#C2413B",
      "#4D7C0F",
      "#0F766E"
    ),
    type = "scatter",
    mode = "lines+markers",
    line = list(width = 2.4),
    marker = list(size = 8),
    hovertemplate = paste0(
      "Album: %{x}",
      "<br>Series: %{fullData.name}",
      "<br>Value: %{y:.4f}",
      "<extra></extra>"
    )
  )

  shapes <- list()
  if (zero_line) {
    shapes <- list(list(
      type = "line",
      x0 = -0.5,
      x1 = length(unique(long$album)) - 0.5,
      y0 = 0,
      y1 = 0,
      line = list(color = "#6B7280", width = 1, dash = "dot")
    ))
  }

  p |>
    layout(
      title = title,
      xaxis = list(title = "Studio album", tickangle = -25),
      yaxis = list(title = y_title),
      legend = list(orientation = "h", x = 0, y = 1.12),
      shapes = shapes,
      margin = list(l = 75, r = 25, t = 95, b = 115)
    ) |>
    plotly_config()
}

plot_audio_mood_evolution <- function(master_dataset, palette) {
  feature_specs <- list(
    aggressive = c(
      "audio_ab_mood_aggressive_probability",
      "audio_mood_aggressive_probability",
      "mood_aggressive_probability",
      "aggressive_probability",
      "mood_aggressive",
      "aggressive"
    ),
    happy = c(
      "audio_ab_mood_happy_probability",
      "audio_mood_happy_probability",
      "mood_happy_probability",
      "happy_probability",
      "mood_happy",
      "happy"
    ),
    relaxed = c(
      "audio_ab_mood_relaxed_probability",
      "audio_mood_relaxed_probability",
      "mood_relaxed_probability",
      "relaxed_probability",
      "mood_relaxed",
      "relaxed"
    ),
    sad = c(
      "audio_ab_mood_sad_probability",
      "audio_mood_sad_probability",
      "mood_sad_probability",
      "sad_probability",
      "mood_sad",
      "sad"
    )
  )

  labels <- c(
    aggressive = "Aggressive",
    happy = "Happy",
    relaxed = "Relaxed",
    sad = "Sad"
  )

  summary <- album_ordered_summary(
    master_dataset,
    feature_specs
  )

  plot_multi_line_album(
    summary,
    labels,
    "Linkin Park Album Evolution: AcousticBrainz Moods",
    "Average classifier probability",
    palette
  )
}

plot_nrc_emotion_evolution <- function(master_dataset, palette) {
  feature_specs <- list(
    anger = c("lyrics_nrc_anger_ratio", "nrc_anger_ratio", "nrc_anger"),
    fear = c("lyrics_nrc_fear_ratio", "nrc_fear_ratio", "nrc_fear"),
    joy = c("lyrics_nrc_joy_ratio", "nrc_joy_ratio", "nrc_joy"),
    sadness = c("lyrics_nrc_sadness_ratio", "nrc_sadness_ratio", "nrc_sadness"),
    trust = c("lyrics_nrc_trust_ratio", "nrc_trust_ratio", "nrc_trust"),
    anticipation = c(
      "lyrics_nrc_anticipation_ratio",
      "nrc_anticipation_ratio",
      "nrc_anticipation"
    )
  )

  labels <- c(
    anger = "Anger",
    fear = "Fear",
    joy = "Joy",
    sadness = "Sadness",
    trust = "Trust",
    anticipation = "Anticipation"
  )

  summary <- album_ordered_summary(
    master_dataset,
    feature_specs
  )

  plot_multi_line_album(
    summary,
    labels,
    "Linkin Park Album Evolution: NRC Emotions",
    "Average emotion ratio",
    palette
  )
}

plot_lyrics_style_evolution <- function(master_dataset, palette) {
  feature_specs <- list(
    type_token_ratio = c(
      "lyrics_type_token_ratio",
      "type_token_ratio"
    ),
    root_type_token_ratio = c(
      "lyrics_root_type_token_ratio",
      "root_type_token_ratio"
    ),
    line_repetition_ratio = c(
      "lyrics_line_repetition_ratio",
      "line_repetition_ratio"
    ),
    negation_ratio = c(
      "lyrics_negation_ratio",
      "negation_ratio"
    ),
    top_5_word_share = c(
      "lyrics_top_5_word_share",
      "top_5_word_share",
      "top5_word_share"
    )
  )

  labels <- c(
    type_token_ratio = "Type-token ratio",
    root_type_token_ratio = "Root type-token ratio",
    line_repetition_ratio = "Line repetition",
    negation_ratio = "Negation",
    top_5_word_share = "Top-five word share"
  )

  summary <- album_ordered_summary(
    master_dataset,
    feature_specs
  )

  plot_multi_line_album(
    summary,
    labels,
    "Linkin Park Album Evolution: Lyrics Style",
    "Album-average ratio",
    palette
  )
}

plot_sentiment_evolution <- function(master_dataset, palette) {
  feature_specs <- list(
    compound = c(
      "lyrics_vader_compound",
      "vader_compound",
      "sentiment_compound"
    ),
    negative = c(
      "lyrics_vader_negative",
      "vader_negative",
      "sentiment_negative",
      "vader_neg"
    ),
    positive = c(
      "lyrics_vader_positive",
      "vader_positive",
      "sentiment_positive",
      "vader_pos"
    )
  )

  labels <- c(
    compound = "Compound",
    negative = "Negative",
    positive = "Positive"
  )

  summary <- album_ordered_summary(
    master_dataset,
    feature_specs
  )

  plot_multi_line_album(
    summary,
    labels,
    "Linkin Park Album Evolution: Sentiment",
    "Average VADER score",
    palette,
    zero_line = TRUE
  )
}

# ------------------------------------------------------------
# Fingerprint detail views
# ------------------------------------------------------------

plot_fingerprint_radar <- function(
    fingerprint,
    palette,
    n = 10) {

  required <- c("feature", "fingerprint_score")
  if (nrow(fingerprint) == 0 || !all(required %in% names(fingerprint))) {
    return(empty_plot("Artist Fingerprint Radar", "artist_fingerprint.csv is unavailable."))
  }

  preferred <- c(
    "lyrics_type_token_ratio",
    "lyrics_line_repetition_ratio",
    "lyrics_nrc_fear_ratio",
    "lyrics_negation_ratio",
    "lyrics_nrc_sadness_ratio",
    "lyrics_nrc_anger_ratio",
    "lyrics_nrc_joy_ratio",
    "lyrics_theme_hope_ratio",
    "lyrics_nrc_trust_ratio",
    "lyrics_theme_isolation_ratio"
  )

  df <- fingerprint |>
    filter(feature %in% preferred) |>
    mutate(
      score = safe_numeric(fingerprint_score),
      label = feature |>
        str_remove("^lyrics_") |>
        str_replace_all("_ratio$", "") |>
        str_replace_all("_", " ")
    )

  if (nrow(df) < 3) {
    df <- fingerprint |>
      slice_max(safe_numeric(fingerprint_score), n = n, with_ties = FALSE) |>
      mutate(
        score = safe_numeric(fingerprint_score),
        label = feature |>
          str_remove("^lyrics_") |>
          str_replace_all("_", " ")
      )
  }

  df <- df |>
    arrange(match(feature, preferred))

  closed <- bind_rows(df, slice_head(df, n = 1))

  plotly::plot_ly(
    closed,
    type = "scatterpolar",
    r = ~score,
    theta = ~label,
    mode = "lines+markers",
    fill = "toself",
    line = list(color = palette$primary, width = 3),
    marker = list(color = palette$secondary, size = 7),
    fillcolor = scales::alpha(palette$primary, 0.25),
    hovertemplate = paste0(
      "Feature: %{theta}",
      "<br>Fingerprint score: %{r:.3f}",
      "<extra></extra>"
    )
  ) |>
    layout(
      title = "Linkin Park Internal Artist Fingerprint",
      polar = list(
        radialaxis = list(range = c(0, 1), tickformat = ".1f")
      ),
      showlegend = FALSE,
      margin = list(l = 75, r = 75, t = 80, b = 65)
    ) |>
    plotly_config()
}

plot_feature_stability_detail <- function(
    fingerprint,
    palette,
    n_stable = 12,
    n_variable = 8) {

  required <- c("feature", "stability_score")
  if (nrow(fingerprint) == 0 || !all(required %in% names(fingerprint))) {
    return(empty_plot("Feature Stability", "artist_fingerprint.csv is unavailable."))
  }

  stable <- fingerprint |>
    slice_max(safe_numeric(stability_score), n = n_stable, with_ties = FALSE) |>
    mutate(group = "Most stable")

  variable <- fingerprint |>
    slice_min(safe_numeric(stability_score), n = n_variable, with_ties = FALSE) |>
    mutate(group = "Most album-variable")

  df <- bind_rows(stable, variable) |>
    distinct(feature, .keep_all = TRUE) |>
    mutate(
      stability = safe_numeric(stability_score),
      label = feature |>
        str_remove("^lyrics_") |>
        str_replace_all("_", " ")
    ) |>
    arrange(stability)

  plotly::plot_ly(
    df,
    x = ~stability,
    y = ~reorder(label, stability),
    color = ~group,
    colors = c(palette$secondary, palette$primary),
    type = "bar",
    orientation = "h",
    hovertemplate = paste0(
      "Feature: %{y}",
      "<br>Stability: %{x:.3f}",
      "<br>Group: %{fullData.name}",
      "<extra></extra>"
    )
  ) |>
    layout(
      title = "Artist Feature Stability Across Albums",
      xaxis = list(title = "Stability score", range = c(0, 1)),
      yaxis = list(title = ""),
      barmode = "overlay",
      margin = list(l = 220, r = 25, t = 75, b = 60)
    ) |>
    plotly_config()
}


# ------------------------------------------------------------
# Embedding atlas
# ------------------------------------------------------------

resolve_embedding_columns <- function(
    embeddings,
    method = c("tsne", "umap")) {

  method <- match.arg(method)

  x_candidates <- if (method == "tsne") {
    c(
      "hybrid_tsne_x",
      "tsne_x",
      "hybrid_tsne_1",
      "tsne_1",
      "tsne1"
    )
  } else {
    c(
      "hybrid_umap_x",
      "umap_x",
      "hybrid_umap_1",
      "umap_1",
      "umap1"
    )
  }

  y_candidates <- if (method == "tsne") {
    c(
      "hybrid_tsne_y",
      "tsne_y",
      "hybrid_tsne_2",
      "tsne_2",
      "tsne2"
    )
  } else {
    c(
      "hybrid_umap_y",
      "umap_y",
      "hybrid_umap_2",
      "umap_2",
      "umap2"
    )
  }

  list(
    x = resolve_feature_column(
      embeddings,
      x_candidates
    ),
    y = resolve_feature_column(
      embeddings,
      y_candidates
    )
  )
}

plot_embedding_space <- function(
    embeddings,
    method = c("tsne", "umap"),
    group_by = c("album", "era"),
    palette) {

  method <- match.arg(method)
  group_by <- match.arg(group_by)

  coordinates <- resolve_embedding_columns(
    embeddings,
    method
  )

  title_col <- pick_first(
    embeddings,
    c(
      "track_title",
      "canonical_title",
      "recording_title"
    )
  )

  group_col <- resolve_feature_column(
    embeddings,
    c(group_by)
  )

  if (
    nrow(embeddings) == 0
    || is.na(coordinates$x)
    || is.na(coordinates$y)
    || is.na(group_col)
  ) {
    return(
      empty_plot(
        paste(
          toupper(method),
          "by",
          stringr::str_to_title(group_by)
        ),
        paste0(
          "Compatible ",
          method,
          " coordinates or grouping columns are unavailable."
        )
      )
    )
  }

  df <- embeddings |>
    transmute(
      x = safe_numeric(.data[[coordinates$x]]),
      y = safe_numeric(.data[[coordinates$y]]),
      song = if (!is.na(title_col)) {
        as.character(.data[[title_col]])
      } else {
        "Song"
      },
      album = if ("album" %in% names(embeddings)) {
        as.character(album)
      } else {
        NA_character_
      },
      era = if ("era" %in% names(embeddings)) {
        as.character(era)
      } else {
        NA_character_
      },
      group = as.factor(.data[[group_col]])
    ) |>
    filter(
      is.finite(x),
      is.finite(y),
      !is.na(group)
    ) |>
    mutate(
      hover = paste0(
        "<b>", song, "</b>",
        ifelse(
          is.na(album),
          "",
          paste0("<br>Album: ", album)
        ),
        ifelse(
          is.na(era),
          "",
          paste0("<br>Era: ", era)
        ),
        "<br>",
        toupper(method),
        " 1: ",
        round(x, 3),
        "<br>",
        toupper(method),
        " 2: ",
        round(y, 3)
      )
    )

  plotly::plot_ly(
    df,
    x = ~x,
    y = ~y,
    type = "scatter",
    mode = "markers",
    color = ~group,
    marker = list(
      size = 10,
      opacity = 0.82,
      line = list(
        color = palette$secondary,
        width = 0.5
      )
    ),
    hovertext = ~hover,
    hoverinfo = "text"
  ) |>
    plotly::layout(
      title = paste0(
        "Linkin Park Hybrid Space — ",
        toupper(method),
        " by ",
        stringr::str_to_title(group_by)
      ),
      xaxis = list(
        title = paste(
          toupper(method),
          "dimension 1"
        )
      ),
      yaxis = list(
        title = paste(
          toupper(method),
          "dimension 2"
        )
      ),
      legend = list(
        title = list(
          text = stringr::str_to_title(group_by)
        )
      ),
      margin = list(
        l = 70,
        r = 30,
        t = 70,
        b = 65
      )
    ) |>
    plotly_config()
}

# ------------------------------------------------------------
# Temporal dynamics detail
# ------------------------------------------------------------

plot_era_transition <- function(
    era_transition,
    palette,
    n = 15,
    fallback_data = NULL) {

  build_from_wide_or_raw <- function(df) {
    if (is.null(df) || nrow(df) == 0) {
      return(tibble::tibble())
    }

    era_col <- resolve_feature_column(
      df,
      c("era", "artist_era", "vocalist_era")
    )

    if (is.na(era_col)) {
      return(tibble::tibble())
    }

    numeric_candidates <- names(df)[vapply(
      df,
      function(x) is.numeric(x) || all(is.na(suppressWarnings(as.numeric(x))) == is.na(x)),
      logical(1)
    )]

    excluded <- c(
      "album_order", "release_year", "track_position",
      "track_length_ms", "duration_ms", "duration_seconds",
      "master_track_id", "track_id", "recording_id"
    )

    feature_cols <- setdiff(
      numeric_candidates,
      c(excluded, era_col)
    )

    if (length(feature_cols) == 0) {
      return(tibble::tibble())
    }

    working <- df |>
      mutate(
        across(
          all_of(feature_cols),
          safe_numeric
        ),
        .era_value = as.character(.data[[era_col]])
      )

    chester_rows <- grepl(
      "chester",
      working$.era_value,
      ignore.case = TRUE
    )

    emily_rows <- grepl(
      "emily",
      working$.era_value,
      ignore.case = TRUE
    )

    if (!any(chester_rows) || !any(emily_rows)) {
      return(tibble::tibble())
    }

    tibble::tibble(
      feature = feature_cols,
      difference = vapply(
        feature_cols,
        function(column) {
          emily_values <- working[[column]][emily_rows]
          chester_values <- working[[column]][chester_rows]

          emily_values <- emily_values[is.finite(emily_values)]
          chester_values <- chester_values[is.finite(chester_values)]

          if (length(emily_values) == 0 || length(chester_values) == 0) {
            NA_real_
          } else {
            mean(emily_values) - mean(chester_values)
          }
        },
        numeric(1)
      )
    ) |>
      filter(is.finite(difference))
  }

  long_df <- tibble::tibble()

  if (!is.null(era_transition) && nrow(era_transition) > 0) {
    feature_col <- pick_first(
      era_transition,
      c(
        "feature", "metric", "variable", "feature_name",
        "column", "measure"
      )
    )

    difference_col <- pick_first(
      era_transition,
      c(
        "difference", "mean_difference", "emily_minus_chester",
        "feature_change", "delta", "value", "change",
        "era_difference", "mean_diff"
      )
    )

    if (!is.na(feature_col) && !is.na(difference_col)) {
      long_df <- era_transition |>
        transmute(
          feature = as.character(.data[[feature_col]]),
          difference = safe_numeric(.data[[difference_col]])
        ) |>
        filter(is.finite(difference))
    } else {
      # Some pipeline versions write a one-row wide table where each
      # feature is a column and the value is Emily mean minus Chester mean.
      numeric_cols <- names(era_transition)[vapply(
        era_transition,
        function(x) {
          values <- suppressWarnings(as.numeric(x))
          any(is.finite(values))
        },
        logical(1)
      )]

      metadata_cols <- c(
        "from_era", "to_era", "era", "comparison",
        "from_year", "to_year", "years_elapsed"
      )

      numeric_cols <- setdiff(numeric_cols, metadata_cols)

      if (length(numeric_cols) > 0) {
        long_df <- era_transition |>
          select(all_of(numeric_cols)) |>
          summarise(
            across(
              everything(),
              ~ {
                values <- safe_numeric(.x)
                values <- values[is.finite(values)]
                if (length(values) == 0) NA_real_ else values[[1]]
              }
            )
          ) |>
          pivot_longer(
            cols = everything(),
            names_to = "feature",
            values_to = "difference"
          ) |>
          filter(is.finite(difference))
      }
    }
  }

  if (nrow(long_df) == 0) {
    long_df <- build_from_wide_or_raw(fallback_data)
  }

  if (nrow(long_df) == 0) {
    return(
      empty_plot(
        "Largest Chester Era to Emily Era Feature Changes",
        "Era-transition columns could not be resolved and no raw era fallback was available."
      )
    )
  }

  df <- long_df |>
    mutate(
      absolute_change = abs(difference),
      label = feature |>
        str_remove("^lyrics_") |>
        str_remove("^audio_ab_") |>
        str_replace_all("_", " ")
    ) |>
    slice_max(
      absolute_change,
      n = n,
      with_ties = FALSE
    ) |>
    arrange(difference)

  plotly::plot_ly(
    df,
    x = ~difference,
    y = ~reorder(label, difference),
    type = "bar",
    orientation = "h",
    marker = list(
      color = palette$primary
    ),
    hovertemplate = paste0(
      "Feature: %{y}",
      "<br>Emily mean − Chester mean: %{x:.4f}",
      "<extra></extra>"
    )
  ) |>
    plotly::layout(
      title = "Largest Chester Era → Emily Era Feature Changes",
      xaxis = list(
        title = "Emily mean minus Chester mean",
        zeroline = TRUE,
        zerolinecolor = palette$accent,
        zerolinewidth = 2
      ),
      yaxis = list(title = ""),
      margin = list(
        l = 245,
        r = 30,
        t = 70,
        b = 65
      )
    ) |>
    plotly_config()
}

plot_era_comparison <- function(
    era_comparison,
    palette,
    n = 16,
    fallback_data = NULL) {

  # Convert either a track-level table or an era-level wide table into
  # feature / Chester mean / Emily mean rows.
  build_from_era_rows <- function(df) {
    if (is.null(df) || nrow(df) == 0) {
      return(tibble::tibble())
    }

    era_col <- resolve_feature_column(
      df,
      c(
        "era", "artist_era", "vocalist_era",
        "period", "artist_period", "vocalist"
      )
    )

    if (is.na(era_col)) {
      return(tibble::tibble())
    }

    era_values <- as.character(df[[era_col]])

    chester_rows <- grepl(
      "chester",
      era_values,
      ignore.case = TRUE
    )

    emily_rows <- grepl(
      "emily",
      era_values,
      ignore.case = TRUE
    )

    if (!any(chester_rows) || !any(emily_rows)) {
      return(tibble::tibble())
    }

    metadata <- c(
      era_col,
      "album_order", "release_year", "track_position",
      "track_number", "track_length_ms", "duration_ms",
      "duration_seconds", "master_track_id", "track_id",
      "recording_id", "album", "track_title",
      "canonical_title", "recording_title"
    )

    candidate_cols <- setdiff(names(df), metadata)

    numeric_cols <- candidate_cols[vapply(
      df[candidate_cols],
      function(x) {
        values <- suppressWarnings(as.numeric(x))
        any(is.finite(values))
      },
      logical(1)
    )]

    if (length(numeric_cols) == 0) {
      return(tibble::tibble())
    }

    result <- tibble::tibble(
      feature = numeric_cols,
      chester = vapply(
        numeric_cols,
        function(column) {
          values <- safe_numeric(df[[column]][chester_rows])
          values <- values[is.finite(values)]
          if (length(values) == 0) NA_real_ else mean(values)
        },
        numeric(1)
      ),
      emily = vapply(
        numeric_cols,
        function(column) {
          values <- safe_numeric(df[[column]][emily_rows])
          values <- values[is.finite(values)]
          if (length(values) == 0) NA_real_ else mean(values)
        },
        numeric(1)
      )
    ) |>
      filter(
        is.finite(chester),
        is.finite(emily)
      )

    result
  }

  parse_comparison_table <- function(df) {
    if (is.null(df) || nrow(df) == 0) {
      return(tibble::tibble())
    }

    feature_col <- resolve_feature_column(
      df,
      c(
        "feature", "metric", "variable", "feature_name",
        "measure", "column", "indicator"
      )
    )

    normalized_names <- normalize_feature_name(names(df))

    # Accept explicit names as well as labels such as
    # "Chester Era", "chester_average", "mean_chester_era", etc.
    chester_candidates <- which(
      grepl("chester", normalized_names)
      & grepl("mean|average|avg|value|era|profile", normalized_names)
    )

    emily_candidates <- which(
      grepl("emily", normalized_names)
      & grepl("mean|average|avg|value|era|profile", normalized_names)
    )

    if (length(chester_candidates) == 0) {
      chester_candidates <- which(grepl("chester", normalized_names))
    }

    if (length(emily_candidates) == 0) {
      emily_candidates <- which(grepl("emily", normalized_names))
    }

    chester_col <- if (length(chester_candidates) > 0) {
      names(df)[chester_candidates[[1]]]
    } else {
      NA_character_
    }

    emily_col <- if (length(emily_candidates) > 0) {
      names(df)[emily_candidates[[1]]]
    } else {
      NA_character_
    }

    # Standard wide format: one feature per row.
    if (
      !is.na(feature_col)
      && !is.na(chester_col)
      && !is.na(emily_col)
    ) {
      result <- df |>
        transmute(
          feature = as.character(.data[[feature_col]]),
          chester = safe_numeric(.data[[chester_col]]),
          emily = safe_numeric(.data[[emily_col]])
        ) |>
        filter(
          !is.na(feature),
          is.finite(chester),
          is.finite(emily)
        )

      if (nrow(result) > 0) {
        return(result)
      }
    }

    # Long format: feature / era / value.
    era_col <- resolve_feature_column(
      df,
      c(
        "era", "artist_era", "vocalist_era",
        "period", "artist_period", "vocalist"
      )
    )

    value_col <- resolve_feature_column(
      df,
      c(
        "value", "mean", "average", "avg",
        "feature_mean", "score", "metric_value"
      )
    )

    if (
      !is.na(feature_col)
      && !is.na(era_col)
      && !is.na(value_col)
    ) {
      wide <- df |>
        transmute(
          feature = as.character(.data[[feature_col]]),
          era = as.character(.data[[era_col]]),
          value = safe_numeric(.data[[value_col]])
        ) |>
        filter(
          !is.na(feature),
          is.finite(value)
        ) |>
        mutate(
          era_key = case_when(
            grepl("chester", era, ignore.case = TRUE) ~ "chester",
            grepl("emily", era, ignore.case = TRUE) ~ "emily",
            TRUE ~ NA_character_
          )
        ) |>
        filter(!is.na(era_key)) |>
        group_by(feature, era_key) |>
        summarise(
          value = mean(value),
          .groups = "drop"
        ) |>
        pivot_wider(
          names_from = era_key,
          values_from = value
        )

      if (all(c("chester", "emily") %in% names(wide))) {
        result <- wide |>
          transmute(
            feature,
            chester = safe_numeric(chester),
            emily = safe_numeric(emily)
          ) |>
          filter(
            is.finite(chester),
            is.finite(emily)
          )

        if (nrow(result) > 0) {
          return(result)
        }
      }
    }

    # Transposed/wide format: one row per era and one column per feature.
    build_from_era_rows(df)
  }

  df <- parse_comparison_table(era_comparison)

  # Guaranteed fallback: calculate Chester/Emily feature means directly
  # from the master analytical dataset when the CSV schema differs.
  if (nrow(df) == 0) {
    df <- build_from_era_rows(fallback_data)
  }

  if (nrow(df) == 0) {
    return(
      empty_plot(
        "Chester Era vs Emily Era Comparison",
        paste0(
          "Era means could not be resolved from era_comparison.csv ",
          "or reconstructed from the master dataset."
        )
      )
    )
  }

  df <- df |>
    mutate(
      difference = emily - chester,
      absolute_change = abs(difference),
      label = feature |>
        str_remove("^lyrics_") |>
        str_remove("^audio_ab_") |>
        str_replace_all("_", " ")
    ) |>
    filter(
      is.finite(difference),
      absolute_change > 0
    ) |>
    slice_max(
      absolute_change,
      n = n,
      with_ties = FALSE
    ) |>
    arrange(absolute_change) |>
    mutate(
      label = factor(label, levels = label)
    )

  if (nrow(df) == 0) {
    return(
      empty_plot(
        "Chester Era vs Emily Era Comparison",
        "The resolved era features contain no finite differences."
      )
    )
  }

  long <- df |>
    select(label, chester, emily, difference) |>
    pivot_longer(
      cols = c(chester, emily),
      names_to = "era",
      values_to = "value"
    ) |>
    mutate(
      era = recode(
        era,
        chester = "Chester Era",
        emily = "Emily Era"
      ),
      hover = paste0(
        "<b>", as.character(label), "</b>",
        "<br>Era: ", era,
        "<br>Mean: ", round(value, 4),
        "<br>Emily − Chester: ", round(difference, 4)
      )
    )

  plotly::plot_ly(
    long,
    x = ~value,
    y = ~label,
    color = ~era,
    colors = c(
      palette$secondary,
      palette$primary
    ),
    type = "bar",
    orientation = "h",
    hovertext = ~hover,
    hoverinfo = "text"
  ) |>
    plotly::layout(
      title = "Chester Era vs Emily Era Feature Means",
      barmode = "group",
      xaxis = list(title = "Era mean"),
      yaxis = list(title = ""),
      legend = list(
        orientation = "h",
        x = 0,
        y = 1.08
      ),
      margin = list(
        l = 245,
        r = 30,
        t = 90,
        b = 65
      )
    ) |>
    plotly_config()
}


plot_song_dispersion <- function(
    song_distances,
    palette,
    fallback_embeddings = NULL) {

  build_from_embeddings <- function(embeddings) {
    if (is.null(embeddings) || nrow(embeddings) == 0) {
      return(tibble::tibble())
    }

    album_col <- resolve_feature_column(
      embeddings,
      c("album", "album_name")
    )

    title_col <- pick_first(
      embeddings,
      c("track_title", "canonical_title", "recording_title", "song")
    )

    coordinates <- resolve_embedding_columns(
      embeddings,
      "umap"
    )

    if (
      is.na(album_col)
      || is.na(coordinates$x)
      || is.na(coordinates$y)
    ) {
      return(tibble::tibble())
    }

    embeddings |>
      transmute(
        album = as.character(.data[[album_col]]),
        song = if (!is.na(title_col)) {
          as.character(.data[[title_col]])
        } else {
          "Song"
        },
        x = safe_numeric(.data[[coordinates$x]]),
        y = safe_numeric(.data[[coordinates$y]])
      ) |>
      filter(
        !is.na(album),
        is.finite(x),
        is.finite(y)
      ) |>
      group_by(album) |>
      mutate(
        centroid_x = mean(x),
        centroid_y = mean(y),
        distance = sqrt(
          (x - centroid_x)^2 +
          (y - centroid_y)^2
        )
      ) |>
      ungroup() |>
      select(album, song, distance)
  }

  df <- tibble::tibble()

  if (!is.null(song_distances) && nrow(song_distances) > 0) {
    album_col <- resolve_feature_column(
      song_distances,
      c(
        "album", "album_name", "studio_album",
        "release_group_title"
      )
    )

    distance_col <- resolve_feature_column(
      song_distances,
      c(
        "distance_to_album_centroid",
        "distance_from_album_centroid",
        "hybrid_distance_to_album_centroid",
        "distance_to_centroid",
        "centroid_distance",
        "song_distance",
        "embedding_distance",
        "hybrid_distance",
        "distance",
        "profile_distance"
      )
    )

    title_col <- resolve_feature_column(
      song_distances,
      c(
        "track_title", "canonical_title",
        "recording_title", "song", "title"
      )
    )

    if (!is.na(album_col) && !is.na(distance_col)) {
      df <- song_distances |>
        transmute(
          album = as.character(.data[[album_col]]),
          distance = safe_numeric(.data[[distance_col]]),
          song = if (!is.na(title_col)) {
            as.character(.data[[title_col]])
          } else {
            "Song"
          }
        ) |>
        filter(
          !is.na(album),
          is.finite(distance)
        )
    }
  }

  if (nrow(df) == 0) {
    df <- build_from_embeddings(fallback_embeddings)
  }

  if (nrow(df) == 0) {
    return(
      empty_plot(
        "Within-Album Song Dispersion",
        "Song-distance columns could not be resolved and embedding fallback was unavailable."
      )
    )
  }

  album_levels <- c(
    "Hybrid Theory",
    "Meteora",
    "Minutes to Midnight",
    "A Thousand Suns",
    "Living Things",
    "The Hunting Party",
    "One More Light",
    "From Zero"
  )

  df <- df |>
    mutate(
      album = factor(album, levels = album_levels)
    ) |>
    filter(!is.na(album))

  plotly::plot_ly(
    df,
    x = ~album,
    y = ~distance,
    type = "box",
    color = ~album,
    boxpoints = "outliers",
    jitter = 0.25,
    pointpos = 0,
    hovertext = ~paste0(
      "<b>", song, "</b>",
      "<br>Album: ", album,
      "<br>Distance: ", round(distance, 3)
    ),
    hoverinfo = "text"
  ) |>
    plotly::layout(
      title = "Within-Album Song Dispersion",
      xaxis = list(
        title = "Studio album",
        tickangle = -25
      ),
      yaxis = list(
        title = "Distance to album centroid"
      ),
      showlegend = FALSE,
      margin = list(
        l = 80,
        r = 30,
        t = 70,
        b = 125
      )
    ) |>
    plotly_config()
}


# ------------------------------------------------------------
# Top emotion and sentiment songs
# ------------------------------------------------------------

plot_top_song_ranking <- function(
    ranking_df,
    title,
    metric_label,
    palette,
    n = 12) {

  if (is.null(ranking_df) || nrow(ranking_df) == 0) {
    return(
      empty_plot(
        title,
        paste0(title, " output is unavailable.")
      )
    )
  }

  title_col <- resolve_feature_column(
    ranking_df,
    c(
      "track_title", "canonical_title",
      "recording_title", "song", "title"
    )
  )

  album_col <- resolve_feature_column(
    ranking_df,
    c("album", "album_name", "studio_album")
  )

  score_col <- resolve_feature_column(
    ranking_df,
    c(
      "score", "value", "ratio",
      "lyrics_nrc_anger_ratio",
      "lyrics_nrc_sadness_ratio",
      "lyrics_vader_negative",
      "lyrics_vader_positive",
      "nrc_anger_ratio",
      "nrc_sadness_ratio",
      "vader_negative",
      "vader_positive"
    )
  )

  if (is.na(score_col)) {
    numeric_cols <- names(ranking_df)[vapply(
      ranking_df,
      function(x) {
        values <- safe_numeric(x)
        any(is.finite(values))
      },
      logical(1)
    )]

    excluded <- c(
      "rank", "album_order", "release_year",
      "track_position", "track_number"
    )

    score_candidates <- setdiff(numeric_cols, excluded)

    if (length(score_candidates) > 0) {
      score_col <- score_candidates[[1]]
    }
  }

  if (is.na(title_col) || is.na(score_col)) {
    return(
      empty_plot(
        title,
        "Song-title or score columns could not be resolved."
      )
    )
  }

  df <- ranking_df |>
    transmute(
      song = as.character(.data[[title_col]]),
      album = if (!is.na(album_col)) {
        as.character(.data[[album_col]])
      } else {
        NA_character_
      },
      score = safe_numeric(.data[[score_col]])
    ) |>
    filter(
      !is.na(song),
      is.finite(score)
    ) |>
    arrange(desc(score)) |>
    slice_head(n = n) |>
    mutate(
      label = ifelse(
        is.na(album) | album == "",
        song,
        paste0(song, " — ", album)
      )
    ) |>
    arrange(score)

  plotly::plot_ly(
    df,
    x = ~score,
    y = ~reorder(label, score),
    type = "bar",
    orientation = "h",
    marker = list(color = palette$primary),
    hovertemplate = paste0(
      "Song: %{y}",
      "<br>", metric_label, ": %{x:.4f}",
      "<extra></extra>"
    )
  ) |>
    plotly::layout(
      title = title,
      xaxis = list(title = metric_label),
      yaxis = list(title = ""),
      margin = list(l = 235, r = 25, t = 70, b = 60)
    ) |>
    plotly_config()
}

# ------------------------------------------------------------
# Minimum spanning tree
# ------------------------------------------------------------

plot_similarity_mst <- function(
    mst_edges,
    embeddings,
    palette,
    similarity_df = NULL) {

  normalize_song_key <- function(x) {
    x |>
      as.character() |>
      str_to_lower() |>
      str_replace_all("[^a-z0-9]+", "") |>
      str_trim()
  }

  resolve_edge_pair <- function(edges, nodes) {
    edge_pairs <- list(
      c("source_master_track_id", "target_master_track_id", "master_track_id"),
      c("source_track_id", "target_track_id", "track_id"),
      c("source_recording_id", "target_recording_id", "recording_id"),
      c("source_id", "target_id", "id"),
      c("node_1", "node_2", "master_track_id"),
      c("source", "target", "master_track_id"),
      c("source_track_title", "target_track_title", "track_title"),
      c("track_title_a", "track_title_b", "track_title"),
      c("song_a", "song_b", "track_title"),
      c("from_song", "to_song", "track_title"),
      c("source", "target", "track_title")
    )

    for (pair in edge_pairs) {
      if (
        pair[[1]] %in% names(edges)
        && pair[[2]] %in% names(edges)
        && pair[[3]] %in% names(nodes)
      ) {
        return(pair)
      }
    }

    NULL
  }

  build_mst_from_similarity <- function(similarity, nodes) {
    if (is.null(similarity) || nrow(similarity) == 0) {
      return(tibble::tibble())
    }

    source_col <- pick_first(
      similarity,
      c(
        "source_track_title", "track_title_a", "song_a",
        "source_song", "track_1"
      )
    )

    target_col <- pick_first(
      similarity,
      c(
        "target_track_title", "track_title_b", "song_b",
        "target_song", "track_2"
      )
    )

    score_col <- pick_first(
      similarity,
      c(
        "hybrid_similarity", "similarity_hybrid",
        "similarity", "cosine_similarity"
      )
    )

    if (any(is.na(c(source_col, target_col, score_col)))) {
      return(tibble::tibble())
    }

    candidates <- similarity |>
      transmute(
        source = as.character(.data[[source_col]]),
        target = as.character(.data[[target_col]]),
        similarity = safe_numeric(.data[[score_col]])
      ) |>
      filter(
        source != target,
        is.finite(similarity)
      ) |>
      mutate(
        source_key = normalize_song_key(source),
        target_key = normalize_song_key(target),
        edge_key = ifelse(
          source_key < target_key,
          paste(source_key, target_key, sep = "||"),
          paste(target_key, source_key, sep = "||")
        )
      ) |>
      group_by(edge_key) |>
      slice_max(similarity, n = 1, with_ties = FALSE) |>
      ungroup() |>
      arrange(desc(similarity))

    node_keys <- unique(nodes$song_key)
    parent <- setNames(node_keys, node_keys)

    find_root <- function(x) {
      while (parent[[x]] != x) {
        parent[[x]] <<- parent[[parent[[x]]]]
        x <- parent[[x]]
      }
      x
    }

    selected <- vector("list", 0)

    for (i in seq_len(nrow(candidates))) {
      a <- candidates$source_key[[i]]
      b <- candidates$target_key[[i]]

      if (!(a %in% node_keys) || !(b %in% node_keys)) {
        next
      }

      root_a <- find_root(a)
      root_b <- find_root(b)

      if (root_a != root_b) {
        parent[[root_b]] <- root_a
        selected[[length(selected) + 1]] <- candidates[i, ]

        if (length(selected) >= length(node_keys) - 1) {
          break
        }
      }
    }

    if (length(selected) == 0) {
      return(tibble::tibble())
    }

    bind_rows(selected)
  }

  if (nrow(embeddings) == 0) {
    return(
      empty_plot(
        "Linkin Park Similarity Minimum Spanning Tree",
        "song_embeddings.parquet is unavailable."
      )
    )
  }

  coordinates <- resolve_embedding_columns(
    embeddings,
    "umap"
  )

  title_col <- pick_first(
    embeddings,
    c(
      "track_title",
      "canonical_title",
      "recording_title"
    )
  )

  if (
    is.na(title_col)
    || is.na(coordinates$x)
    || is.na(coordinates$y)
  ) {
    return(
      empty_plot(
        "Linkin Park Similarity Minimum Spanning Tree",
        "Embedding title or UMAP coordinate columns could not be resolved."
      )
    )
  }

  nodes <- embeddings |>
    mutate(
      song = as.character(.data[[title_col]]),
      song_key = normalize_song_key(song),
      x = safe_numeric(.data[[coordinates$x]]),
      y = safe_numeric(.data[[coordinates$y]]),
      album_value = if ("album" %in% names(embeddings)) {
        as.character(album)
      } else {
        "Unknown"
      },
      era_value = if ("era" %in% names(embeddings)) {
        as.character(era)
      } else {
        NA_character_
      }
    ) |>
    filter(
      is.finite(x),
      is.finite(y)
    ) |>
    distinct(
      song_key,
      .keep_all = TRUE
    ) |>
    mutate(
      x = mean(x) + 1.18 * (x - mean(x)),
      y = mean(y) + 1.18 * (y - mean(y))
    )

  edges_joined <- tibble::tibble()

  if (!is.null(mst_edges) && nrow(mst_edges) > 0) {
    pair <- resolve_edge_pair(
      mst_edges,
      embeddings
    )

    if (!is.null(pair)) {
      source_col <- pair[[1]]
      target_col <- pair[[2]]
      node_col <- pair[[3]]

      node_id_lookup <- embeddings |>
        transmute(
          node_id = as.character(.data[[node_col]]),
          song_key = normalize_song_key(
            as.character(.data[[title_col]])
          )
        ) |>
        distinct(node_id, .keep_all = TRUE)

      node_lookup <- node_id_lookup |>
        left_join(
          nodes |>
            select(song_key, song, x, y),
          by = "song_key"
        ) |>
        filter(
          is.finite(x),
          is.finite(y)
        ) |>
        distinct(
          node_id,
          .keep_all = TRUE
        )

      edges_joined <- mst_edges |>
        transmute(
          source_id = as.character(.data[[source_col]]),
          target_id = as.character(.data[[target_col]])
        ) |>
        left_join(
          node_lookup |>
            select(
              source_id = node_id,
              source_song = song,
              source_key = song_key,
              x_source = x,
              y_source = y
            ),
          by = "source_id"
        ) |>
        left_join(
          node_lookup |>
            select(
              target_id = node_id,
              target_song = song,
              target_key = song_key,
              x_target = x,
              y_target = y
            ),
          by = "target_id"
        ) |>
        filter(
          is.finite(x_source),
          is.finite(y_source),
          is.finite(x_target),
          is.finite(y_target)
        )
    }

    if (nrow(edges_joined) == 0) {
      source_col <- pick_first(
        mst_edges,
        c(
          "source_track_title", "track_title_a", "song_a",
          "from_song", "source", "node_1"
        )
      )

      target_col <- pick_first(
        mst_edges,
        c(
          "target_track_title", "track_title_b", "song_b",
          "to_song", "target", "node_2"
        )
      )

      if (!is.na(source_col) && !is.na(target_col)) {
        edges_joined <- mst_edges |>
          transmute(
            source_song = as.character(.data[[source_col]]),
            target_song = as.character(.data[[target_col]]),
            source_key = normalize_song_key(source_song),
            target_key = normalize_song_key(target_song)
          ) |>
          left_join(
            nodes |>
              select(
                source_key = song_key,
                x_source = x,
                y_source = y
              ),
            by = "source_key"
          ) |>
          left_join(
            nodes |>
              select(
                target_key = song_key,
                x_target = x,
                y_target = y
              ),
            by = "target_key"
          ) |>
          filter(
            is.finite(x_source),
            is.finite(y_source),
            is.finite(x_target),
            is.finite(y_target)
          )
      }
    }
  }

  if (nrow(edges_joined) == 0) {
    fallback_edges <- build_mst_from_similarity(
      similarity_df,
      nodes
    )

    if (nrow(fallback_edges) > 0) {
      edges_joined <- fallback_edges |>
        transmute(
          source_song = source,
          target_song = target,
          source_key,
          target_key
        ) |>
        left_join(
          nodes |>
            select(
              source_key = song_key,
              x_source = x,
              y_source = y
            ),
          by = "source_key"
        ) |>
        left_join(
          nodes |>
            select(
              target_key = song_key,
              x_target = x,
              y_target = y
            ),
          by = "target_key"
        ) |>
        filter(
          is.finite(x_source),
          is.finite(y_source),
          is.finite(x_target),
          is.finite(y_target)
        )
    }
  }

  if (nrow(edges_joined) == 0) {
    return(
      empty_plot(
        "Linkin Park Similarity Minimum Spanning Tree",
        "MST identifiers could not be matched; similarity fallback also produced no usable edges."
      )
    )
  }

  p <- plotly::plot_ly()

  for (i in seq_len(nrow(edges_joined))) {
    p <- p |>
      plotly::add_trace(
        x = c(
          edges_joined$x_source[[i]],
          edges_joined$x_target[[i]]
        ),
        y = c(
          edges_joined$y_source[[i]],
          edges_joined$y_target[[i]]
        ),
        type = "scatter",
        mode = "lines",
        line = list(
          color = scales::alpha(
            palette$secondary,
            0.55
          ),
          width = 1.15
        ),
        hovertext = paste0(
          edges_joined$source_song[[i]],
          " ↔ ",
          edges_joined$target_song[[i]]
        ),
        hoverinfo = "text",
        showlegend = FALSE
      )
  }

  p |>
    plotly::add_trace(
      data = nodes,
      x = ~x,
      y = ~y,
      type = "scatter",
      mode = "markers+text",
      text = ~song,
      textposition = "top center",
      textfont = list(
        size = 7,
        color = palette$secondary
      ),
      color = ~album_value,
      marker = list(
        size = 9,
        opacity = 0.9,
        line = list(
          color = palette$secondary,
          width = 0.7
        )
      ),
      hovertext = ~paste0(
        "<b>", song, "</b>",
        "<br>Album: ", album_value,
        ifelse(
          is.na(era_value),
          "",
          paste0("<br>Era: ", era_value)
        )
      ),
      hoverinfo = "text"
    ) |>
    plotly::layout(
      title = "Linkin Park Similarity Minimum Spanning Tree",
      xaxis = list(
        title = "UMAP dimension 1",
        zeroline = FALSE
      ),
      yaxis = list(
        title = "UMAP dimension 2",
        zeroline = FALSE
      ),
      legend = list(
        title = list(text = "Album")
      ),
      margin = list(
        l = 90,
        r = 70,
        t = 85,
        b = 85
      )
    ) |>
    plotly_config()
}

