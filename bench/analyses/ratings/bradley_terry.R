library(BradleyTerry2)

args <- commandArgs(trailingOnly = TRUE)
input_csv <- args[1]
output_csv <- args[2]
margin <- if (length(args) >= 3 && args[3] != "NA") as.numeric(args[3]) else NA
reference_type <- if (length(args) >= 4) args[4] else "Vanilla"

# Read game-level data: game_id, player_type, adjusted_strength, slot_id
df <- read.csv(input_csv, stringsAsFactors = FALSE)

slot_ids <- sort(unique(df$slot_id))
n_slots <- length(slot_ids)

# Decompose each game into pairwise comparisons between different player types
game_ids <- unique(df$game_id)
n_games <- length(game_ids)

pairs <- data.frame(
  winner = character(0),
  loser = character(0),
  score_diff = numeric(0),
  stringsAsFactors = FALSE
)

for (gid in game_ids) {
  game <- df[df$game_id == gid, ]
  n_players <- nrow(game)

  for (a in seq_len(n_players - 1)) {
    for (b in (a + 1):n_players) {
      # Skip same player_type pairs
      if (game$player_type[a] == game$player_type[b]) next

      diff <- game$adjusted_strength[a] - game$adjusted_strength[b]

      # Skip ties (diff < 0.01)
      if (abs(diff) < 0.01) next

      if (diff > 0) {
        pairs <- rbind(pairs, data.frame(
          winner = game$slot_id[a],
          loser = game$slot_id[b],
          score_diff = abs(diff),
          stringsAsFactors = FALSE
        ))
      } else {
        pairs <- rbind(pairs, data.frame(
          winner = game$slot_id[b],
          loser = game$slot_id[a],
          score_diff = abs(diff),
          stringsAsFactors = FALSE
        ))
      }
    }
  }
}

n_pairs <- nrow(pairs)

# Compute margin (median pairwise score diff if not provided)
if (is.na(margin)) {
  margin <- median(pairs$score_diff)
}

# Compute per-pair weights: mirrors OpenSkill's log1p(score_diff / margin)
pairs$weight <- 1 + log1p(pairs$score_diff / margin)

# Convert to factors with reference type as the first (reference) level
vanilla_ref <- paste0(reference_type, "_0")
ordered_slots <- c(vanilla_ref, setdiff(slot_ids, vanilla_ref))
pairs$winner <- factor(pairs$winner, levels = ordered_slots)
pairs$loser <- factor(pairs$loser, levels = ordered_slots)

# Fit Bradley-Terry model
# BTm expects outcome (1 = player1 wins), player1, player2
mod <- BTm(
  outcome = rep(1, n_pairs),
  player1 = data.frame(player = pairs$winner),
  player2 = data.frame(player = pairs$loser),
  weights = pairs$weight
)

# Extract slot-level results (Vanilla_0 is already the reference level)
log_ability <- BTabilities(mod)

# Build slot-level dataframe
slot_results <- data.frame(
  slot_id = rownames(log_ability),
  log_worth = as.numeric(log_ability[, "ability"]),
  se_log_worth = as.numeric(log_ability[, "s.e."]),
  stringsAsFactors = FALSE
)

# Extract the base player_type from slot_id (remove trailing _N)
slot_results$player_type <- sub("_[0-9]+$", "", slot_results$slot_id)

# Re-center on Vanilla_0
vanilla_lw_slot <- slot_results$log_worth[slot_results$slot_id == vanilla_ref]
slot_results$log_worth <- slot_results$log_worth - vanilla_lw_slot

# Aggregate by player_type: average log-worth, pooled SE
player_types <- sort(unique(slot_results$player_type))
results <- data.frame(
  player_type = character(0),
  log_worth = numeric(0),
  worth = numeric(0),
  se_log_worth = numeric(0),
  n_slots = integer(0),
  stringsAsFactors = FALSE
)

for (pt in player_types) {
  subset <- slot_results[slot_results$player_type == pt, ]
  n <- nrow(subset)
  # Exclude reference slots (se = 0) — they are fixed anchor points
  est <- subset[subset$se_log_worth > 0, ]
  if (nrow(est) == 0) {
    # All slots are reference level (only the reference type)
    avg_log_worth <- 0
    pooled_se <- 0
  } else if (nrow(est) == 1) {
    avg_log_worth <- est$log_worth
    pooled_se <- est$se_log_worth
  } else {
    # Inverse-variance weighting: well-estimated slots dominate
    weights <- 1 / est$se_log_worth^2
    avg_log_worth <- sum(weights * est$log_worth) / sum(weights)
    pooled_se <- 1 / sqrt(sum(weights))
  }
  results <- rbind(results, data.frame(
    player_type = pt,
    log_worth = avg_log_worth,
    worth = exp(avg_log_worth),
    se_log_worth = pooled_se,
    n_slots = n,
    stringsAsFactors = FALSE
  ))
}

# Re-center on reference type
vanilla_lw <- results$log_worth[results$player_type == reference_type]
results$log_worth <- results$log_worth - vanilla_lw
results$worth <- exp(results$log_worth)

# Add z-value and p-value (testing H0: log_worth = 0, i.e., same as Vanilla)
results$z_value <- ifelse(results$se_log_worth > 0,
                          results$log_worth / results$se_log_worth,
                          NA_real_)
results$p_value <- ifelse(!is.na(results$z_value),
                          2 * pnorm(-abs(results$z_value)),
                          NA_real_)

# Write results
write.csv(results, output_csv, row.names = FALSE)

# Write diagnostics
diagnostics <- data.frame(
  metric = c("deviance", "df_residual", "AIC", "n_games", "n_pairs", "n_slots",
             "n_types", "margin", "mean_weight", "min_weight", "max_weight"),
  value = c(deviance(mod), mod$df.residual, AIC(mod), n_games, n_pairs, n_slots,
            length(player_types), margin, mean(pairs$weight), min(pairs$weight),
            max(pairs$weight))
)
diag_csv <- sub("\\.csv$", "_diagnostics.csv", output_csv)
write.csv(diagnostics, diag_csv, row.names = FALSE)

# Write slot-level details for diagnostics
slot_csv <- sub("\\.csv$", "_slots.csv", output_csv)
slot_results$log_worth_centered <- slot_results$log_worth - mean(slot_results$log_worth[slot_results$player_type == reference_type])
write.csv(slot_results, slot_csv, row.names = FALSE)

cat("Bradley-Terry model fitted successfully.\n")
cat("Slots:", n_slots, "| Types:", length(player_types), "| Games:", n_games, "| Pairs:", n_pairs, "\n")
cat("Margin:", margin, "| Mean weight:", round(mean(pairs$weight), 3), "\n")
cat("Results written to:", output_csv, "\n")
