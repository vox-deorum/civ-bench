library(PlackettLuce)

args <- commandArgs(trailingOnly = TRUE)
input_csv <- args[1]
output_csv <- args[2]
# Reference player_type (default Vanilla, overridable for composite/strategy types)
reference_type <- if (length(args) >= 3) args[3] else "Vanilla"

# Read game-level data: game_id, player_type, adjusted_strength, slot_id
df <- read.csv(input_csv, stringsAsFactors = FALSE)

# Use slot_id as the item (each unique player slot gets its own worth parameter)
slot_ids <- sort(unique(df$slot_id))
n_slots <- length(slot_ids)

# Build rankings matrix: rows = games, cols = slots, values = rank (1=best, 0=absent)
game_ids <- unique(df$game_id)
n_games <- length(game_ids)

rankings_mat <- matrix(0L, nrow = n_games, ncol = n_slots)
colnames(rankings_mat) <- slot_ids

for (i in seq_along(game_ids)) {
  game <- df[df$game_id == game_ids[i], ]
  game <- game[order(-game$adjusted_strength), ]
  # Assign ranks with ties: differences within 0.01 are treated as tied
  rank <- 1L
  for (j in seq_len(nrow(game))) {
    if (j > 1 && (game$adjusted_strength[j - 1] - game$adjusted_strength[j]) >= 0.01) {
      rank <- j
    }
    col_idx <- match(game$slot_id[j], slot_ids)
    rankings_mat[i, col_idx] <- rank
  }
}

# Fit Plackett-Luce model on expanded slots
R <- as.rankings(rankings_mat)
mod <- PlackettLuce(R)

# Extract slot-level results with <reference>_0 as reference
ref_slot <- paste0(reference_type, "_0")
vanilla_ref <- match(ref_slot, slot_ids)
log_worth_all <- coef(mod, ref = vanilla_ref)
# Standard errors come from the variance-covariance matrix, whose information matrix
# can be singular (non-positive-definite) for sparse/disconnected ranking data —
# vcov.PlackettLuce then aborts inside chol(). The point estimates (coef) are
# unaffected, so treat a vcov failure as "SEs unavailable" (NA) and still emit the
# log-worth / Elo ratings, rather than killing the whole `civ-bench run`.
coef_table <- tryCatch(
  summary(mod, ref = vanilla_ref)$coefficients,
  error = function(e) {
    message("plackett_luce: vcov/summary failed (", conditionMessage(e),
            "); standard errors set to NA.")
    NULL
  }
)

# Build slot-level dataframe
slot_results <- data.frame(
  slot_id = slot_ids,
  log_worth = as.numeric(log_worth_all[slot_ids]),
  se_log_worth = NA_real_,
  stringsAsFactors = FALSE
)

# Extract the base player_type from slot_id (remove trailing _N)
slot_results$player_type <- sub("_[0-9]+$", "", slot_results$slot_id)

if (!is.null(coef_table)) {
  # Fill in standard errors from the coefficients table
  for (sid in rownames(coef_table)) {
    idx <- match(sid, slot_results$slot_id)
    if (!is.na(idx)) {
      slot_results$se_log_worth[idx] <- coef_table[sid, "Std. Error"]
    }
  }
  # The reference slot is absent from the coefficients table by construction, so its
  # SE is genuinely 0. Only the reference is zeroed: when vcov failed every other SE
  # stays NA rather than being coerced to 0 (which would falsely imply certainty).
  slot_results$se_log_worth[slot_results$slot_id == ref_slot] <- 0
}

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
  avg_log_worth <- mean(subset$log_worth)
  # Pooled SE: sqrt(sum of variances) / n_slots
  n <- nrow(subset)
  pooled_se <- sqrt(sum(subset$se_log_worth^2)) / n
  results <- rbind(results, data.frame(
    player_type = pt,
    log_worth = avg_log_worth,
    worth = exp(avg_log_worth),
    se_log_worth = pooled_se,
    n_slots = n,
    stringsAsFactors = FALSE
  ))
}

# Re-center on the reference type
vanilla_lw <- results$log_worth[results$player_type == reference_type]
results$log_worth <- results$log_worth - vanilla_lw
results$worth <- exp(results$log_worth)

# Add z-value and p-value (testing H0: log_worth = 0, i.e., same as reference)
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
  metric = c("deviance", "df_residual", "AIC", "n_games", "n_slots", "n_types", "n_iterations"),
  value = c(deviance(mod), mod$df.residual, AIC(mod), n_games, n_slots, length(player_types), mod$iter)
)
diag_csv <- sub("\\.csv$", "_diagnostics.csv", output_csv)
write.csv(diagnostics, diag_csv, row.names = FALSE)

# Write slot-level details for diagnostics
slot_csv <- sub("\\.csv$", "_slots.csv", output_csv)
slot_results$log_worth_centered <- slot_results$log_worth - mean(slot_results$log_worth[slot_results$player_type == reference_type])
write.csv(slot_results, slot_csv, row.names = FALSE)

cat("Plackett-Luce model fitted successfully.\n")
cat("Slots:", n_slots, "| Types:", length(player_types), "| Games:", n_games, "\n")
cat("Results written to:", output_csv, "\n")
