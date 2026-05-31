# ============================================================
# Snippet of code, taken from a larger notebook
# ============================================================

# Resampling unit:
#   "case"        — resample case_ids; all model rows for a case move together
#   "observation" — resample individual (model, case) delta values independently
BOOTSTRAP_RESAMPLE_UNIT = "observation"

# Stratify by model (only applies when BOOTSTRAP_RESAMPLE_UNIT = "observation"):
#   True  — resample within each model separately, then pool (preserves balanced
#           model representation; CI reflects case-sampling uncertainty only)
#   False — resample from all (model, case) pairs in the chapter regardless of model
#           (CI reflects both case-sampling and model-composition uncertainty)
BOOTSTRAP_STRATIFY_BY_MODEL = True

# Model overall means:
#   True  — compute once from full dataset, hold fixed across iterations
#   False — recompute from resampled data each iteration (slower, ~50x)
BOOTSTRAP_FIXED_MODEL_MEANS = True


def _print_bootstrap_config(resample_unit, stratify, fixed_means, n_iter, seed):
    """Log the active bootstrap configuration."""
    print("=" * 60)
    print("BOOTSTRAP CONFIGURATION")
    print("=" * 60)
    print(f"  Resampling unit:      {resample_unit}")
    if resample_unit == "observation":
        print(f"  Stratify by model:    {stratify}")
    else:
        print(f"  Stratify by model:    N/A (case-level resampling)")
    print(f"  Model overall means:  {'Fixed (pre-computed once)' if fixed_means else 'Recomputed per iteration'}")
    print(f"  Iterations:           {n_iter:,}")
    print(f"  Seed:                 {seed}")
    print("=" * 60)


def _bootstrap_fixed_means(
    df_model_case, score_cols, cases_by_chapter,
    resample_unit, stratify,
    n_iter, seed, progress_every,
    case_col, model_col, chapter_col,
):
    """
    Bootstrap with model overall means held fixed.
    Pre-computes case-level deltas, then resamples per config.
    """
    rng = np.random.default_rng(seed)
    boot_summaries = []

    for score_col in score_cols:
        if score_col not in df_model_case.columns:
            continue

        score_df = df_model_case.dropna(
            subset=[case_col, model_col, chapter_col, score_col]
        ).copy()

        if score_df.empty:
            continue

        # Compute model overall means once (fixed baselines)
        model_means = score_df.groupby(model_col)[score_col].mean()
        score_df["_delta"] = (
            score_df[score_col] - score_df[model_col].map(model_means)
        )

        if resample_unit == "case":
            # Index: case_id -> array of all model deltas for that case
            case_deltas = (
                score_df.groupby(case_col)["_delta"]
                .apply(np.array)
            )

            for i in range(n_iter):
                for ch, case_ids in cases_by_chapter.items():
                    valid_ids = np.intersect1d(case_ids, case_deltas.index)
                    if len(valid_ids) == 0:
                        continue

                    sampled = rng.choice(valid_ids, size=len(valid_ids), replace=True)
                    all_deltas = np.concatenate([case_deltas[cid] for cid in sampled])
                    boot_summaries.append({
                        "score_column": score_col,
                        chapter_col: ch,
                        "mean_delta": np.nanmean(all_deltas),
                        "bootstrap_iter": i,
                    })

                if progress_every and (i + 1) % progress_every == 0:
                    print(f"  [{score_col}] Iteration {i + 1:,}/{n_iter:,}")

        elif resample_unit == "observation" and not stratify:
            # Index: chapter -> array of all individual delta values
            chapter_deltas = (
                score_df.groupby(chapter_col)["_delta"]
                .apply(np.array)
            )

            for i in range(n_iter):
                for ch, case_ids in cases_by_chapter.items():
                    if ch not in chapter_deltas.index:
                        continue
                    deltas = chapter_deltas[ch]
                    if len(deltas) == 0:
                        continue

                    sampled = rng.choice(deltas, size=len(deltas), replace=True)
                    boot_summaries.append({
                        "score_column": score_col,
                        chapter_col: ch,
                        "mean_delta": np.nanmean(sampled),
                        "bootstrap_iter": i,
                    })

                if progress_every and (i + 1) % progress_every == 0:
                    print(f"  [{score_col}] Iteration {i + 1:,}/{n_iter:,}")

        elif resample_unit == "observation" and stratify:
            # Index: (chapter, model) -> array of delta values
            chapter_model_deltas = {}
            for (ch, mdl), grp in score_df.groupby([chapter_col, model_col]):
                chapter_model_deltas[(ch, mdl)] = grp["_delta"].to_numpy()

            # Pre-compute which models are present per chapter
            models_by_chapter = {}
            for ch in cases_by_chapter:
                models_by_chapter[ch] = [
                    mdl for (c, mdl) in chapter_model_deltas if c == ch
                ]

            for i in range(n_iter):
                for ch, case_ids in cases_by_chapter.items():
                    models = models_by_chapter.get(ch, [])
                    if not models:
                        continue

                    # Resample within each model separately, then pool
                    all_sampled = []
                    for mdl in models:
                        deltas = chapter_model_deltas.get((ch, mdl))
                        if deltas is None or len(deltas) == 0:
                            continue
                        sampled = rng.choice(deltas, size=len(deltas), replace=True)
                        all_sampled.append(sampled)

                    if all_sampled:
                        pooled = np.concatenate(all_sampled)
                        boot_summaries.append({
                            "score_column": score_col,
                            chapter_col: ch,
                            "mean_delta": np.nanmean(pooled),
                            "bootstrap_iter": i,
                        })

                if progress_every and (i + 1) % progress_every == 0:
                    print(f"  [{score_col}] Iteration {i + 1:,}/{n_iter:,}")

    return boot_summaries


def _bootstrap_recomputed_means(
    df_model_case, score_cols, cases_by_chapter,
    resample_unit, stratify,
    n_iter, seed, progress_every,
    case_col, model_col, chapter_col,
):
    """
    Bootstrap with model overall means recomputed per iteration.
    Resamples across ALL chapters simultaneously, rebuilds model baselines,
    then extracts chapter-level mean deltas.

    Note: ~50x slower than the fixed-means variant.
    """
    rng = np.random.default_rng(seed)
    boot_summaries = []

    for score_col in score_cols:
        if score_col not in df_model_case.columns:
            continue

        score_df = df_model_case.dropna(
            subset=[case_col, model_col, chapter_col, score_col]
        ).copy()

        if score_df.empty:
            continue

        if resample_unit == "case":
            # Build per-chapter case arrays and full score lookup
            case_rows = {
                cid: grp[[model_col, chapter_col, score_col]].to_numpy()
                for cid, grp in score_df.groupby(case_col)
            }

            for i in range(n_iter):
                # Resample case_ids within each chapter, collect all rows
                resampled_rows = []
                for ch, case_ids in cases_by_chapter.items():
                    valid_ids = np.intersect1d(case_ids, list(case_rows.keys()))
                    if len(valid_ids) == 0:
                        continue
                    sampled = rng.choice(valid_ids, size=len(valid_ids), replace=True)
                    for cid in sampled:
                        resampled_rows.append(case_rows[cid])

                if not resampled_rows:
                    continue

                combined = np.vstack(resampled_rows)
                boot_df = pd.DataFrame(
                    combined, columns=[model_col, chapter_col, score_col]
                )
                boot_df[score_col] = pd.to_numeric(boot_df[score_col])

                # Recompute model overall means from resampled data
                boot_model_means = boot_df.groupby(model_col)[score_col].mean()
                boot_df["_delta"] = (
                    boot_df[score_col] - boot_df[model_col].map(boot_model_means)
                )

                for ch, grp in boot_df.groupby(chapter_col):
                    boot_summaries.append({
                        "score_column": score_col,
                        chapter_col: ch,
                        "mean_delta": grp["_delta"].mean(),
                        "bootstrap_iter": i,
                    })

                if progress_every and (i + 1) % progress_every == 0:
                    print(f"  [{score_col}] Iteration {i + 1:,}/{n_iter:,}")

        elif resample_unit == "observation":
            # Build per-chapter observation arrays
            chapter_obs = {}
            for ch, case_ids in cases_by_chapter.items():
                ch_df = score_df.loc[score_df[chapter_col] == ch]
                if stratify:
                    chapter_obs[ch] = {
                        mdl: grp[[model_col, score_col]].to_numpy()
                        for mdl, grp in ch_df.groupby(model_col)
                    }
                else:
                    chapter_obs[ch] = ch_df[[model_col, score_col]].to_numpy()

            for i in range(n_iter):
                resampled_rows = []

                for ch in cases_by_chapter:
                    if ch not in chapter_obs:
                        continue

                    if stratify:
                        for mdl, obs in chapter_obs[ch].items():
                            if len(obs) == 0:
                                continue
                            idx = rng.choice(len(obs), size=len(obs), replace=True)
                            sampled = obs[idx]
                            # Tag with chapter
                            tagged = np.column_stack([
                                sampled,
                                np.full(len(sampled), ch),
                            ])
                            resampled_rows.append(tagged)
                    else:
                        obs = chapter_obs[ch]
                        if len(obs) == 0:
                            continue
                        idx = rng.choice(len(obs), size=len(obs), replace=True)
                        sampled = obs[idx]
                        tagged = np.column_stack([
                            sampled,
                            np.full(len(sampled), ch),
                        ])
                        resampled_rows.append(tagged)

                if not resampled_rows:
                    continue

                combined = np.vstack(resampled_rows)
                boot_df = pd.DataFrame(
                    combined, columns=[model_col, score_col, chapter_col]
                )
                boot_df[score_col] = pd.to_numeric(boot_df[score_col])

                boot_model_means = boot_df.groupby(model_col)[score_col].mean()
                boot_df["_delta"] = (
                    boot_df[score_col] - boot_df[model_col].map(boot_model_means)
                )

                for ch, grp in boot_df.groupby(chapter_col):
                    boot_summaries.append({
                        "score_column": score_col,
                        chapter_col: ch,
                        "mean_delta": grp["_delta"].mean(),
                        "bootstrap_iter": i,
                    })

                if progress_every and (i + 1) % progress_every == 0:
                    print(f"  [{score_col}] Iteration {i + 1:,}/{n_iter:,}")

    return boot_summaries


def bootstrap_model_normalised_delta_cis(
    df_model_case,
    score_cols,
    n_iter=BOOTSTRAP_N_ITER,
    seed=BOOTSTRAP_SEED,
    threshold=N_CASE_THRESHOLD,
    case_col=CASE_COL,
    model_col=MODEL_COL,
    chapter_col=CHAPTER_COL,
    progress_every=BOOTSTRAP_PROGRESS_EVERY,
    resample_unit=BOOTSTRAP_RESAMPLE_UNIT,
    stratify_by_model=BOOTSTRAP_STRATIFY_BY_MODEL,
    fixed_model_means=BOOTSTRAP_FIXED_MODEL_MEANS,
):
    """
    Bootstrap CIs for model-normalised ICD chapter deltas.

    Supports three config dimensions:
      - resample_unit: "case" (cluster) or "observation" (model/case pairs)
      - stratify_by_model: balance model representation per iteration
        (only used when resample_unit="observation")
      - fixed_model_means: hold model baselines fixed, or recompute per iteration
    """
    validate_required_columns(
        df_model_case, [case_col, model_col, chapter_col], "df_model_case"
    )

    _print_bootstrap_config(resample_unit, stratify_by_model, fixed_model_means, n_iter, seed)

    if n_iter <= 0:
        return pd.DataFrame(), pd.DataFrame()

    if resample_unit not in ("case", "observation"):
        raise ValueError(f"resample_unit must be 'case' or 'observation', got '{resample_unit}'")

    # Build case-chapter lookup
    case_chapter = (
        df_model_case[[case_col, chapter_col]]
        .drop_duplicates()
        .dropna(subset=[case_col, chapter_col])
        .drop_duplicates(case_col, keep="first")
    )
    cases_by_chapter = {
        ch: grp[case_col].to_numpy()
        for ch, grp in case_chapter.groupby(chapter_col, dropna=False)
    }

    if not cases_by_chapter:
        warnings.warn("No cases available for bootstrap.")
        return pd.DataFrame(), pd.DataFrame()

    # Dispatch
    if fixed_model_means:
        boot_summaries = _bootstrap_fixed_means(
            df_model_case, score_cols, cases_by_chapter,
            resample_unit, stratify_by_model,
            n_iter, seed, progress_every,
            case_col, model_col, chapter_col,
        )
    else:
        boot_summaries = _bootstrap_recomputed_means(
            df_model_case, score_cols, cases_by_chapter,
            resample_unit, stratify_by_model,
            n_iter, seed, progress_every,
            case_col, model_col, chapter_col,
        )

    if not boot_summaries:
        warnings.warn("Bootstrap produced no usable summaries.")
        return pd.DataFrame(), pd.DataFrame()

    bootstrap_long = pd.DataFrame(boot_summaries)

    ci_summary = (
        bootstrap_long.groupby(["score_column", chapter_col], dropna=False)
        .agg(
            ci_low=("mean_delta", lambda x: x.quantile(0.025)),
            ci_high=("mean_delta", lambda x: x.quantile(0.975)),
            bootstrap_n=("mean_delta", "count"),
        )
        .reset_index()
    )

    return ci_summary, bootstrap_long
