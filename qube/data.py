def read_table(path: str) -> pd.DataFrame:
    file_path = pathlib.Path(path)
    if file_path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)
    return pd.read_csv(file_path)


def sanitize_feature_names(columns) -> list[str]:
    """Make dataframe feature names safe for XGBoost and similar estimators."""
    replacements = {
        "[": "_",
        "]": "_",
        "<": "_",
        ">": "_",
        " ": "_",
        "(": "_",
        ")": "_",
        "/": "_",
        "\\": "_",
        ":": "_",
        ";": "_",
        ",": "_",
    }
    seen: dict[str, int] = {}
    used: set[str] = set()
    sanitized: list[str] = []

    for column in columns:
        name = str(column)
        for old, new in replacements.items():
            name = name.replace(old, new)
        while "__" in name:
            name = name.replace("__", "_")
        name = name.strip("_") or "feature"

        base = name
        count = seen.get(base, 0) + 1
        candidate = base if count == 1 else f"{base}_{count}"
        while candidate in used:
            count += 1
            candidate = f"{base}_{count}"
        seen[base] = count
        used.add(candidate)
        sanitized.append(candidate)

    return sanitized


def load_dataset(
    path: str,
    target: str | None = None,
    drug: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray, list[str], dict]:
    if drug:
        return load_ccle_gdsc_drug_dataset(path, drug)

    if target is None:
        raise ValueError("Either --target or --drug must be supplied.")

    df = read_table(path)

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Columns: {list(df.columns)}")

    y_raw = df[target]
    X_df = df.drop(columns=[target])
    numeric_cols = [col for col in X_df.columns if pd.api.types.is_numeric_dtype(X_df[col])]
    if not numeric_cols:
        raise ValueError("No numeric gene/features columns found.")
    X_df = X_df[numeric_cols]
    X_df.columns = sanitize_feature_names(X_df.columns)
    feature_names = X_df.columns.tolist()
    print("Sanitized feature names for ML compatibility.")

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    summary = {
        "mode": "target",
        "target": target,
        "samples": int(len(X_df)),
        "features": int(len(feature_names)),
        "classes": {
            str(label): int(count)
            for label, count in zip(*np.unique(y_raw.astype(str), return_counts=True))
        },
    }
    return X_df, y, feature_names, summary


def load_ccle_gdsc_drug_dataset(
    path: str,
    drug: str,
) -> tuple[pd.DataFrame, np.ndarray, list[str], dict]:
    file_path = pathlib.Path(path)
    if file_path.suffix.lower() in {".xlsx", ".xls"}:
        df = read_table(path)
    else:
        df = read_drug_rows_csv(path, drug)

    required = {"DRUG_NAME", "LN_IC50"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Drug mode requires columns {missing}.")

    if file_path.suffix.lower() in {".xlsx", ".xls"}:
        drug_df = df.loc[df["DRUG_NAME"] == drug].copy()
    else:
        drug_df = df

    if drug_df.empty:
        raise ValueError(f"No samples found for drug: {drug}")

    drug_df["LN_IC50"] = pd.to_numeric(drug_df["LN_IC50"], errors="coerce")
    drug_df = drug_df.dropna(subset=["LN_IC50"]).copy()
    if drug_df.empty:
        raise ValueError(f"No non-missing LN_IC50 values found for drug '{drug}'.")

    median = drug_df["LN_IC50"].median()
    drug_df["label"] = (drug_df["LN_IC50"] < median).astype(int)
    y = drug_df["label"].to_numpy(dtype=int)
    if len(np.unique(y)) < 2:
        raise ValueError(
            f"Median split for drug '{drug}' produced one class only. "
            "Check whether LN_IC50 values are all identical."
        )

    candidate_cols = [
        col
        for col in drug_df.columns
        if col not in CCLE_GDSC_METADATA_COLUMNS
        and pd.api.types.is_numeric_dtype(drug_df[col])
    ]
    if not candidate_cols:
        raise ValueError(
            "No numeric gene-expression columns found after excluding CCLE/GDSC metadata."
        )

    X_df = drug_df[candidate_cols].astype(np.float32, copy=False)
    X_df.columns = sanitize_feature_names(X_df.columns)
    feature_names = X_df.columns.tolist()
    print("Sanitized feature names for ML compatibility.")
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    print(f"Filtered samples: {len(X_df)}")
    print(f"Genes: {len(feature_names)}")
    print("Memory-efficient loading complete.")
    summary = {
        "mode": "drug",
        "drug": drug,
        "samples": int(len(X_df)),
        "genes": int(len(feature_names)),
        "positive_class": positives,
        "negative_class": negatives,
        "median_ln_ic50": float(median),
    }
    return X_df, y, feature_names, summary


def read_drug_rows_csv(path: str, drug: str, chunksize: int = 50000) -> pd.DataFrame:
    print("Loading dataset in chunks...")
    filtered_chunks: list[pd.DataFrame] = []

    for chunk_number, chunk in enumerate(pd.read_csv(path, chunksize=chunksize), start=1):
        print(f"Chunk {chunk_number}...")
        required = {"DRUG_NAME", "LN_IC50"}
        missing = sorted(required - set(chunk.columns))
        if missing:
            raise ValueError(f"Drug mode requires columns {missing}.")

        chunk = chunk[chunk["DRUG_NAME"] == drug]
        if chunk.empty:
            continue
        filtered_chunks.append(chunk.copy())

    if not filtered_chunks:
        raise ValueError(f"No samples found for drug: {drug}")

    return pd.concat(filtered_chunks, axis=0, ignore_index=True, copy=False)


def print_dataset_summary(summary: dict) -> None:
    if summary["mode"] == "drug":
        print(f"Drug: {summary['drug']}")
        print(f"Samples: {summary['samples']}")
        print(f"Genes: {summary['genes']}")
        print(f"Positive class: {summary['positive_class']}")
        print(f"Negative class: {summary['negative_class']}")
        return

    print(f"Target: {summary['target']}")
    print(f"Samples: {summary['samples']}")
    print(f"Features: {summary['features']}")
    for label, count in summary["classes"].items():
        print(f"Class {label}: {count}")


def load_edge_list(path: str | None) -> pd.DataFrame | None:
    if path is None:
        return None
    return read_table(path)


def count_drug_samples(path: str, chunksize: int = 50000) -> pd.Series:
    file_path = pathlib.Path(path)
    if file_path.suffix.lower() in {".xlsx", ".xls"}:
        df = read_table(path)
        if "DRUG_NAME" not in df.columns:
            raise ValueError("--all-drugs requires a DRUG_NAME column.")
        return df["DRUG_NAME"].dropna().astype(str).value_counts()

    counts = {}
    print("Counting samples per drug in chunks...")
    for chunk_number, chunk in enumerate(pd.read_csv(path, chunksize=chunksize), start=1):
        print(f"Count chunk {chunk_number}...")
        if "DRUG_NAME" not in chunk.columns:
            raise ValueError("--all-drugs requires a DRUG_NAME column.")
        chunk_counts = chunk["DRUG_NAME"].dropna().astype(str).value_counts()
        for drug, count in chunk_counts.items():
            counts[drug] = counts.get(drug, 0) + int(count)
    return pd.Series(counts, dtype=np.int64).sort_values(ascending=False)


def _metric_mean(result: dict, model: str, metric: str) -> float | None:
    value = result.get("aggregate", {}).get(model, {}).get(metric, {}).get("mean")
    return None if value is None else float(value)


def _format_optional(value: float | None) -> str:
    return "" if value is None or not np.isfinite(value) else f"{value:.4f}"