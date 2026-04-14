import numpy as np
import pandas as pd
from tabulate import tabulate
import json
from pydantic import ValidationError
from .schemas import ProcessedDataset
from pathlib import Path

class Statistics:
    def __init__(self, json_dir, categories):
        self.json_dir = Path(json_dir)
        self.categories = categories
        self.json_paths = self._collect_json_files()

    def _collect_json_files(self):
        # Cerca tutti i file .json nella directory e nelle sottocartelle
        json_files = list(sorted(str(f) for f in self.json_dir.glob('*.json')))
        if not json_files:
            for subdir in self.json_dir.iterdir():
                if subdir.is_dir():
                    json_files.extend(str(f) for f in subdir.glob('*.json'))
        return json_files

    @staticmethod
    def mean(matrix):
        return np.mean(matrix, axis=0) if matrix.size > 0 else np.array([])

    @staticmethod
    def variance(matrix):
        return np.var(matrix, axis=0, ddof=1) if matrix.size > 0 else np.array([])

    @staticmethod
    def dispersion(variance, mean):
        # Evita divisioni per zero
        with np.errstate(divide='ignore', invalid='ignore'):
            disp = np.true_divide(variance, mean)
            disp[~np.isfinite(disp)] = 0  # imposta a 0 se nan o inf
        return disp

    def process_all(self):
        if not self.json_paths:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Nessun file JSON trovato in {self.json_dir} o sottocartelle per le statistiche.")
            return
        for file_path in self.json_paths:
            self.process_file(file_path)

    def process_file(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        try:
            dataset = ProcessedDataset(**data)
        except ValidationError as e:
            print(f"[ERROR] {file_path} non valido secondo ProcessedDataset: {e}")
            return
        df = pd.DataFrame([w.dict() for w in dataset.windows])
        # Mappa le categorie ai nomi delle colonne effettive (case-insensitive)
        df_cols_lower = {col.lower(): col for col in df.columns}
        cat_map = [df_cols_lower[cat.lower()] for cat in self.categories if cat.lower() in df_cols_lower]
        if not cat_map:
            print(f"[ERROR] Nessuna delle categorie richieste trovata tra le colonne del file {file_path}.")
            return
        matrix = df[cat_map].to_numpy()
        means = self.mean(matrix)
        variances = self.variance(matrix)
        dispersions = self.dispersion(variances, means)
        silent_rows = (matrix.sum(axis=1) == 0).sum()
        silent_pct = 100 * silent_rows / matrix.shape[0] if matrix.shape[0] > 0 else 0
        table = [
            [proto, f"{means[i]:.2f}", f"{variances[i]:.2f}", f"{dispersions[i]:.2f}"]
            for i, proto in enumerate(cat_map)
        ]
        headers = ["Protocol", "Mean", "Variance", "Dispersion Index"]
        report = tabulate(table, headers, tablefmt="github")
        print(f"# Overdispersion Report for {file_path}\n")
        print(f"**Total Instances (R):** {matrix.shape[0]}")
        print(f"**Silent Windows:** {silent_rows} ({silent_pct:.1f}%)\n")
        print(report)
