
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

    def mean_variance_per_column(self, matrix):
        if matrix.size == 0:
            return np.array([]), np.array([])
        means = np.mean(matrix, axis=0)
        variances = np.var(matrix, axis=0, ddof=1)
        return means, variances

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
        # Usa le categorie/protocolli passati dal chiamante
        matrix = df[self.categories].to_numpy()
        means, variances = self.mean_variance_per_column(matrix)
        dispersion = variances / means
        verdicts = [
            'Overdispersed' if v > m else 'Poisson-compliant' for v, m in zip(variances, means)
        ]
        silent_rows = (matrix.sum(axis=1) == 0).sum()
        silent_pct = 100 * silent_rows / matrix.shape[0] if matrix.shape[0] > 0 else 0
        table = []
        for i, proto in enumerate(self.categories):
            table.append([
                proto,
                f"{means[i]:.2f}",
                f"{variances[i]:.2f}",
                f"{dispersion[i]:.2f}",
                verdicts[i]
            ])
        headers = ["Protocol", "Mean", "Variance", "Dispersion Index", "Status"]
        report = tabulate(table, headers, tablefmt="github")
        if all(variances > means) and all(d > 1 for d in dispersion):
            final = "**Dataset is Overdispersed and ready for DM-Modeling**"
        else:
            final = "**Warning: Dataset lacks overdispersion**"
        print(f"# Overdispersion Report for {file_path}\n")
        print(f"**Total Instances (R):** {matrix.shape[0]}")
        print(f"**Silent Windows:** {silent_rows} ({silent_pct:.1f}%)\n")
        print(report)
        print(f"\n{final}\n")
