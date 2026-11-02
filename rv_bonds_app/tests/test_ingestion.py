import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts import ingest_csv_to_parquet as ingest

def test_floating_rate_bonds_are_not_ingested(tmp_path, monkeypatch):
    (tmp_path / 'app').mkdir()
    bonds = tmp_path / 'bonds' / 'x'
    bonds.mkdir(parents=True)
    for isin in ('FIXED1', 'FLOAT1'):
        (bonds / f'{isin}.csv').write_text(';px_last;yld_ytm_mid;asset_swap_spd_mid;z_sprd_mid\n2026-01-05;100;2.0;10;12\n')
    (tmp_path / 'mapping_indiv_bond.csv').write_text(
        'id_isin;security_name;maturity;inflation_linked_indicator;cpn_typ\n'
        'FIXED1;B 1 2030;2030-01-01;N;FIXED\nFLOAT1;B Float 2030;2030-01-01;N;FLOATING\n')
    written = []
    monkeypatch.setattr(ingest, 'RAW_DATA_DIR', tmp_path / 'bonds')
    monkeypatch.setattr(ingest, 'BASE_DIR', tmp_path / 'app')
    monkeypatch.setattr(ingest, 'write_market_data', written.append)
    ingest.run_ingestion()
    assert list(written[0]['isin']) == ['FIXED1']
