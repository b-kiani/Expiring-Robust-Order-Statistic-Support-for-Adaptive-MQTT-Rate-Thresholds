CICIoT2023_OpenSet_Pipeline.ipynb writes ONLY inside this folder.
_shared/00_inventory     data inventory (folders, files, headers, feature profile)
_shared/01_audit         duplicates / conflicts / file manifest / raw sample / meta.json
_shared/02_clean_parquet de-duplicated data reused by every profile
<PROFILE>/tables         result tables (CSV)
<PROFILE>/figures        figures (PNG, 300 dpi)
<PROFILE>/artifacts      pickled intermediate results
<PROFILE>/logs           run_log.txt (full console log)
<PROFILE>/run_config.json settings + package versions of the last run
