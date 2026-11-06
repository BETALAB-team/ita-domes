from pathlib import Path
p=Path('output/article_method_2021_20261104/02_probabilities/run_probabilities.py');s=p.read_text();exec(s[:s.index('from sklearn.cluster import KMeans')]);dall.to_csv(OUT/'prefit_cohort_all_ids.csv',index=False);print(dall.eligible.value_counts());print(dall.exclusion_reason.value_counts())
