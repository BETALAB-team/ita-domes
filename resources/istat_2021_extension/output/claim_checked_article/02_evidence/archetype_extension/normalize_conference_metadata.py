"""Normalize verified citation labels in the conference-audit input records."""
import json
from pathlib import Path
p=Path(__file__).parent/'conference_source_additions.json'
records=json.loads(p.read_text(encoding='utf-8'))
titles={
 'CF-SRC-EPBD2024':'Directive (EU) 2024/1275 of the European Parliament and of the Council of 24 April 2024 on the energy performance of buildings (recast)',
 'CF-SRC-EESC2022':"Opinion of the European Economic and Social Committee on the Communication: 'Fit for 55': delivering the EU's 2030 Climate Target on the way to climate neutrality [COM(2021) 550 final]",
 'CF-SRC-FIT552021':"'Fit for 55': delivering the EU's 2030 Climate Target on the way to climate neutrality",
 'CF-SRC-ISTAT2013PUF':'Indagine sui consumi energetici delle famiglie: microdati ad uso pubblico, anno 2013',
 'CF-SRC-DPR4121993':'Decreto del Presidente della Repubblica 26 agosto 1993, n. 412',
 'CF-SRC-UNI113002019':"UNI/TS 11300-2:2019. Prestazioni energetiche degli edifici - Parte 2: Determinazione del fabbisogno di energia primaria e dei rendimenti per la climatizzazione invernale, per la produzione di acqua calda sanitaria, per la ventilazione e per l'illuminazione in edifici non residenziali",
 'CF-SRC-UNI103492016':"UNI 10349-1:2016. Riscaldamento e raffrescamento degli edifici - Dati climatici - Parte 1: Medie mensili per la valutazione della prestazione termo-energetica dell'edificio e metodi per ripartire l'irradianza solare nella frazione diretta e diffusa e per calcolare l'irradianza solare su di una superficie inclinata",
 'CF-SRC-ISO159272005':'ISO 15927-4:2005. Hygrothermal performance of buildings - Calculation and presentation of climatic data - Part 4: Hourly data for assessing the annual energy use for heating and cooling'
}
for r in records:
    if r['id'] in titles: r['title']=titles[r['id']]
    if r['id']=='CF-SRC-RSE2021': r['pages']='';r['number']='21012539';r['container']='Ricerca sul Sistema Energetico (RSE)'
    if r['id']=='CF-SRC-STAFFELL2012': r['number']='11'
p.write_text(json.dumps(records,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
