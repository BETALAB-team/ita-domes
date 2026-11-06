"""Reviewer 02: independent CSV conservation, provenance, input and sky-forcing audit.
No import of the authors' checker, summarizer or build scripts. No output mutations.
"""
from pathlib import Path
import csv, json, hashlib, math, ast
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def rows(p): return [{k:float(v) for k,v in r.items()} for r in csv.DictReader(Path(p).open())]
report = {"draft_sha256":sha(ROOT/"06_claim_drafts/article_claims_v01.md"),"cases":{},"weather":{},"hash_failures":[]}
for site in ("Padova","Bolzano","Roma","Palermo"):
    directory=ROOT/"05_simulations/climate_campaign_v3"/site
    manifest=read(directory/"completion_manifest.json")
    for group in ("inputs","package_source_sha256"):
        for path,digest in manifest[group].items():
            if sha(path)!=digest: report["hash_failures"].append(path)
    for key,filename in (("script_sha256","adapter_snapshot.py"),("correction_module_sha256","single_commit_1c.py"),("config_sha256","eureca_config.json"),("design_sha256","design.json")):
        if sha(directory/filename)!=manifest[key]: report["hash_failures"].append(str(directory/filename))
    summary={r["case"]:r for r in read(directory/"summary.json")}
    allrows={}
    for case in ("seeded","early","late","repeat_seeded","warmup_seeded"):
        data=rows(directory/f"{case}_hourly.csv");allrows[case]=data
        p=read(directory/f"{case}_network_parameters.json")
        hi,hw,hm,he,cm,ca,dt=[p[k] for k in ("Htr_is","Htr_w","Htr_ms","Htr_em","Cm","_air_thermal_capacity","timestep_seconds")]
        maxima=[0.,0.,0.]; continuity=0.; thermostat=0.;failures=0
        for i,r in enumerate(data):
            ta,ts,tm,oa,om,ext,q,inf,rad,conv=[r[k] for k in ("air_temperature_C","surface_C","mass_C","old_air_C","old_mass_C","outdoor_temperature_C","sensible_load_W","infiltration_H_W_K","sigma_radiant","sigma_convective")]
            # Accumulation equals gains plus net conduction/infiltration/load.
            residuals=[
                ca*(ta-oa)/dt-(r["phi_air_W"]+q*conv+hi*(ts-ta)+inf*(ext-ta)),
                -(r["phi_surface_W"]+q*rad+hi*(ta-ts)+hw*(ext-ts)+hm*(tm-ts)),
                cm*(tm-om)/dt-(r["phi_mass_W"]+hm*(ts-tm)+he*(ext-tm))]
            maxima=[max(a,abs(b)) for a,b in zip(maxima,residuals)]
            failures+=max(map(abs,residuals))>1e-6
            if i: continuity=max(continuity,abs(oa-data[i-1]["air_temperature_C"]),abs(om-data[i-1]["mass_C"]))
            if q>1e-6: thermostat=max(thermostat,abs(ta-r["heating_setpoint_C"]))
            else: thermostat=max(thermostat,max(r["heating_setpoint_C"]-ta,0))
        energy=math.fsum(max(r["sensible_load_W"],0) for r in data)*dt/3600000
        peak=max(max(r["sensible_load_W"],0) for r in data)/1000
        setback=[r for r in data if r["heating_setpoint_C"]==19]
        report["cases"][site+"/"+case]={"rows":len(data),"annual_kWh":energy,"peak_kW":peak,"residual_max_W_air_surface_mass":maxima,"residual_failures":failures,"continuity_max_K":continuity,"thermostat_max_error_K":thermostat,"summary_energy_error":energy-summary[case]["annual_sensible_heating_kWh"],"summary_peak_error":peak-summary[case]["peak_sensible_heating_kW"],"setback_hours":len(setback),"setback_positive_hours":sum(r["sensible_load_W"]>1e-6 for r in setback),"setback_energy_kWh":math.fsum(max(r["sensible_load_W"],0) for r in setback)/1000,"maximum_air_C":max(r["air_temperature_C"] for r in data)}
        if sha(directory/f"{case}_hourly.csv")!=summary[case]["hourly_sha256"]:report["hash_failures"].append(str(directory/f"{case}_hourly.csv"))
    for case in ("early","late"):
        a=report["cases"][site+"/"+case];b=report["cases"][site+"/seeded"]
        a["annual_percent"]=100*(a["annual_kWh"]/b["annual_kWh"]-1)
        a["peak_percent"]=100*(a["peak_kW"]/b["peak_kW"]-1)
        a["same_forcing_and_parameters"]=all(all(r[k]==brow[k] for k in ("phi_air_W","phi_surface_W","phi_mass_W","outdoor_temperature_C","infiltration_H_W_K")) for r,brow in zip(allrows[case],allrows["seeded"])) and read(directory/f"{case}_network_parameters.json")==read(directory/"seeded_network_parameters.json")
    report["cases"][site+"/repeat_seeded"]["exact_repeat"]=allrows["repeat_seeded"]==allrows["seeded"]
    epw=ROOT/"05_simulations/climate_inputs_v2"/f"{site}.epw"
    with epw.open() as f:
        for _ in range(8):next(f)
        raw=list(csv.reader(f))
    temps=[float(r[6]) for r in raw]
    cloud=[float(r[23]) for r in raw]
    emissivity=[];sky=[];irsky=[]
    for i,r in enumerate(raw):
        t,dew,press,opaque,ir=[float(r[j]) for j in (6,7,9,23,12)]
        clear=.711+.56*dew/100+.73*(dew/100)**2+.013*math.cos(2*math.pi*(i%24+1)/24)+.00012*(press/100-1000)
        eps=clear+(1-clear)*(opaque*.1*.9)
        emissivity.append(eps);sky.append((t+273)*eps**.25-273)
        irsky.append((ir/5.670374419e-8)**.25-273.15)
    days=[math.fsum(temps[i:i+24])/24 for i in range(0,8760,24)]
    meta=read(epw.with_suffix(".json"))["inputs"]
    report["weather"][site]={"opaque_sky_unique":sorted(set(cloud)),"invalid_opaque_sky_count":sum(not 0<=x<=10 for x in cloud),"source_model_sky_emissivity_range":[min(emissivity),max(emissivity)],"source_model_mean_air_minus_sky_K":math.fsum(t-s for t,s in zip(temps,sky))/8760,"IR_derived_mean_air_minus_sky_K_diagnostic_only":math.fsum(t-s for t,s in zip(temps,irsky))/8760,"IR_Wh_m2_range":[min(float(r[12]) for r in raw),max(float(r[12]) for r in raw)],"HDD18":math.fsum(max(18-t,0) for t in days),"CDD26":math.fsum(max(t-26,0) for t in days),"GHI_kWh_m2":math.fsum(float(r[13]) for r in raw)/1000,"RH_mean_percent":math.fsum(float(r[8]) for r in raw)/8760,"metadata":meta}
    report["weather"][site]["first_hour"]={k:allrows["seeded"][0][k] for k in ("phi_air_W","phi_surface_W","phi_mass_W","outdoor_temperature_C")}
data_root=ROOT.parents[1]
rawpath=data_root/"istat/ECH_2013_IT/MICRODATI/ECH_Microdati_Anno_2013.txt"
compact=data_root/"istat_microdata_2013.csv"
report["survey"]={}
for name,p,key in (("raw",rawpath,"nprog"),("compact",compact,"id")):
    with p.open(encoding="utf-8-sig") as f:
        rr=list(csv.DictReader(f,delimiter="\t" if name=="raw" else ";"))
    record=next(r for r in rr if r[key].strip()=="3523")
    report["survey"][name]={"sha256":sha(p),"records":len(rr),"values":{k:record[k] for k in ("reg","q_3_11A","q_3_11B","q_3_11C","q_3_13_2","q_2_32C","q_2_32D")}}
    if name=="compact":
        eligible=[r for r in rr if r["reg"]=="5" and all(0<float(r[k])<8 for k in ("q_3_11A","q_3_11B","q_3_11C"))]
        report["survey"]["eligible_count"]=len(eligible);report["survey"]["minimum_id"]=min(int(r["id"]) for r in eligible)
ledger=read(ROOT/"06_claim_drafts/claim_ledger_v01.json")
report["ledger"]={"count":len(ledger["claims"]),"hash_matches":ledger["draft_sha256"]==report["draft_sha256"],"all_claim_text_in_draft":all(c["exact_claim"] in (ROOT/"06_claim_drafts/article_claims_v01.md").read_text(encoding="utf-8") for c in ledger["claims"])}
report["conference_gas_percent"]=[100*(18090/18073-1),100*(17115/15151-1)]
(OUT/"independent_audit.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps({"cases":len(report["cases"]),"max_node_residual_W":max(max(c["residual_max_W_air_surface_mass"]) for c in report["cases"].values()),"hash_failures":report["hash_failures"],"weather":report["weather"],"survey":report["survey"],"ledger":report["ledger"]},indent=2))
