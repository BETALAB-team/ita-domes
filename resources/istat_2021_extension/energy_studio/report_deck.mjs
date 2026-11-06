// Editable, data-driven PowerPoint run report using the installed artifact runtime.
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {spawnSync} from 'node:child_process';
const out=path.resolve(process.argv[2]);
const data=JSON.parse(await fs.readFile(path.join(out,'report_data.json'),'utf8'));
const cfg=data.config,rt=cfg.report,build=path.join(out,'report_build');
await fs.mkdir(build,{recursive:true});
process.env.RUNTIME_NODE_MODULES=rt.node_modules;
const {Presentation,PresentationFile}=await import(pathToFileURL(path.join(rt.node_modules,'@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(path.join(rt.skill_dir,'container_tools/artifact_tool_utils.mjs')).href);
const marked=spawnSync(rt.node,[path.join(rt.skill_dir,'container_tools/mark_artifact_operation_started.mjs'),'--operation-kind','create','--expected-output-count','1','--output-format','pptx'],{cwd:rt.skill_dir,encoding:'utf8',windowsHide:true});
if(marked.status!==0)throw Error('Could not mark report operation: '+marked.stderr);
const p=Presentation.create({slideSize:{width:1280,height:720}});
const colors={energyplus:'#145c85',hybrid_signature:'#388670',hybrid_istat:'#cf7f32',istat_only:'#778894'};
const names={energyplus:'EnergyPlus',hybrid_signature:'Climate + gains hybrid',hybrid_istat:'ISTAT-ratio baseline',istat_only:'ISTAT only'};
const slides=[];
const extraChartOwners=[];
function text(s,value,x,y,w,h,size=24,bold=false,color='#223746'){
 const q=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 q.text=value;q.text.style={typeface:'Arial',fontSize:size,bold,color,autoFit:'none'};return q;
}
function slide(title){const s=p.slides.add();s.background.fill='#FFFFFF';text(s,title,58,35,1164,65,40,true,'#145c85');slides.push(s);text(s,`${slides.length}   GRINS Energy Studio · ${path.basename(out)}`,58,680,1160,25,14,false,'#667985');return s;}
function table(s,values,y=145,widths=null,height=440,size=23){
 const n=values.length,m=values[0].length;
 const t=s.tables.add({rows:n,columns:m,left:58,top:y,width:1164,height,values,columnWidths:widths?widths.map(x=>x*1164):Array(m).fill(1164/m)});
 t.styleOptions={headerRow:false,bandedRows:false};t.borders.assign({fill:'#D1DCE3',width:.6});
 t.cells.block({row:0,column:0,rowCount:n,columnCount:m}).assign({margins:{top:9,bottom:9,left:12,right:12},anchor:'center'});
 for(let r=0;r<n;r++){t.rows[r].height=height/n;for(let c=0;c<m;c++){
  const cell=t.getCell(r,c);cell.fill=r===0?'#E8F0F5':'#FFFFFF';cell.text.style={typeface:'Arial',fontSize:size,bold:r===0,color:'#223746',alignment:'left',autoFit:'none'};
 }}return t;
}
function chart(s,type,title,x,series,categories,yTitle){
 // Chart precision is display precision; full values remain in the CSV/JSON.
 series=series.map(s=>({...s,values:s.values.map(v=>v==null?null:Number(v.toFixed(3)))}));
 const ch=s.charts.add(type,{position:{left:x,top:145,width:560,height:455},title,titleTextStyle:{fontSize:24},categories,series,hasLegend:series.length>1,legend:{position:'bottom',textStyle:{fontSize:15}},barOptions:{direction:'column',grouping:'clustered'},lineOptions:{smooth:false},xAxis:{title:{text:type==='line'?'Number of groups':'Climate',textStyle:{fontSize:17}},textStyle:{fontSize:15}},yAxis:{title:{text:yTitle,textStyle:{fontSize:17}},textStyle:{fontSize:15},numberFormatCode:'0'},dataLabels:{showValue:false}});
 applyPresentationChartFont(ch,{fontFamily:'Arial'});return ch;
}
const sm=cfg.simulation,a=cfg.analysis,m=data.manifest;
let s=slide('Reference dwelling experiment');
text(s,`${sm.area_m2} m² · ${m.scenario_count} scenarios`,58,160,1164,70,42,true);
text(s,'Detailed simulation, one-reference climate + ISTAT transfer and survey-only indices',58,255,1110,95,30);
text(s,`Reference case: ${m.anchor_scenario.replaceAll('__',' / ')}`,58,385,1164,70,26);
text(s,'Research prototype. Annual sensible heating and cooling demand, with appliance electricity reported separately.',58,520,1110,95,26);
s=slide('Experiment configuration');
table(s,[['Parameter','Setting'],['Climates',cfg.climates.filter(c=>c.enabled).map(c=>c.name).join(', ')],['Contexts',cfg.contexts.filter(c=>c.enabled).map(c=>c.name).join(', ')],['Archetypes',`${a.n_archetypes}; ${a.archetype_mode}`],['Energy groups',`${a.group_selection}; H=${m.selected_energy_groups?.heating??a.heating_groups}, E=${m.selected_energy_groups?.electricity??a.electricity_groups}`],['Grouping factors',a.group_factors.join(', ')+(a.electricity_include_cooling?' (+ cooling for E)':'')],['Envelope U-values',`Wall ${sm.wall_u}, roof ${sm.roof_u}, floor ${sm.floor_u}, window ${sm.window_u} W/m²K`],['Controls',`Heating ${sm.heat_setpoint_c} °C / setback ${sm.heat_setback_c} °C; cooling ${sm.cool_setpoint_c} °C`]],125,[.28,.72],510,22);
s=slide('Exposure configurations');
table(s,[['Context','Outdoor walls','Roof','Floor'],...cfg.contexts.filter(c=>c.enabled).map(c=>[c.name,c.exposed_walls.join(', '),c.roof,c.floor])],155,[.22,.38,.2,.2],330,24);
text(s,'Adiabatic boundaries represent neighbouring spaces at similar temperatures. Contexts are explicit experimental assumptions, not directly observed building geometries.',58,540,1164,95,24);
s=slide('Three calculation methods');
table(s,[['Method','Calculation','EnergyPlus runs'],['1. Detailed','Annual single-zone simulation with scenario weather, exposures and survey-informed schedules','Every scenario'],['2. Hybrid','Reference kWh × exposure ratio × adjusted degree-sum ratio. Weather and ISTAT inputs; no target thermal simulation.','One reference'],['3. ISTAT only','Heating/cooling hours and reconstructed appliance kWh. No thermal kWh.','None']],135,[.2,.56,.24],390,23);
text(s,'The original ISTAT-hours ratio is retained as a comparison baseline. The new hybrid learns its weather response from the reference case only.',58,555,1164,100,24);
s=slide('Energy-group diminishing returns');
for(const [i,key] of ['heating','electricity'].entries()){
 const rows=data.curves[key];const total=rows[0].wcss||1;
 chart(s,'line',key==='heating'?'Heating':'Appliance electricity',i?660:58,[{name:'Remaining WCSS (%)',values:rows.map(r=>100*r.wcss/total),line:{fill:'#145c85',width:2},fill:'#145c85'}],rows.map(r=>String(r.k)),'Between-cell dispersion (%)');
}
text(s,'Dispersion concerns training cell means, not total household variation. Fixed group counts remain user choices.',58,620,1164,45,21);
s=slide('Annual demand by climate');
const climates=cfg.climates.filter(c=>c.enabled).map(c=>c.name);
const methods=[...new Set(data.annual.map(r=>r.method))].filter(x=>x!=='energyplus'||sm.run_full_benchmark);
for(const [i,enduse] of ['heating_kwh','cooling_kwh'].entries()){
 const series=methods.filter(method=>data.annual.some(r=>r.method===method&&r[enduse]!=null)).map(method=>({name:names[method],fill:colors[method],values:climates.map(climate=>{const rows=data.annual.filter(r=>r.method===method&&r.climate===climate&&r[enduse]!=null);return rows.length?rows.reduce((v,r)=>v+r[enduse],0)/rows.length:null;})}));
 if(series.length)chart(s,'bar',i?'Sensible cooling':'Sensible heating',i?660:58,series,climates,'Mean kWh/year');
}
text(s,'Unweighted scenario means, not population estimates. Thermal demand excludes HVAC efficiency.',58,620,1164,45,21);
s=slide('Accuracy and computational effort');
const fmt=n=>n==null?'—':Number(n).toFixed(1);
table(s,[['Method / end use','Cases*','MAE kWh','RMSE kWh','Bias kWh','CVRMSE %'],...data.metrics.map(r=>[`${names[r.method]} / ${r.end_use.split('_')[0]}`,String(r.n_excluding_anchor),fmt(r.mae_kwh),fmt(r.rmse_kwh),fmt(r.bias_kwh),fmt(r.cvrmse_percent)])],135,[.34,.1,.14,.14,.14,.14],330,21);
text(s,`*Excludes the reference case. New EnergyPlus processes: ${m.energyplus_processes_run}; reference ${m.reused_reference?'reused':'simulated'}: ${fmt(m.hybrid_anchor_simulation_seconds)} s.`,58,495,1164,70,23);
text(s,'Process timings exclude common data preparation. Agreement with EnergyPlus does not establish accuracy against real buildings.',58,575,1164,65,23);
for(const spec of data.plots??[]){
 s=slide(spec.key==='istat_transfer_ratios'?'ISTAT-ratio baseline factors':spec.title);extraChartOwners.push(slides.length);
 const count=spec.panels.length,gap=24,width=(1164-gap*(count-1))/count;
 for(const [i,panel] of spec.panels.entries()){
  const round=v=>v==null?null:Number(v.toFixed(3));
  const scatter=panel.type==='scatter'||panel.type==='duration';
  const series=panel.series.map(row=>({name:row.name,values:row.values.map(round),...(scatter?{xValues:row.x.map(round)}:{}),fill:row.color,line:{fill:row.color,width:panel.type==='scatter'?0:2},marker:{symbol:panel.type==='scatter'?'circle':'none',size:5}}));
  const xlabel=panel.xlabel;
  const ylabel=panel.ylabel==='Estimate minus EnergyPlus (kWh/year)'?'Error (kWh/year)':panel.ylabel;
  const chart=s.charts.add(scatter?'scatter':panel.type,{position:{left:58+i*(width+gap),top:145,width,height:455},title:panel.title,titleTextStyle:{fontSize:count===3?23:25},categories:panel.categories,series,hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:count===3?13:15}},barOptions:{direction:'column',grouping:'clustered'},scatterOptions:{style:panel.type==='scatter'?'marker':'line'},lineOptions:{smooth:false},xAxis:{title:{text:xlabel,textStyle:{fontSize:count===3?15:17}},textStyle:{fontSize:count===3?12:15},numberFormatCode:'0',...(panel.limits?.x?{min:panel.limits.x[0],max:panel.limits.x[1]}:{})},yAxis:{title:{text:ylabel,textStyle:{fontSize:count===3?15:17}},textStyle:{fontSize:count===3?13:15},numberFormatCode:(panel.type==='duration'||spec.key==='istat_transfer_ratios')?'0.0':'0',...(panel.limits?.y?{min:panel.limits.y[0],max:panel.limits.y[1]}:{})},dataLabels:{showValue:false},displayBlanksAs:'gap'});
  applyPresentationChartFont(chart,{fontFamily:'Arial'});
  chart.xAxis={tickLabelPosition:'low',title:{text:xlabel,textStyle:{typeface:'Arial',fontSize:count===3?15:17}},textStyle:{typeface:'Arial',fontSize:count===3?12:15},numberFormatCode:'0',...(panel.limits?.x?{min:panel.limits.x[0],max:panel.limits.x[1]}:{})};
 }
 text(s,(spec.key==='istat_transfer_ratios'?'Baseline only. ':'')+spec.note,58,610,1164,65,21);
}
s=slide('The three methods in plain language');
const hybridTables=[slides.length];
table(s,[['Method','Think of it this way'],['1. EnergyPlus','Build a virtual home for every case and simulate its year in the chosen weather and surroundings.'],['2. Hybrid','Watch how one simulated home responds to weather. Use the survey and exposure settings to estimate how another case would differ.'],['3. ISTAT only','Use reported heating/cooling hours and estimated appliance electricity directly. The survey table alone does not identify thermal kWh.']],145,[.22,.78],390,25);
text(s,'The hybrid uses a fitted equation with weather inputs. It does not simulate target indoor temperatures or use a 1R1C model.',58,580,1164,75,24);
s=slide('Hybrid transfer process');hybridTables.push(slides.length);
table(s,[['Step','Calculation'],['1. Reference run',m.anchor_scenario.replaceAll('__',' / ')+'. Fit daily heating and cooling against weather using only this case.'],['2. Target information','Read target EPW weather, exposed surfaces, ISTAT operating hours, household size and appliance electricity.'],['3. Annual transfer','Reference EP kWh × target/reference conductance × target/reference adjusted degree sum.'],['4. Evaluation','Save and hash all predictions before target EP runs. Target results enter only the error calculation.']],135,[.24,.76],400,23);
text(s,'The same fitted reference parameters serve every target. Separate heating and cooling relations allow different seasonal responses.',58,575,1164,80,24);
if(data.signature_fit){
 s=slide('Parameters learned from the reference');hybridTables.push(slides.length);
 const fits=data.signature_fit.fits;
 table(s,[['End use','Balance °C','Solar coefficient','Active days','Daily fit RMSE kWh'],...Object.entries(fits).map(([use,f])=>[use.split('_')[0],fmt(f.balance_c),fmt(f.solar_coefficient),String(f.active_days??0),fmt(f.daily_fit_rmse_kwh)])],160,[.2,.2,.2,.16,.24],280,24);
 const atEdge=Object.values(fits).some(f=>f.balance_at_search_edge||f.solar_at_search_edge);
 text(s,'The balance temperature marks the fitted heating/cooling threshold. The solar coefficient describes the response to the reference irradiation feature.',58,475,1164,80,24);
 text(s,atEdge?'At least one fitted parameter reaches a search boundary. Its value and transferability need scrutiny. Daily fit error is in-sample, not target validation.':'All fit inputs come from the reference run. Daily fit error is in-sample, not target validation. Target energy is never used to choose these parameters.',58,580,1164,85,23);
}
s=slide('Worked annual transfer');hybridTables.push(slides.length);
const transfers=data.signature_transfer??data.transfer??[];
const exampleKey=transfers.find(r=>r.scenario==='Palermo__Apartment__A4')?.scenario??transfers.find(r=>r.scenario!==m.anchor_scenario)?.scenario??m.anchor_scenario;
const examples=transfers.filter(r=>r.scenario===exampleKey && r.end_use!=='appliance_kwh');
text(s,m.anchor_scenario.replaceAll('__',' / ')+' to '+exampleKey.replaceAll('__',' / '),58,125,1164,65,25);
table(s,[['End use','Ref. EP kWh','Exposure ratio','Degree ratio','Estimate kWh','Target EP kWh*'],...examples.map(r=>[r.end_use.split('_')[0],fmt(r.reference_energy_kwh),r.conductance_ratio==null?'—':r.conductance_ratio.toFixed(3),r.target_degree_sum==null?'—':(r.target_degree_sum/r.reference_degree_sum).toFixed(3),fmt(r.estimate_kwh),fmt(data.annual.find(a=>a.scenario===exampleKey&&a.method==='energyplus')?.[r.end_use])])],215,[.15,.18,.17,.16,.17,.17],280,22);
text(s,'*Target EP is shown only for evaluation. Appliance electricity retains its survey-derived transfer. The audit CSV records every term and its support count.',58,540,1164,100,23);
s=slide('Meaning of the transfer adjustments');hybridTables.push(slides.length);
table(s,[['Input','Effect on the estimate'],['Target weather','Changes the annual heating/cooling degree sums. The reference fit supplies the balance points and solar coefficients.'],['Exposed surfaces','Changes nominal heat-loss conductance, ground coupling and projected window irradiation.'],['ISTAT heating/cooling hours','Changes assumed mean thermostat settings. Hours do not multiply the weather ratio a second time.'],['ISTAT household and appliances','Changes mean internal gains. Added gains lower the heating threshold and increase cooling demand.']],135,[.27,.73],420,23);
text(s,'These are static transfer assumptions. Thermal mass, delayed heat flows and target indoor temperatures remain outside the model.',58,585,1164,70,23);
s=slide('Interpretation and reproducibility');
text(s,'Annual estimates and an exploratory benchmark',58,140,1164,45,28,true);
const chosen=data.signature_fit?'hybrid_signature':'hybrid_istat';
const accuracy=data.metrics.filter(r=>r.method===chosen).map(r=>r.end_use.split('_')[0]+' CVRMSE '+fmt(r.cvrmse_percent)+'%').join(', ');
text(s,accuracy?'Selected hybrid: '+accuracy+'. Compare pooled bias with case-level absolute error. These are model-to-model results for the configured scenarios, not measured-building validation.':'No non-reference benchmark is available. Predictions remain unvalidated for the target cases.',58,205,1164,140,26);
text(s,'Saved inputs and calculations',58,390,1164,45,28,true);
text(s,'config.json and manifest.json preserve the run settings.\nsignature_fit.json and signature_transfer_audit.csv expose the fitted model and every transfer term. Predictions are saved before benchmarking.',58,450,1164,120,24);
text(s,'Daily energy is used to fit the reference only. Target outputs are annual totals. The model has no thermal capacity or indoor-temperature state.',58,590,1164,65,22);
const climateErrors=[];
for(const climate of climates)for(const use of ['heating_kwh','cooling_kwh']){
 const pairs=data.annual.filter(r=>r.method===chosen && r.climate===climate && !r.is_anchor && r[use]!=null).map(r=>({estimate:r[use],benchmark:data.annual.find(e=>e.method==='energyplus'&&e.scenario===r.scenario)?.[use]})).filter(r=>r.benchmark!=null);
 if(!pairs.length)continue;
 const mean=pairs.reduce((s,r)=>s+r.benchmark,0)/pairs.length,rmse=Math.sqrt(pairs.reduce((s,r)=>s+(r.estimate-r.benchmark)**2,0)/pairs.length);
 climateErrors.push([climate,use.split('_')[0],String(pairs.length),fmt(pairs.reduce((s,r)=>s+Math.abs(r.estimate-r.benchmark),0)/pairs.length),fmt(rmse),mean>0?fmt(100*rmse/mean):'—']);
}
if(climateErrors.length){
 s=slide('Hybrid accuracy by climate');hybridTables.push(slides.length);
 table(s,[['Climate','End use','Cases*','MAE kWh','RMSE kWh','CVRMSE %'],...climateErrors],140,[.2,.18,.12,.17,.17,.16],420,22);
 text(s,'*Reference excluded. Relative errors depend on each climate’s mean demand. Palermo has low heating demand; assess its kWh error alongside the percentage.',58,580,1164,85,23);
}
s=slide('Method basis and remaining limits');
text(s,'Building-specific balance temperatures',58,135,1164,45,28,true);
text(s,'CIBSE TM41 relates the degree-day base temperature to the building and system energy balance. This experiment adds explicit survey and exposure corrections to a fitted reference signature.',58,200,1164,115,25);
text(s,'CIBSE, TM41: Degree days, theory and application (2006)\nhttps://www.cibse.org/knowledge-research/knowledge-portal',58,340,1164,75,20);
text(s,'Scope of this implementation',58,455,1164,45,28,true);
text(s,'Ideal sensible loads only. Constant ground temperature, assumed daily schedules and approximate window solar exposure. No measured thermal-energy calibration, complex shading or HVAC efficiency. Appliance agreement is not independent validation.',58,515,1164,130,24);
const candidate=path.join(build,'candidate.pptx');await(await PresentationFile.exportPptx(p)).save(candidate);
await fs.mkdir(path.join(out,'presentation'),{recursive:true});
for(let i=0;i<slides.length;i++){
 const blob=await p.export({slide:slides[i],format:'png',scale:1});
 await fs.writeFile(path.join(build,`slide-${i+1}.png`),new Uint8Array(await blob.arrayBuffer()));
}
const tableOwners=[2,3,4,7,...hybridTables];
const result=await finalizePresentation({workspaceDir:out,candidatePath:candidate,finalPath:path.join(out,'presentation/run_report.pptx'),explicitTotalSlideCount:slides.length,pythonExecutable:rt.python,integrityValidatorPath:path.join(rt.skill_dir,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(rt.skill_dir,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',...tableOwners.flatMap(n=>['--require-native-table-slide',String(n)])],requiredNativeTableOwnerSlides:tableOwners,requiredNativeChartOwnerSlides:[5,...(data.annual.some(r=>r.heating_kwh!=null)?[6]:[]),...extraChartOwners],materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:['Arial']},verifyArtifactToolImport:true,receiptPath:path.join(build,'validation.json')});
console.log(JSON.stringify({finalPath:result.finalPath,sha:result.finalSha256}));
