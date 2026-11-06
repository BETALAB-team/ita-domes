// Native editable report for independent-schedule experiments.
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {spawnSync} from 'node:child_process';
const out=path.resolve(process.argv[2]);
const data=JSON.parse(await fs.readFile(path.join(out,'report_data.json'),'utf8'));
const {config:cfg,manifest:m}=data,rt=cfg.report,build=path.join(out,'report_build');
await fs.mkdir(build,{recursive:true});process.env.RUNTIME_NODE_MODULES=rt.node_modules;
const {Presentation,PresentationFile}=await import(pathToFileURL(path.join(rt.node_modules,'@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(path.join(rt.skill_dir,'container_tools/artifact_tool_utils.mjs')).href);
const mark=spawnSync(rt.node,[path.join(rt.skill_dir,'container_tools/mark_artifact_operation_started.mjs'),'--operation-kind','create','--expected-output-count','1','--output-format','pptx'],{cwd:rt.skill_dir,encoding:'utf8',windowsHide:true});
if(mark.status!==0)throw Error(mark.stderr);
const p=Presentation.create({slideSize:{width:1280,height:720}}),slides=[],tableOwners=[],chartOwners=[];
const names={known_summary:'Known aggregates',istat_profile:'ISTAT profiles'};
const f=x=>x==null?'—':Number(x).toFixed(1);
function text(s,value,x,y,w,h,size=24,bold=false,color='#223746'){
 const q=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 q.text=value;q.text.style={typeface:'Arial',fontSize:size,bold,color,autoFit:'none'};return q;
}
function slide(title){const s=p.slides.add();s.background.fill='#FFFFFF';slides.push(s);text(s,title,58,35,1164,65,39,true,'#145c85');text(s,`${slides.length}   GRINS Energy Studio 0.5 · Independent schedules`,58,680,1164,24,14,false,'#667985');return s;}
function table(s,values,widths,y=140,height=430,size=23){
 tableOwners.push(slides.length);const n=values.length,k=values[0].length;
 const t=s.tables.add({rows:n,columns:k,left:58,top:y,width:1164,height,values,columnWidths:widths.map(x=>x*1164)});
 t.styleOptions={headerRow:false,bandedRows:false};t.borders.assign({fill:'#D1DCE3',width:.6});
 t.cells.block({row:0,column:0,rowCount:n,columnCount:k}).assign({margins:{top:9,bottom:9,left:12,right:12},anchor:'center'});
 for(let r=0;r<n;r++){t.rows[r].height=height/n;for(let c=0;c<k;c++){let cell=t.getCell(r,c);cell.fill=r===0?'#E8F0F5':'#FFFFFF';cell.text.style={typeface:'Arial',fontSize:size,bold:r===0,color:'#223746',alignment:'left',autoFit:'none'};}}
}
let s=slide('A benchmark with independent household inputs');
text(s,`${m.scenario_count} cases · ${cfg.simulation.area_m2} m²`,58,145,1164,70,44,true);
text(s,`Reference: ${m.anchor_scenario.replaceAll('__',' / ')}`,58,255,1164,70,29);
text(s,'EnergyPlus gets detailed schedules.\nThe hybrid gets aggregates or ISTAT-predicted behaviour.',58,360,1164,120,33);
text(s,`Schedule evidence: ${data.profiles.evidence_kind}. Model-to-model comparison; no measured-building validation.`,58,550,1164,90,26);
s=slide('What changes from the shared-input benchmark');
table(s,[['Component','Shared-input baseline','Independent benchmark'],['Household inputs to EP','ISTAT-derived household means','Separate schedule specification'],['Target behaviour for hybrid','Same survey profile used by EP','Known aggregates OR survey prediction'],['Archetype assignment','Survey labels','Demographics only; frozen model'],['Target EP outcomes','Evaluation only','Evaluation only; predictions frozen first'],['Status of earlier results','Conditional on shared inputs','Preserved, not pooled with new results']],[.25,.35,.4],135,450,22);
text(s,'The geometry and weather remain shared controls. Independence here concerns household behaviour, not every modelling assumption.',58,610,1164,55,21);
s=slide('The three methods, simply');
table(s,[['Method','Think of it as…','What it can tell us'],['1. EnergyPlus','Run the same room through a full year with the actual specified household schedule.','Hourly and annual sensible loads'],['2. Hybrid','Learn from one simulated room, then adjust for climate, exposure and household behaviour.','Annual heating and cooling estimates'],['3. Data only','Look up what similar ISTAT households report or what the appliance model reconstructs.','Hours and appliance kWh; not thermal kWh']],[.19,.53,.28],140,435,25);
text(s,'The hybrid does not simulate a target room, use a 1R1C model, or create an hourly target load profile.',58,600,1164,60,23);
s=slide('Two tests against the same EnergyPlus cases');
table(s,[['Test','Inputs allowed to the hybrid','Question'],['Known aggregates','Actual X, mean presence, equivalent heating/cooling hours and annual appliance kWh. No target hourly schedules.','How much error comes from the thermal transfer and lost timing?'],['ISTAT profiles','Demographics identify A1–A4. Frozen survey cohorts supply members, hours and appliance indices.','What additional error appears when behaviour must be predicted?'],['Shared reference','One detailed EP run; the same reference fit in both tests. No target outcome enters fitting.','Can one known case transfer to the other contexts?']],[.2,.48,.32],140,430,23);
text(s,'Supplying appliance kWh makes known-aggregate appliance agreement an identity. It is excluded from accuracy metrics.',58,600,1164,65,22);
s=slide('Households and demographic-only assignments');
const assignment=new Map(data.assignments.map(r=>[r.profile_id,r]));
table(s,[['Profile','Members','Age range','Education*','Assigned A'],...data.profiles.profiles.map(r=>[r.id+' · '+r.label,String(r.members),r.demographics?`${r.demographics.agemin}–${r.demographics.agemax}`:'—',r.demographics?f(r.demographics.education):'—',assignment.has(r.id)?'A'+assignment.get(r.id).archetype:'not requested'])],[.4,.13,.16,.15,.16],140,400,22);
text(s,'*Education is the existing 0–1 attainment proxy. P2_shift keeps P2 demographics and totals but moves all daily schedules by eight hours.',58,565,1164,90,23);
s=slide('The transfer equations stay explicit');
text(s,'Mean internal gains',58,125,1164,38,26,true);
text(s,'G = X × sensible watts/person × mean presence + e × 1000 / 8760 × heat fraction',58,178,1164,72,24);
text(s,'From reference A to target B',58,280,1164,38,26,true);
text(s,'BalanceB = BalanceA + (mean setpointB − mean setpointA) − (GB/HB − GA/HA)\nDemandB = DemandA × (HB/HA) × (adjusted degree sumB / degree sumA)',58,337,1164,125,26);
text(s,'H is the heat-loss conductance. Heating hours determine the mean heating setpoint. The reference fit supplies the balance points and solar response.',58,510,1164,95,25);
text(s,'For ISTAT targets, mean presence remains an assumption. The known-aggregate test supplies the independent schedule’s mean presence.',58,615,1164,48,20);
s=slide('Annual accuracy, excluding the reference');
table(s,[['Test / end use','Cases','MAE kWh','RMSE kWh','Bias kWh','CVRMSE %'],...data.metrics.map(r=>[`${names[r.method]} / ${r.end_use.split('_')[0]}`,String(r.n),f(r.mae_kwh),f(r.rmse_kwh),f(r.bias_kwh),f(r.cvrmse_percent)])],[.33,.09,.145,.145,.145,.145],140,390,22);
text(s,`${m.energyplus_processes_run} EP simulations. Reference: ${f(m.reference_seconds)} s. Total EP process time: ${f(m.energyplus_seconds_sum)} s.`,58,555,1164,50,23);
text(s,'Ideal sensible loads, not heating fuel or HVAC electricity. These selected synthetic cases are not a population sample.',58,615,1164,48,21);
for(const spec of data.plots){
 s=slide(spec.title);chartOwners.push(slides.length);
 const count=spec.panels.length,gap=30,width=(1164-gap*(count-1))/count;
 for(const [i,panel] of spec.panels.entries()){
  const round=v=>v==null?null:Number(v.toFixed(3));const scatter=panel.type==='scatter';
  const series=panel.series.map(r=>({name:r.name,values:r.values.map(round),...(scatter?{xValues:r.x.map(round)}:{}),fill:r.color,line:{fill:r.color,width:scatter?0:2},marker:{symbol:scatter?'circle':'none',size:6}}));
  const chart=s.charts.add(panel.type,{position:{left:58+i*(width+gap),top:140,width,height:465},title:panel.title,titleTextStyle:{fontSize:25},categories:panel.categories??panel.series[0].x.map(String),series,hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:15}},barOptions:{direction:'column',grouping:'clustered'},scatterOptions:{style:'marker'},lineOptions:{smooth:false},xAxis:{title:{text:panel.xlabel,textStyle:{fontSize:17}},textStyle:{fontSize:14},numberFormatCode:'0'},yAxis:{title:{text:panel.ylabel,textStyle:{fontSize:17}},textStyle:{fontSize:14},numberFormatCode:panel.ylabel.includes('fraction')?'0.0':'0'},dataLabels:{showValue:false},displayBlanksAs:'gap'});
  applyPresentationChartFont(chart,{fontFamily:'Arial'});
 }
 const caption=spec.key==='appliance_prediction'?'Known-aggregate appliance agreement is an identity, not validation. ISTAT supplies a separate prediction.':spec.note;
 text(s,caption,58,620,1164,50,21);
}
s=slide('What this benchmark can and cannot establish');
table(s,[['Supported interpretation','Remaining limitation'],['Transfer error under non-ISTAT schedules','Synthetic behaviour is not evidence of actual Italian household behaviour.'],['Archetype assignment from demographics alone','The survey-based behaviour predictor still requires independent empirical validation.'],['Timing sensitivity at equal aggregate inputs','An annual aggregate model cannot distinguish schedules with identical summaries.'],['Climate and exposure transfer','One zone, fixed ground temperature, simple envelope and ideal sensible loads remain assumptions.']],[.46,.54],140,425,24);
text(s,'Larger errors are retained. Neither target results nor a desired error level are used to tune the fitted model.',58,615,1164,55,23);
s=slide('Replace the synthetic inputs when data are available');
text(s,'Configurable input paths',58,135,1164,40,29,true);
text(s,'Choose independent_schedules in the app. Load a profiles JSON or keep the built-in synthetic cases. Each profile may use weekday/weekend schedules or a complete hourly CSV.',58,195,1164,110,26);
text(s,'Hourly CSV fields',58,345,1164,40,29,true);
text(s,'hour (1–8760), occupancy (0–1), heat (°C), cool (°C), appliance_w (W).\nKeep demographics separate. Preserve source and evidence labels.',58,405,1164,100,25);
text(s,'EnergyPlus 25.1 implementation references',58,550,1164,35,23,true);
text(s,'bigladdersoftware.com/epx/docs/25-1/input-output-reference/group-schedules.html\nbigladdersoftware.com/epx/docs/25-1/input-output-reference/group-internal-gains-people-lights-other.html',58,600,1164,62,17);
const candidate=path.join(build,'candidate.pptx');await(await PresentationFile.exportPptx(p)).save(candidate);
await fs.mkdir(path.join(out,'presentation'),{recursive:true});
for(let i=0;i<slides.length;i++){const blob=await p.export({slide:slides[i],format:'png',scale:1});await fs.writeFile(path.join(build,`slide-${i+1}.png`),new Uint8Array(await blob.arrayBuffer()));}
const result=await finalizePresentation({workspaceDir:out,candidatePath:candidate,finalPath:path.join(out,'presentation/independent_benchmark.pptx'),explicitTotalSlideCount:slides.length,pythonExecutable:rt.python,integrityValidatorPath:path.join(rt.skill_dir,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(rt.skill_dir,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',...tableOwners.flatMap(n=>['--require-native-table-slide',String(n)])],requiredNativeTableOwnerSlides:tableOwners,requiredNativeChartOwnerSlides:chartOwners,materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:['Arial']},verifyArtifactToolImport:true,receiptPath:path.join(build,'validation.json')});
console.log(JSON.stringify({path:result.finalPath,sha256:result.finalSha256,slides:slides.length}));
