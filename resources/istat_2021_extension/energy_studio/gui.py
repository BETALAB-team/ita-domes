"""Native Windows/Tk front end: edit configuration, validate, run, cancel, inspect."""
import copy
import json
import os
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import ttk,filedialog,messagebox
from .config import default_config,validate,save,load


class Studio:
    def __init__(self,root,cfg):
        cfg=copy.deepcopy(cfg);cfg.setdefault('benchmark',default_config()['benchmark'])
        self.root=root;self.cfg=copy.deepcopy(cfg);self.fields={};self.messages=queue.Queue();self.cancel=threading.Event();self.worker=None;self.last_output=None
        root.title('GRINS Energy Studio');root.geometry('1160x830');root.minsize(980,690)
        style=ttk.Style();style.theme_use('clam');style.configure('TNotebook.Tab',padding=(15,8));style.configure('TButton',padding=7)
        style.configure('Title.TLabel',font=('Segoe UI',21,'bold'),foreground='#145c85');style.configure('TLabel',font=('Segoe UI',10))
        top=ttk.Frame(root,padding=(20,15));top.pack(fill='x')
        ttk.Label(top,text='GRINS Energy Studio',style='Title.TLabel').pack(anchor='w')
        ttk.Label(top,text='Survey archetypes  /  a reference dwelling  /  three modelling approaches').pack(anchor='w',pady=(3,8))
        toolbar=ttk.Frame(top);toolbar.pack(fill='x')
        for title,cmd in [('Load configuration',self.load_config),('Save configuration',self.save_config),('Reset defaults',self.reset),('Validate',self.check)]:ttk.Button(toolbar,text=title,command=cmd).pack(side='left',padx=(0,8))
        self.summary=ttk.Label(top);self.summary.pack(anchor='w',pady=(10,0))
        self.tabs=ttk.Notebook(root);self.tabs.pack(fill='both',expand=True,padx=20)
        paths=self.tab('Data & execution');analysis=self.tab('Clustering');building=self.tab('Building & controls');context=self.tab('Weather & exposures');advanced=self.tab('Advanced JSON')
        self.field(paths,('benchmark','mode'),'Benchmark household inputs',choices=['shared_istat','independent_schedules'])
        self.field(paths,('benchmark','profiles_file'),'Independent profiles JSON (blank = synthetic test)',browse='file')
        self.field(paths,('benchmark','anchor_profile'),'Independent reference profile ID')
        self.field(paths,('benchmark','compare_istat'),'Also test ISTAT-predicted profiles',kind=bool)
        self.field(paths,('benchmark','demographic_model'),'Frozen demographic model (existing labels)',browse='file')
        for key,label in [('households','Household analysis CSV'),('energyplus','EnergyPlus 25.1 executable'),('output_root','Results directory')]:self.field(paths,('paths',key),label,browse='directory' if key=='output_root' else 'file')
        self.field(paths,('paths','reference_run'),'Reuse reference from completed run (optional)',browse='directory')
        self.field(paths,('simulation','enabled'),'Enable EnergyPlus reference / benchmark',kind=bool)
        self.field(paths,('simulation','run_full_benchmark'),'Simulate every scenario for benchmarking (off = reference run only)',kind=bool)
        self.field(paths,('simulation','workers'),'Concurrent EnergyPlus processes',kind=int)
        self.field(paths,('simulation','timeout_seconds'),'Maximum seconds per simulation',kind=int)
        self.field(paths,('hybrid','method'),'Hybrid method',choices=['climate_exposure_gains','istat_ratio'])
        self.field(paths,('hybrid','index_source'),'ISTAT transfer indices',choices=['group_means','cohort_means'])
        for key,label in [('adjust_exposure','Adjust for target exposed surfaces'),('adjust_operating_hours','Adjust for ISTAT heating/cooling hours'),('adjust_internal_gains','Adjust for ISTAT household and appliance gains'),('adjust_solar','Include the solar weather feature')]:
            self.field(paths,('hybrid',key),label,kind=bool)
        self.field(paths,('report','powerpoint'),'Generate a PowerPoint report for this run',kind=bool)
        ttk.Label(paths,text='Independent mode: EP gets non-ISTAT schedules. The known-summary test gets only aggregates.\nThe optional ISTAT test gets demographics and survey-predicted behaviour, never target schedules.\nBuilt-in cases are synthetic stress tests, not measured-building validation.',wraplength=950).pack(anchor='w',pady=10)
        ttk.Label(paths,text='Hybrid: one reference EP weather signature, adjusted using target climate, exposure and ISTAT.\nThe ISTAT-ratio baseline is also reported. Advanced JSON exposes fitting bounds and component switches.\nISTAT only: reported hours and appliance kWh; thermal kWh remain blank.\nEnergyPlus is installed separately, not bundled in this package.',wraplength=950).pack(anchor='w',pady=15)
        self.field(analysis,('analysis','archetype_mode'),'Household archetypes',choices=['existing','refit'])
        self.field(analysis,('analysis','n_archetypes'),'Number of household archetypes (choose refit to change)',kind=int)
        self.checks(analysis,('analysis','features'),'Numeric inputs for refitting', ['agemin','agemax','nmembers','education','area_proxy','age_proxy'])
        self.field(analysis,('analysis','pca_components'),'PCA components (0 = all standardized inputs)',kind=int)
        self.field(analysis,('analysis','kmeans_starts'),'K-means starts',kind=int)
        self.field(analysis,('analysis','seed'),'Random seed',kind=int)
        self.checks(analysis,('analysis','group_factors'),'Factors defining energy-group cells', ['location','dwelling_type','archetype','cooling','insulation_improvement'])
        self.field(analysis,('analysis','electricity_include_cooling'),'Additionally retain cooling-system presence for electricity',kind=bool)
        self.field(analysis,('analysis','group_selection'),'Energy-group selection',choices=['fixed','elbow'])
        for key,label in [('heating_groups','Heating groups'),('electricity_groups','Electricity groups'),('elbow_max_k','Maximum groups for elbow'),('minimum_train_cell','Minimum training households per cell')]:self.field(analysis,('analysis',key),label,kind=int)
        ttk.Label(analysis,text='Refitted labels are new clusters, not automatically the published A1–A4.\nChanging grouping factors refits the energy groups. Missing selected factors are excluded.\nThe 2021 insulation field records improvements, not current insulation performance.',wraplength=950).pack(anchor='w',pady=12)
        for key,label in [('area_m2','Floor area (m²)'),('aspect_ratio','Length / width'),('height_m','Ceiling height (m)'),('north_axis_deg','North axis (degrees)'),('window_wall_ratio','Window / exposed-wall area'),('wall_u','Wall U (W/m²K)'),('roof_u','Roof U (W/m²K)'),('floor_u','Floor U (W/m²K)'),('window_u','Window U (W/m²K)'),('window_shgc','Window SHGC'),('infiltration_ach','Air changes per hour'),('ground_temperature_c','Constant ground temperature (°C)'),('heat_setpoint_c','Heating setpoint (°C)'),('heat_setback_c','Heating setback (°C)'),('cool_setpoint_c','Cooling setpoint (°C)'),('cool_setback_c','Cooling setback (°C)'),('people_sensible_w','Sensible heat per person (W)'),('appliance_heat_fraction','Appliance heat released into zone (fraction)')]:self.field(building,('simulation',key),label,kind=float)
        self.field(building,('simulation','cooling_enabled'),'Cooling enabled',kind=bool)
        self.field(building,('simulation','timestep'),'EnergyPlus timesteps per hour',kind=int)
        self.climate_widgets=[]
        for i,c in enumerate(self.cfg['climates']):
            group=ttk.LabelFrame(context,text=c['name'],padding=8);group.pack(fill='x',pady=5)
            enabled=tk.BooleanVar();file=tk.StringVar();self.climate_widgets.append((enabled,file))
            ttk.Checkbutton(group,text='Include',variable=enabled).pack(side='left')
            ttk.Entry(group,textvariable=file,width=85).pack(side='left',fill='x',expand=True,padx=8)
            ttk.Button(group,text='EPW…',command=lambda v=file:self.browse(v,'file')).pack(side='right')
        self.context_widgets=[]
        for c in self.cfg['contexts']:
            group=ttk.LabelFrame(context,text=c['name'],padding=8);group.pack(fill='x',pady=5)
            enabled=tk.BooleanVar();walls={x:tk.BooleanVar() for x in ['North','East','South','West']};roof=tk.StringVar();floor=tk.StringVar()
            self.context_widgets.append((enabled,walls,roof,floor))
            ttk.Checkbutton(group,text='Include',variable=enabled).pack(side='left')
            for x,v in walls.items():ttk.Checkbutton(group,text=x+' exposed',variable=v).pack(side='left')
            ttk.Label(group,text='Roof').pack(side='left');ttk.Combobox(group,textvariable=roof,values=['Adiabatic','Outdoors'],width=10,state='readonly').pack(side='left',padx=5)
            ttk.Label(group,text='Floor').pack(side='left');ttk.Combobox(group,textvariable=floor,values=['Adiabatic','Ground'],width=10,state='readonly').pack(side='left',padx=5)
        self.field(context,('simulation','anchor_climate'),'Reference climate',choices=[c['name'] for c in self.cfg['climates']])
        self.field(context,('simulation','anchor_context'),'Reference dwelling',choices=['Apartment','Multi-family','Single-family'])
        self.field(context,('simulation','anchor_archetype'),'Reference archetype',kind=int)
        ttk.Label(context,text='Exposure presets are explicit experimental assumptions, not classifications inferred from ISTAT.\nEach EPW maps to the macro-region specified in Advanced JSON. Weather files come from the EnergyPlus database.',wraplength=950).pack(anchor='w',pady=12)
        bar=ttk.Frame(advanced);bar.pack(fill='x')
        ttk.Button(bar,text='Refresh JSON from controls',command=self.refresh_json).pack(side='left')
        ttk.Button(bar,text='Apply JSON to controls',command=self.apply_json).pack(side='left',padx=8)
        ttk.Label(advanced,text='Edit all parameters here, including 24-hour profiles and per-archetype overrides. Apply JSON before running.').pack(anchor='w',pady=8)
        self.json_text=tk.Text(advanced,wrap='none',font=('Consolas',10),height=24);self.json_text.pack(fill='both',expand=True)
        bottom=ttk.Frame(root,padding=(20,10));bottom.pack(fill='x')
        self.run_button=ttk.Button(bottom,text='▶  Run experiment',command=self.start);self.run_button.pack(side='left')
        self.cancel_button=ttk.Button(bottom,text='Cancel',command=self.cancel.set,state='disabled');self.cancel_button.pack(side='left',padx=8)
        ttk.Button(bottom,text='Open last results',command=self.open_results).pack(side='left')
        self.progress=ttk.Progressbar(bottom,mode='indeterminate');self.progress.pack(side='right',fill='x',expand=True,padx=(20,0))
        self.logbox=tk.Text(root,height=7,font=('Consolas',9),state='disabled',background='#f2f6f8');self.logbox.pack(fill='x',padx=20,pady=(0,15))
        self.populate();root.after(150,self.poll);root.protocol('WM_DELETE_WINDOW',self.close)

    def tab(self,title):
        shell=ttk.Frame(self.tabs);self.tabs.add(shell,text=title)
        canvas=tk.Canvas(shell,highlightthickness=0);scroll=ttk.Scrollbar(shell,orient='vertical',command=canvas.yview);canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right',fill='y');canvas.pack(side='left',fill='both',expand=True)
        inner=ttk.Frame(canvas,padding=15);window=canvas.create_window((0,0),window=inner,anchor='nw')
        inner.bind('<Configure>',lambda e:canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>',lambda e:canvas.itemconfigure(window,width=e.width))
        return inner

    def field(self,parent,path,label,kind=str,choices=None,browse=None):
        line=ttk.Frame(parent);line.pack(fill='x',pady=4)
        ttk.Label(line,text=label,width=61).pack(side='left')
        var=tk.BooleanVar() if kind is bool else tk.StringVar();self.fields[path]=(var,kind)
        widget=ttk.Checkbutton(line,variable=var) if kind is bool else ttk.Combobox(line,textvariable=var,values=choices,state='readonly',width=32) if choices else ttk.Entry(line,textvariable=var,width=36)
        widget.pack(side='left',fill='x',expand=True)
        if browse:ttk.Button(line,text='Browse…',command=lambda:self.browse(var,browse)).pack(side='right',padx=5)

    def checks(self,parent,path,label,choices):
        frame=ttk.LabelFrame(parent,text=label,padding=8);frame.pack(fill='x',pady=8)
        variables={key:tk.BooleanVar() for key in choices};self.fields[path]=(variables,list)
        for i,(key,var) in enumerate(variables.items()):ttk.Checkbutton(frame,text=key,variable=var).grid(row=i//3,column=i%3,sticky='w',padx=12,pady=3)

    def browse(self,var,kind):
        value=filedialog.askdirectory() if kind=='directory' else filedialog.askopenfilename()
        if value:var.set(value)

    def populate(self):
        for (section,key),(var,kind) in self.fields.items():
            value=self.cfg[section][key]
            if kind is list:
                for k,v in var.items():v.set(k in value)
            else:var.set(value)
        for c,(enabled,file) in zip(self.cfg['climates'],self.climate_widgets):enabled.set(c['enabled']);file.set(c['epw'])
        for c,(enabled,walls,roof,floor) in zip(self.cfg['contexts'],self.context_widgets):
            enabled.set(c['enabled']);roof.set(c['roof']);floor.set(c['floor'])
            for key,var in walls.items():var.set(key in c['exposed_walls'])
        self.json_text.delete('1.0','end');self.json_text.insert('1.0',json.dumps(self.cfg,indent=2))
        self.update_summary()

    def collect(self):
        cfg=copy.deepcopy(self.cfg)
        for (section,key),(var,kind) in self.fields.items():
            try:cfg[section][key]=[k for k,v in var.items() if v.get()] if kind is list else kind(var.get())
            except (ValueError,tk.TclError):raise ValueError(f'Invalid value for {key}.')
        for c,(enabled,file) in zip(cfg['climates'],self.climate_widgets):c.update(enabled=enabled.get(),epw=file.get())
        for c,(enabled,walls,roof,floor) in zip(cfg['contexts'],self.context_widgets):c.update(enabled=enabled.get(),exposed_walls=[k for k,v in walls.items() if v.get()],roof=roof.get(),floor=floor.get())
        return cfg

    def update_summary(self):
        c=self.cfg;n=sum(x['enabled'] for x in c['climates'])*sum(x['enabled'] for x in c['contexts'])*c['analysis']['n_archetypes']
        if c.get('benchmark',{}).get('mode')=='independent_schedules':
            self.summary.config(text=f'Independent household schedules  ·  {c["simulation"]["area_m2"]:g} m²  ·  synthetic defaults or your profile file  ·  Previous results preserved')
            return
        self.summary.config(text=f'{n} scenarios  ·  {c["simulation"]["area_m2"]:g} m²  ·  H={c["analysis"]["heating_groups"]} / E={c["analysis"]["electricity_groups"]}  ·  No prior results are overwritten')

    def check(self):
        try:self.cfg=validate(self.collect());self.update_summary();messagebox.showinfo('Configuration','Configuration is valid. EPW files and input columns are checked when running.')
        except Exception as e:messagebox.showerror('Configuration',str(e))

    def refresh_json(self):
        try:self.cfg=self.collect();self.populate()
        except Exception as e:messagebox.showerror('Configuration',str(e))

    def apply_json(self):
        try:
            cfg=json.loads(self.json_text.get('1.0','end'))
            if len(cfg['climates'])!=len(self.climate_widgets) or len(cfg['contexts'])!=len(self.context_widgets):raise ValueError('This UI supports three climate slots and three context slots. Rename/replace them, or use CLI for a larger matrix.')
            self.cfg=validate(cfg);self.populate()
        except Exception as e:messagebox.showerror('JSON configuration',str(e))

    def load_config(self):
        path=filedialog.askopenfilename(filetypes=[('Configuration','*.json')])
        if path:
            try:
                cfg=load(path)
                if len(cfg['climates'])!=3 or len(cfg['contexts'])!=3:raise ValueError('Use three climate/context slots in the GUI; CLI supports other counts.')
                self.cfg=cfg;self.populate()
            except Exception as e:messagebox.showerror('Load configuration',str(e))

    def save_config(self):
        try:
            cfg=validate(self.collect());path=filedialog.asksaveasfilename(defaultextension='.json',filetypes=[('Configuration','*.json')])
            if path:save(cfg,path);self.cfg=cfg;self.update_summary()
        except Exception as e:messagebox.showerror('Save configuration',str(e))

    def reset(self):
        if messagebox.askyesno('Reset','Replace the current settings with defaults?'):self.cfg=default_config();self.populate()

    def start(self):
        if self.worker and self.worker.is_alive():return
        try:self.cfg=validate(self.collect())
        except Exception as e:messagebox.showerror('Cannot run',str(e));return
        self.update_summary();cfg=copy.deepcopy(self.cfg);self.cancel.clear();self.run_button.config(state='disabled');self.cancel_button.config(state='normal');self.progress.start()
        def work():
            try:
                from .engine import run
                out=run(cfg,lambda m:self.messages.put(('log',m)),self.cancel);self.messages.put(('done',out))
            except Exception as e:self.messages.put(('error',str(e)))
        self.worker=threading.Thread(target=work,daemon=True);self.worker.start()

    def poll(self):
        while not self.messages.empty():
            kind,value=self.messages.get()
            self.logbox.config(state='normal');self.logbox.insert('end',str(value)+'\n');self.logbox.see('end');self.logbox.config(state='disabled')
            if kind in ['done','error']:
                self.progress.stop();self.run_button.config(state='normal');self.cancel_button.config(state='disabled')
                if kind=='done':self.last_output=Path(value);messagebox.showinfo('Complete','Results saved. Click Open last results to view the report.')
                else:messagebox.showerror('Run stopped',value)
        self.root.after(150,self.poll)

    def open_results(self):
        if self.last_output and (self.last_output/'report.html').exists():os.startfile(str(self.last_output/'report.html'))
        else:messagebox.showinfo('Results','Run an experiment first.')

    def close(self):
        if self.worker and self.worker.is_alive():
            if messagebox.askyesno('Cancel run','Cancel the active run and close?'):
                self.cancel.set();self.root.after(300,self.close_after_cancel)
        else:self.root.destroy()

    def close_after_cancel(self):
        if self.worker.is_alive():self.root.after(300,self.close_after_cancel)
        else:self.root.destroy()


def launch(cfg,self_test=False):
    root=tk.Tk()
    if self_test:root.withdraw()
    app=Studio(root,cfg)
    if self_test:
        root.update();validate(app.collect());assert len(app.tabs.tabs())==5
        app.fields[('simulation','area_m2')][0].set('120')
        assert app.collect()['simulation']['area_m2']==120
        app.fields[('analysis','electricity_groups')][0].set('3')
        assert app.collect()['analysis']['electricity_groups']==3
        root.destroy()
        return
    root.mainloop()
