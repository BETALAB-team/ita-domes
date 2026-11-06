"""GUI by default. CLI: --config path.json --run; --write-default path.json."""
import argparse
import sys
from pathlib import Path
from energy_studio.config import default_config,load,save


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config');parser.add_argument('--run',action='store_true')
    parser.add_argument('--write-default');parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--write-independent-config',metavar='JSON')
    parser.add_argument('--write-benchmark-profiles',metavar='JSON')
    parser.add_argument('--refresh-report',metavar='RUN_DIRECTORY',help='Regenerate plots/reports from a completed run without simulations')
    parser.add_argument('--with-powerpoint',action='store_true',help='Enable PowerPoint when refreshing a completed run')
    args=parser.parse_args()
    if args.write_benchmark_profiles:
        import json
        from energy_studio.independent_inputs import default_profiles
        Path(args.write_benchmark_profiles).write_text(json.dumps(default_profiles(),indent=2),encoding='utf-8');return
    if args.write_independent_config:
        cfg=default_config();cfg['benchmark']['mode']='independent_schedules';cfg['simulation']['run_full_benchmark']=True
        save(cfg,args.write_independent_config);return
    if args.refresh_report:
        from energy_studio.engine import refresh_reports
        refresh_reports(args.refresh_report,powerpoint=True if args.with_powerpoint else None)
        return
    if args.write_default:save(default_config(),args.write_default);return
    packaged_default=Path(sys.executable).with_name('default_config.json')
    cfg=load(args.config) if args.config else load(packaged_default) if getattr(sys,'frozen',False) and packaged_default.is_file() else default_config()
    if args.run:
        from energy_studio.engine import run
        out=run(cfg)
        if sys.stdout:print(out)
    else:
        from energy_studio.gui import launch
        launch(cfg,self_test=args.self_test)


if __name__=='__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    try:main()
    except Exception:
        import traceback
        if sys.stderr:traceback.print_exc()
        elif '--run' in sys.argv or '--self-test' in sys.argv or '--refresh-report' in sys.argv:
            Path(sys.executable).with_name('energy_studio_error.log').write_text(traceback.format_exc(),encoding='utf-8')
        else:
            import tkinter.messagebox
            tkinter.messagebox.showerror('Energy Studio',traceback.format_exc())
        raise SystemExit(1)
