"""Sequential actual-data research interface, with a raw-free numeric audit."""
import argparse,subprocess,sys
from .acquire import ROOT


def main():
    ap=argparse.ArgumentParser();ap.add_argument("command",choices=["baseline","mapping","smoke-test","fetch","build-pit","analyze","audit","report","test","verify-public"])
    ap.add_argument("--stage",default="all",choices=["smoke","50","150","all"])
    args=ap.parse_args()
    if args.command=="baseline":
        from .acquire import baseline;baseline()
    elif args.command=="mapping":
        from .acquire import mapping;mapping()
    elif args.command=="smoke-test":
        from .acquire import fetch;fetch("smoke")
    elif args.command=="fetch":
        from .acquire import fetch;fetch(args.stage)
    elif args.command=="build-pit":
        from .build_governance_panel import build;build()
    elif args.command=="analyze":
        from .run_ff3_alpha import analyze;analyze()
        from .supplementary_analysis import analyze;analyze()
        from .supplementary_robustness import run;run()
    elif args.command=="audit":
        from .bias_overfit_audit import audit;audit()
        from .audit_return_data import audit;audit()
        from .public_audit import audit;audit()
    elif args.command=="report":
        from .make_report import make_report;make_report()
    elif args.command=="test":
        subprocess.run([sys.executable,"-m","pytest","-q",str(ROOT/"tests")],check=True)
    elif args.command=="verify-public":
        from .public_audit import audit;audit()

if __name__=="__main__":main()
