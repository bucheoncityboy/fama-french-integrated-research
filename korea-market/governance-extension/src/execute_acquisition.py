"""Fixed acquisition-only stages; no arbitrary subprocess command interface."""
import getpass
import json
import os
from .acquire import fetch,ROOT
from .build_governance_panel import build


def main():
    if not (os.environ.get("DART_API_KEY") or os.environ.get("OPENDART_API_KEY")):
        os.environ["DART_API_KEY"]=getpass.getpass("OpenDART credential (hidden; not saved): ")
    for stage in ("50","150","all"):
        fetch(stage)
        build()
        status=json.loads((ROOT/"output/pit_status.json").read_text())
        if status["future_leaks"] or not status["G_PIT_observations"]:
            raise RuntimeError("PIT_gate_failed")
        (ROOT/"output"/("checkpoint_"+stage+".json")).write_text(json.dumps(status,indent=2))
        if stage=="150":
            import shutil
            shutil.copyfile(ROOT/"data/private/joint_panel.parquet",ROOT/"data/private/stage150_joint_panel.parquet")
        print("GATE_B_CHECKPOINT="+stage,flush=True)

if __name__=="__main__":main()
