"""Runs the plugin's simulated-host test (plugin/tests/test_host.cpp) and keeps the residuals of the
plugin's output against the core's filter chain, with the recommended filter off and on.

    python plugin_host.py path/to/hpf_host_test      -> results/plugin_host_chain.json"""
import json
import re
import subprocess
import sys

from _paths import RESULTS

out = subprocess.run([sys.argv[1]], capture_output=True, text=True, timeout=600)
text = out.stdout
assert out.returncode == 0 and text.rstrip().endswith("Passed."), text[-2000:]
res = {m.group(1): float(m.group(2))
       for m in re.finditer(r"Chain with the recommended filter (on|off): residual (-?[\d.]+) dB", text)}
assert set(res) == {"on", "off"}, text
(RESULTS / "plugin_host_chain.json").write_text(json.dumps({"residual_db": res}, indent=1))
print(res)
