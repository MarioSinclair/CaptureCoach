# Capture Coach

Checks intake photos of electronic equipment for blur, lighting, framing and
label obstruction, tells you exactly what to re-shoot, and flags which required
views are missing. Built for the American Circular track at Bay Hacks.

31/34 on the challenge practice set, with zero false alarms and zero misses.
About a cent a photo. Thresholds were cut from the measured gaps in the
practice data, and each one records that gap in `coach/config.py`.

```bash
pip install -r requirements.txt
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env   # optional; measured checks work without it
uvicorn app:app --port 8050
```

The challenge photographs and CSVs aren't in this repo. Drop `images/` and the
`*.csv` files from the challenge package into the project root to enable the
practice-set buttons; uploading your own photos works without them.
