# AGENTS.md

## Command Log

- `git init` – repository initialized (already done).
- `git remote add origin https://github.com/E1merM0ren0/Vulcan-AI-test.git` – set remote.
- `git add .gitignore vulcan/train_out.txt` – stage initial files.
- `git commit -am "Add initial state"` – first commit (hash b44c2a3).
- `git push -u origin main` – push to remote (now on GitHub).

## Dataset attempts (no URLs found)

Attempted to clone the following datasets but the repository names did not resolve on GitHub:

```
https://github.com/WithinUsAI/claude_mythos_distilled_25k
https://github.com/aisamdasu/algocean-fable5-traces
https://github.com/TRACCERR/Sumtables-Cuniform-Small-Fable5-Remaster-v2
```

Result: `Repository not found` for each. Awaiting correct URLs or alternative sources.

## Next steps (pending)

1. Provide correct dataset repository URLs (or direct download links).
2. Confirm where to store subset data (e.g., `data/` directory).
3. Decide on push strategy for new branch or main.

Once clarified, I will:
- Shallow‑clone each dataset.
- Extract the first ~1 000 rows of the relevant files (`head -n 1000`).
- Run `vulcan_bitnet_gpt` on those subsets.
- Log the exact commands in this file.
- Commit and push the results.
