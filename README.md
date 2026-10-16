# Checkpoint download statistics

The six public MaLoW memory modules use Hugging Face's native download counter.
Each release includes a descriptive `config.json`, which is fetched by the normal
`python -m malow.download` snapshot download. It describes the memory module;
the evaluation runtime continues to read `c2l_metadata.json`. The weights, base
model revisions, evaluation settings, and download commands are unchanged.

The [daily workflow](https://github.com/dragonlzm/MaLoW/actions/workflows/download-stats.yml)
runs at **09:17 UTC** and saves public aggregates on the
[`download-stats` branch](https://github.com/dragonlzm/MaLoW/tree/download-stats).
That branch contains `daily/YYYY-MM-DD.json`, with the first complete successful
snapshot for each UTC day. The script reads model IDs from the release registry
and requests only Hub metadata APIs; it never downloads model/config files and
does not increase the counters itself. No Hugging Face token is required.

Each file has `schema_version: 1`, a UTC collection `date`, and a `models` list.
Each model record contains:

| Field | Meaning |
| --- | --- |
| `repo_id` | Hugging Face checkpoint repository |
| `observed_at` | UTC timestamp when the aggregate was observed |
| `downloads_30d` | Downloads counted over the preceding 30 days |
| `downloads_all_time` | Hugging Face's cumulative count |

These are observations of aggregate counters, not individual download events.
Hugging Face counts requests to designated files, including `HEAD` requests, so
counts do not represent unique people or verified complete downloads. Manual
weights-only downloads that skip `config.json` may be missed. Old pinned releases
without the config still work but do not gain the new counting file retroactively.
Previously uncounted downloads cannot be reconstructed from these snapshots.
See [Hugging Face's counting rules](https://huggingface.co/docs/hub/models-download-stats).

The rolling 30-day value may decrease. Preserve raw all-time values too: a decrease
should be treated as a counter correction, not negative downloads. Differences
between all-time observations are estimates over the observation interval, not
guaranteed daily download totals. Initial validation downloads may be counted.

## Run or recover a snapshot

Use **Run workflow** on the workflow page, or run locally from the source checkout:

```bash
PYTHONPATH=src python -m malow.download_stats --output-dir /path/to/history/daily
```

The command needs only Python 3.10+. A complete existing snapshot is preserved
without making requests. Missing/invalid counters or persistent API errors fail
the run without creating a partial daily file or substituting zero. Transient
network failures, HTTP 429, and server errors are retried up to three attempts.

The workflow must be on the default branch, with Actions enabled and its built-in
token allowed to write repository contents. No model packages or credentials are
installed in this job. Failures are visible in GitHub Actions; a rerun on the same
UTC day can recover a missed snapshot. A missed past day cannot be backfilled
with that day's actual counts from the aggregate API.

GitHub can delay schedules and can disable scheduled workflows after 60 days of
repository inactivity. If disabled, re-enable the workflow in Actions and run it
manually. See [GitHub's schedule documentation](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/disable-and-enable-workflows).
