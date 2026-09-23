"""Translate historical Motion commands into installed PyMicroglia actions."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys


def read(path):
    from pymicroglia._results import document, read_document
    path = Path(path)
    if not document(path).is_file() and path.parent.name == "analysis" and path.name.endswith(".example.json"):
        from importlib.resources import files
        example = files("pymicroglia").joinpath("data", path.name)
        if example.is_file():
            return json.loads(example.read_text(encoding="utf-8"))
    return read_document(path)


def invoke(action, params, claim="", dry_run=False):
    from pymicroglia.run import run_recorded
    encoded = [name + "=" + repr(value)
               for name, value in params.items()]
    # Single quotes preserve Python literals in PowerShell, including embedded spaces.
    quote = lambda value: "'" + str(value).replace("'", "''") + "'"
    command = "pymicroglia run " + action + " " + " ".join(map(quote, encoded))
    if claim:
        command += " --claim " + quote(claim)
    print("equivalent: " + command, flush=True)
    if dry_run:
        return {"ok": True, "action": action, "params": params}
    result = run_recorded(action, claim=claim, entry="cli", **params)
    print(json.dumps(result, default=str, indent=2))
    outcome = result["result"]
    if getattr(outcome, "successful", True) is False:
        failures = {key: value.outcome.reason for key, value in outcome.results.items()
                    if value.outcome.status not in {"completed", "reused", "skipped-empty"}}
        raise RuntimeError("Workflow failed: " + json.dumps(failures))
    return result


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    for name in ("modules", "figures", "doctor", "conditions", "theme"):
        p = sub.add_parser(name)
        p.add_argument("--config")
        if name == "figures": p.add_argument("--slug")
        if name == "conditions":
            p.add_argument("--try", dest="try_stem", action="append")
            p.set_defaults(config=None)
        if name == "theme":
            p.add_argument("--list", action="store_true")
            p.add_argument("--preset")
            p.add_argument("--check", action="store_true")
    p = sub.add_parser("run")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stem", action="append")
    p.add_argument("--module", action="append")
    p.add_argument("--out-is-name", action="store_true")
    p.add_argument("--print-summary", action="store_true")
    p.add_argument("--claim", default="Measure the declared tracked recordings")
    for name in ("pool", "window", "contrasts", "states", "cluster", "videos", "plots", "pipeline"):
        p = sub.add_parser(name)
        p.add_argument("run")
        p.add_argument("--claim", default="Run the declared " + name + " analysis")
        if name in ("window", "contrasts"):
            p.add_argument("--config", required=True)
        if name in ("states", "cluster"):
            p.add_argument("--out", required=True)
            p.add_argument("--options")
            if name == "states": p.add_argument("--model")
        if name == "pipeline":
            declaration = p.add_mutually_exclusive_group(required=True)
            declaration.add_argument("--request")
            declaration.add_argument("--config")
            p.add_argument("--out")
            p.add_argument("--name", action="append")
            p.add_argument("--step", action="append")
            p.add_argument("--presentation")
        if name == "plots":
            p.add_argument("--plan")
            p.add_argument("--only", action="append", default=[])
            p.add_argument("--dry-run", action="store_true")
            p.add_argument("--draft", action="store_true")
        if name == "videos":
            p.add_argument("--stem")
            p.add_argument("--identity", action="append", type=int)
            p.add_argument("--cells", type=int, default=6)
            p.add_argument("--events")
            p.add_argument("--span", choices=("recording", "lifespan", "event"))
            p.add_argument("--event-hours", type=float)
            p.add_argument("--fps", type=int)
            p.add_argument("--crop-px", type=int)
            p.add_argument("--lut", dest="cell_lut")
            p.add_argument("--no-locator", action="store_true")
            p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("pipeline-export")
    p.add_argument("report")
    p.add_argument("--choices", required=True)
    p.add_argument("--scope", choices=("measurement", "dataset"), default="measurement")
    p.add_argument("--out")
    p.add_argument("--override-reason")
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    command = args.command
    if command in {"doctor", "modules", "figures"}:
        from pymicroglia.cli import main as cli
        target = ["describe", args.slug.replace('-', '_')] if command == "figures" and args.slug else ["doctor" if command == "doctor" else "discover"]
        print("equivalent: pymicroglia " + " ".join(target), flush=True)
        return cli(target)
    if command in {"conditions", "theme"}:
        from pymicroglia.measure import load_config
        if command == "conditions":
            if not args.config: raise ValueError("conditions requires --config")
            config = load_config(args.config)
            rows = list(config.assignments())
            rows.extend(config.conditions.assign(stem) for stem in (args.try_stem or []))
            print(json.dumps([row.as_dict() for row in rows], indent=2))
            return 1 if any(not row.ok for row in rows) else 0
        else:
            from analysis_kit.style import themes
            print("Figure themes are supplied by analysis-kit; pass theme to a PyMicroglia figure action.")
            block = read(args.config).get("theme", {}) if args.config else {}
            name = args.preset or (block.get("preset") if isinstance(block, dict) else block)
            if args.list: print(json.dumps(themes.theme_names()))
            print(json.dumps(themes.theme(name, strict=True), indent=2))
            if args.check: print("Theme supplied directly by analysis-kit; no local copy.")
        return 0
    if command == "run":
        from pymicroglia.measure import load_config
        config = load_config(args.config)
        movies = [movie.as_dict() for movie in config.movies if not args.stem or movie.stem in args.stem]
        if not movies: raise ValueError("No configured movies match --stem")
        output = Path(args.out)
        if args.out_is_name and not output.is_absolute():
            output = config.output_root / output.name
        output = output.resolve()
        params = dict(analysis_config=str(Path(args.config).resolve()),
                      output_dir=str(output.parent),run_label=output.name,if_exists='error')
        if args.stem: params['movies'] = movies
        if args.module: params['enabled_modules'] = args.module
        invoke('measure', params, args.claim)
        return 0
    if command in {"pool", "window", "contrasts"}:
        params = {'run_dir': args.run}
        if command != 'pool':
            config = read(args.config)
            params[{'window':'windows', 'contrasts':'contrasts'}[command]] = config.get({'window':'windows', 'contrasts':'contrasts'}[command], [])
            if command == 'contrasts': params['metric_groups'] = config.get('metric_groups', {})
        invoke(command, params, args.claim)
    elif command in {"states", "cluster"}:
        output = Path(args.out).resolve()
        params = dict(run=args.run, output_dir=str(output.parent), run_label=output.name, if_exists='error')
        params['state_options' if command == 'states' else 'clustering_options'] = read(args.options) if args.options else {}
        if command == 'states' and args.model: params['replay'] = args.model
        invoke(command, params, args.claim)
    elif command == 'videos':
        params = dict(run=args.run, identities=args.identity or [], cell_count=args.cells,
                      events=[item.strip() for item in (args.events or '').split(',') if item.strip()],
                      locator=not args.no_locator, dry_run=args.dry_run)
        params.update({key:getattr(args,key) for key in ('stem','span','event_hours','fps','crop_px','cell_lut') if getattr(args,key) is not None})
        invoke('follow', params, args.claim)
    elif command == 'pipeline':
        from pymicroglia.pipelines._requests import FAMILIES
        document = read(args.request or args.config)
        requests = document.get('pipelines', [document]) if isinstance(document,dict) else document
        actions = {value[0]:key for key,value in FAMILIES.items()}
        selected = [item for item in requests if not args.name or item.get('name') in args.name]
        if not selected: raise ValueError('No pipeline requests match --name')
        for request in selected:
            params = dict(run=args.run, pipeline_request=request, if_exists='skip')
            if args.out: params['output_dir'] = args.out
            if args.step: params['only'] = args.step
            if args.presentation: params['presentation'] = read(args.presentation)
            invoke(actions[request['pipeline']], params, args.claim)
    elif command == 'pipeline-export':
        from pymicroglia.pipelines.audit.profiles import export_profile
        choices = (args.choices if args.scope == 'dataset' else
                   read(args.choices) if Path(args.choices).is_file() else json.loads(args.choices))
        print('equivalent Python: pymicroglia.pipelines.audit.profiles.export_profile')
        result = export_profile(args.report, choices, scope=args.scope, output=args.out, override_reason=args.override_reason)
        print(result)
    elif command == 'plots':
        manifest = read(Path(args.run)/'manifest.json')
        document = read(args.plan) if args.plan else manifest.get('settings', {}).get('plots', [])
        entries = document.get('plots', []) if isinstance(document,dict) else document
        # Translate historical option spellings before plan validation.
        translated = []
        for entry in entries:
            entry = dict(entry)
            for block in ('options', 'for_each'):
                values = dict(entry.get(block, {}))
                for old, new in {'rings':'ring_count','normalise':'transition_normalisation'}.items():
                    if old in values:
                        if new in values: raise ValueError(f"Declare only one of {old} and {new}")
                        values[new] = values.pop(old)
                        if 'as' in entry: entry['as'] = entry['as'].replace('{'+old+'}', '{'+new+'}')
                if block in entry: entry[block] = values
            translated.append(entry)
        entries = translated
        if entries and not all('name' in item for item in entries):
            from pymicroglia.figure_tables.plans import expand, parse
            from pymicroglia.measure.spec import parse_metric_groups
            from pymicroglia.visualisation.figures import load
            groups = document.get('metric_groups', {}) if isinstance(document, dict) else manifest.get('settings', {}).get('metric_groups', {})
            specs = {spec.slug: spec for spec in load().values()}
            entries = [item.as_dict() for item in expand(parse(entries, parse_metric_groups(groups)), specs)]
        if not entries:
            from pymicroglia._results import figure_plans
            entries = list(figure_plans(args.run, manifest=manifest).values())
        if not entries: raise ValueError('No saved plot plan; pass --plan')
        selected = [item for item in entries if not args.only or item.get('name') in args.only or item['figure'] in args.only]
        if not selected: raise ValueError('No plot items match --only')
        for item in selected:
            key = item['figure'].replace('-', '_')
            options = dict(item.get('options', {}))
            for old, new in {'rings':'ring_count','normalise':'transition_normalisation'}.items():
                if old in options:
                    if new in options: raise ValueError(f"Declare only one of {old} and {new}")
                    options[new] = options.pop(old)
            params = dict(run=args.run, **options)
            for name in ('stem','text','view'):
                if name in item: params[name] = item[name]
            if item.get('pipeline'): params['item'] = item['name']
            # Every expanded request has its own destination, including two
            # views of the same recording with different declared settings.
            base = Path(args.run).parent if args.draft else Path(args.run)
            name = item.get('name', key).replace('/', '_').replace('\\', '_')
            if name in {'.', '..'} or Path(name).name != name:
                raise ValueError('Plot item name must identify a folder, not an absolute path')
            params['output_dir'] = str(base / 'figures' / name)
            if args.draft: params['overwrite'] = True
            invoke(key, params, args.claim, dry_run=args.dry_run)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, RuntimeError, FileNotFoundError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
