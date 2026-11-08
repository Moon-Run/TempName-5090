"""Read-only validation and paired statistics for standalone RTX5090 experiments."""
import math
from pathlib import Path
import random
import statistics

import protocol as p


def pair_statistics(candidate, reference, balanced, bootstrap=10000):
    p.require(len(candidate) == len(reference) and candidate, 'Unpaired samples')
    logs = [math.log(x / y) for x, y in zip(candidate, reference)]
    estimate = 100 * (1 - math.exp(statistics.mean(logs)))
    interval = None
    if balanced and len(logs) >= 8:
        rng = random.Random(20261108)
        samples = sorted(100 * (1 - math.exp(statistics.mean(rng.choices(logs, k=len(logs)))))
                         for _ in range(bootstrap))
        interval = [samples[int((bootstrap - 1) * .025)], samples[int((bootstrap - 1) * .975)]]
    return dict(percent=estimate, ci95=interval, paired_blocks=len(logs),
                faster_blocks=sum(x < y for x, y in zip(candidate, reference)))


def summary(samples):
    return dict(mean=statistics.mean(samples), median=statistics.median(samples),
                min=min(samples), max=max(samples), windows=len(samples))


def summarize_stage(samples, stage, trials, blocks):
    result = {}
    roles = ('attn', 'mlp') if stage == 'operator' else ('full_step',)
    for trial in range(1, trials + 1):
        by_role = {}
        for role in roles:
            values = {arm: [next(s['value'] for s in samples if s['trial'] == trial
                                and s['block'] == block and s['arm'] == arm and s['role'] == role)
                           for block in range(blocks)] for arm in p.ARMS}
            pairs = {}
            for x, y in (('flux_taco', 'flux'), ('flux_taco_interleaved', 'flux_taco'),
                         (p.ARRIVAL, 'flux_taco_interleaved'),
                         ('flux_taco_interleaved', 'flux'), (p.ARRIVAL, 'flux')):
                pairs[f'{x}/{y}'] = pair_statistics(values[x], values[y], p.balanced(blocks))
            arm_summaries = {a: summary(v) for a, v in values.items()}
            for arm in p.ARMS:
                selected = [s for s in samples if s['trial'] == trial and s['arm'] == arm and s['role'] == role]
                if stage == 'train':
                    arm_summaries[arm].update(tokens_per_second=selected[0]['tokens_per_step'] * 1000 /
                        arm_summaries[arm]['mean'],
                        mean_training_loss=statistics.mean(s['mean_loss'] for s in selected),
                        mean_final_loss=statistics.mean(s['final_loss'] for s in selected),
                        mean_grad_norm=statistics.mean(s['mean_grad_norm'] for s in selected),
                        mean_final_grad_norm=statistics.mean(s['final_grad_norm'] for s in selected),
                        max_peak_allocated_gib=max(s['peak_allocated_bytes'] for s in selected) / 2**30,
                        max_peak_reserved_gib=max(s['peak_reserved_bytes'] for s in selected) / 2**30,
                        max_native_workspace_gib=max(s['native_workspace_bytes'] for s in selected) / 2**30,
                        max_allocated_plus_native_gib=max(s['allocated_plus_native_bytes'] for s in selected) / 2**30,
                        max_operator_scratch_gib=max(s['operator_scratch_bytes'] for s in selected) / 2**30)
                else:
                    arm_summaries[arm].update(
                        full_gemm_only_mean_us=statistics.mean(s['full_gemm_only_us'] for s in selected),
                        max_bf16_relative_l2=max(s['bf16_relative_l2'] for s in selected),
                        max_codec_reference_relative_l2=max(s['codec_reference_relative_l2'] for s in selected),
                        artificial_bypass_rank_windows=sum(len(s['artificial_bypass_ranks']) for s in selected),
                        artificial_diagnostic_rank_windows=sum(s['artificial_diagnostic_ranks'] for s in selected))
            by_role[role] = dict(unit='us/call' if stage == 'operator' else 'ms/step',
                                 arms=arm_summaries, pairs=pairs)
        result[f'trial{trial}'] = by_role
    return result


def validate_artifacts(root, directory, rows, config):
    """Validate actual native library and optional independent traces in-place.

    Report replay deliberately fails when native build artifacts were omitted
    from an archive; it does not silently claim to revalidate absent binaries.
    """
    import hashlib
    import json
    root, directory = Path(root), Path(directory)
    evidence = {}
    codec_source_sha = p.sha(root / 'sources/src/taco/codec.cu')
    source_manifest = p.read(root / 'source-manifest.json')
    checked_codec, checked_gemm = set(), set()
    for row in rows:
        build = row['codec_build']
        for label, flags in (('codec', build['identity'].get('flags', [])),
                             ('GEMM', row['native_build'].get('flags', []))):
            for option in ('--threads', '--split-compile'):
                p.require(flags.count(option) == 1 and flags.index(option) + 1 < len(flags)
                          and flags[flags.index(option) + 1] == str(config['build_jobs']),
                          f'{label} compile parallelism differs from frozen build_jobs')
        p.require(build['identity']['source_sha256'] == codec_source_sha,
                  'Native codec source differs from frozen source')
        key = hashlib.sha256(json.dumps(build['identity'], sort_keys=True).encode()).hexdigest()[:24]
        native = root / 'build/codec' / key
        manifest = native / 'manifest.json'
        library = native / 'libflux_5090_taco.so'
        if key not in checked_codec:
            p.require(p.read(manifest) == build, 'On-disk native codec build manifest changed')
            p.require(p.sha(library) == build['library_sha256'], 'On-disk native codec library bytes changed')
            checked_codec.add(key)
        evidence[str(manifest.relative_to(root))] = p.sha(manifest)
        evidence[str(library.relative_to(root))] = build['library_sha256']
        gemm = row['native_build']
        compiler = gemm.get('compiler', {})
        compiler_environment = compiler.get('environment')
        p.require(isinstance(compiler.get('path'), str) and compiler['path']
                  and isinstance(compiler.get('version'), str) and compiler['version']
                  and p.valid_hash(compiler.get('sha256')) and isinstance(compiler_environment, dict)
                  and set(compiler_environment) == {'CUDA_HOME', 'CUDA_PATH', 'NVCC_CCBIN',
                      'CPATH', 'CPLUS_INCLUDE_PATH', 'LIBRARY_PATH'}
                  and all(value is None or isinstance(value, str) for value in compiler_environment.values()),
                  'Missing compiler binary/environment provenance')
        gemm_identity = {name: gemm[name] for name in ('compiler', 'flags', 'inputs')}
        p.require(p.digest(gemm_identity) == gemm['cache_key'], 'Native GEMM cache identity mismatch')
        gemm_dir = root / 'build/gemm' / gemm['cache_key']
        gemm_manifest, gemm_library = gemm_dir / 'manifest.json', gemm_dir / 'libflux_5090_gemm.so'
        if gemm['cache_key'] not in checked_gemm:
            expected_inputs = {name: source_manifest[name] for name in
                               ('src/gemm/native.cu', 'python/flux_5090/native.py', 'python/flux_5090/build_options.py')}
            for upstream in ('flux', 'cutlass'):
                relative = ('third_party/cutlass.lock.json' if upstream == 'cutlass'
                            else f'third_party/{upstream}/manifest.json')
                upstream_manifest = p.read(root / 'sources' / relative)
                expected_inputs[relative] = source_manifest[relative]
                if upstream == 'cutlass' and 'third_party/cutlass/manifest.json' in source_manifest:
                    local = 'third_party/cutlass/manifest.json'
                    p.require(source_manifest[local] == source_manifest[relative],
                              'CUTLASS installed manifest differs from the external lock')
                    expected_inputs[local] = source_manifest[local]
                for file, expected in upstream_manifest['files'].items():
                    path = f'third_party/{upstream}/{file}'
                    p.require(source_manifest.get(path) == expected, 'Compiled upstream source was not frozen')
                    expected_inputs[path] = expected
                p.require(gemm['sources'][upstream] == dict(revision=upstream_manifest['revision'],
                    manifest_sha256=source_manifest[relative]), 'Native GEMM upstream provenance differs')
            p.require(gemm['inputs'] == expected_inputs, 'Native GEMM input set/hash differs from frozen sources')
            p.require(gemm.get('protocol') == 'flux-5090-native-v1' and gemm.get('backend') == p.BACKEND
                      and gemm.get('uses_flux_gemm_templates') is True
                      and gemm.get('original_flux_kernel_unchanged') is False
                      and gemm.get('original_cuda_flux_reduce_scatter') is False
                      and gemm.get('cta_level_swizzle') is True and gemm.get('producer_launches_per_call') == 1
                      and gemm.get('split_k') is False and gemm.get('binary_arch') == 'sm_120'
                      and '-arch=sm_120' in gemm['flags'], 'Native GEMM is not the declared Flux DP SM120 implementation')
            p.require(p.read(gemm_manifest) == gemm, 'On-disk native GEMM manifest changed')
            p.require(p.sha(gemm_library) == gemm['library_sha256'], 'On-disk native GEMM library bytes changed')
            checked_gemm.add(gemm['cache_key'])
        evidence[str(gemm_manifest.relative_to(root))] = p.sha(gemm_manifest)
        evidence[str(gemm_library.relative_to(root))] = gemm['library_sha256']
        if row['stage'] == 'operator' and config['profile']:
            for record in row['operator_results']:
                diagnostic = record['profile']
                name = diagnostic['trace']
                p.require(Path(name).name == name, 'Profile trace must be inside its worker directory')
                file = directory / name
                p.require(p.sha(file) == diagnostic['trace_sha256'], 'Profile trace bytes changed')
                evidence[str(file.relative_to(root))] = diagnostic['trace_sha256']
    return evidence


def validate_reference_sources(source_root):
    """Cross-check retained upstream reference files against their provenance."""
    source_root = Path(source_root)
    definitions = (
        ('flux', 'manifest.json', 'revision', '19831ca2d820e3e782ed1d15d8b52d0898b78b26'),
        ('cutlass', 'manifest.json', 'revision', 'df8a550d3917b0e97f416b2ed8c2d786f7f686a3'),
        ('taco', 'manifest.json', 'source_revision', 'ca717e05b6dd6d21fff4d2617d209c95be63f3ef'),
    )
    references = {}
    for name, filename, revision_key, revision in definitions:
        directory = source_root / 'third_party' / name
        manifest = (source_root / 'third_party/cutlass.lock.json' if name == 'cutlass'
                    else directory / filename)
        if name == 'cutlass':
            p.require(directory.is_dir() and manifest.is_file(),
                      'CUTLASS sources/lock missing; run bash env/setup_5090.sh in TempName-5090')
            local = directory / filename
            p.require(not local.exists() or (local.is_file() and local.read_bytes() == manifest.read_bytes()),
                      'CUTLASS installed manifest differs from third_party/cutlass.lock.json')
        record = p.read(manifest)
        p.require(record.get(revision_key) == revision, f'Unexpected {name} reference revision')
        files = record.get('files')
        p.require(isinstance(files, dict) and files, f'Missing {name} reference file hashes')
        for relative, detail in files.items():
            path = Path(relative)
            p.require(not path.is_absolute() and '..' not in path.parts,
                      'Reference paths must stay inside third_party')
            expected = detail.get('sha256') if isinstance(detail, dict) else detail
            p.require(p.valid_hash(expected) and (directory / path).is_file()
                      and p.sha(directory / path) == expected,
                      f'Upstream reference missing/changed: {name}/{relative}'
                      + ('; run bash env/setup_5090.sh after preserving local changes' if name == 'cutlass' else ''))
        references[name] = record
    return references


def execution_plan_summary(bundle, config):
    if bundle is None:
        return None
    model, world = config['model'], config['world_size']
    base = [item['physical'] for item in p.expected_schedule(model, world, 'flux_taco_interleaved')]
    count = len(base)
    roles = {}
    for role, plan in bundle['roles'].items():
        roles[role] = dict(arrival_identity=plan['arrival_order'] == base,
            arrival_changed_slots=sum(a != b for a, b in zip(plan['arrival_order'], base)),
            physical_blocks=count,
            role_plan_sha256=p.digest(plan), diagnostics=plan['diagnostics'])
    return dict(execution_plan_sha256=p.digest(bundle),
                transport_lanes=2, roles=roles,
                limitation='Arrival identity describes producer order only; D still uses per-lane-ready consumers.')


def validate_calibration_evidence(root, directory, rows, config):
    """Re-fit the frozen planner from rank-local measurements without a GPU."""
    import importlib.util
    source = Path(root) / 'sources/python/flux_5090/planning.py'
    spec = importlib.util.spec_from_file_location('rtx5090_frozen_planner_audit', source)
    planner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(planner)
    plan = p.calibration_from_rows(rows, config)
    evidence = {}
    all_samples = []
    for row in rows:
        record = row['calibration_results']
        p.require(record.get('plan_sha256') == p.digest(plan), 'Calibration digest mismatch')
        samples = record.get('raw_samples')
        p.require(isinstance(samples, list) and samples, 'Missing raw calibration measurements')
        name = record.get('raw_samples_file')
        p.require(isinstance(name, str) and Path(name).name == name, 'Invalid calibration sample path')
        file = Path(directory) / name
        p.require(p.read(file) == dict(schema='rtx5090-calibration-samples-v1', rank=row['rank'], samples=samples),
                  'Calibration sample file differs from rank evidence')
        evidence[str(file.relative_to(root))] = p.sha(file)
        all_samples.append(samples)
    p.require(all(len(samples) == len(all_samples[0]) for samples in all_samples), 'Calibration rank sample count differs')
    for items in zip(*all_samples):
        p.require(all({k: v for k, v in item.items() if k != 'local_elapsed_seconds'} ==
                      {k: v for k, v in items[0].items() if k != 'local_elapsed_seconds'} for item in items),
                  'Calibration labels/inputs/aggregate times differ across ranks')
        p.require(all(p.finite_number(item.get('local_elapsed_seconds')) and item['local_elapsed_seconds'] > 0
                      for item in items), 'Invalid rank-local calibration time')
        p.require(items[0].get('max_rank_elapsed_seconds') == max(item['local_elapsed_seconds'] for item in items),
                  'Calibration timing must use actual maximum rank elapsed')
        for item in items:
            p.require(item.get('correctness_passed') is True
                      and p.finite_number(item.get('bf16_relative_l2'), True) and item['bf16_relative_l2'] <= .05
                      and p.finite_number(item.get('codec_reference_relative_l2'), True)
                      and item['codec_reference_relative_l2'] <= .01,
                      'Calibration candidate failed numerical gates')
    model, cfg, world = config['model'], config['calibration'], config['world_size']
    samples = iter(all_samples[0])
    for role_index, role in enumerate(('attn', 'mlp')):
        def evaluate(candidate, seed, arm, label):
            sample = next(samples, None)
            p.require(sample is not None, 'Missing calibration measurement during replay')
            expected = dict(role=role, label=label, seed=seed, arm=arm,
                plan_digest=planner.role_digest(candidate), arrival_order=candidate['arrival_order'],
                iterations=cfg['iterations'], warmup=cfg['warmup'])
            p.require(all(sample.get(k) == v for k, v in expected.items()),
                      'Calibration candidate/seed/order does not match frozen planner replay')
            return sample['max_rank_elapsed_seconds']
        k = model['hidden' if role == 'attn' else 'ffn'] // world
        rebuilt = planner.fit_role(model['sequence'] * model['micro_batch'], model['hidden'], k, world,
            model['chunk_rows'], evaluate, config=cfg,
            seed=8000000 + 10000000 * role_index + config['seed'])
        p.require(rebuilt == plan['roles'][role], 'Measured calibration replay does not reproduce frozen role plan')
    p.require(next(samples, None) is None, 'Unexpected extra calibration samples')
    return evidence


def replay(root):
    root = Path(root)
    plan = p.read(root / 'plan.json')
    config = plan['config']
    p.require(config['protocol'] == p.PROTOCOL, 'Wrong protocol')
    p.validate_model(config['model'], config['world_size'])
    p.validate_runtime_environment(config)
    p.validate_calibration_options(config['calibration'])
    p.require(config.get('execution_plan') is None and config.get('execution_plan_sha256') is None,
              'The immutable base plan must not be rewritten after calibration')
    p.require(plan['jobs'] == p.jobs(plan['trials'], plan['blocks'], plan['scope']),
              'Frozen schedule changed')
    manifest = p.read(root / 'source-manifest.json')
    p.require(p.digest(manifest) == config['source_manifest_sha256'], 'Source manifest changed')
    for name, expected in manifest.items():
        p.require(p.sha(root / 'sources' / name) == expected, f'Frozen source changed: {name}')
    references = validate_reference_sources(root / 'sources')
    completed = p.read(root / 'completed-jobs.json')
    p.require(len(completed) == len(plan['jobs']), 'Incomplete worker sequence')
    samples = {'operator': [], 'train': []}
    trajectories = {}
    identities = None
    operator_identities = None
    environment = None
    native_identity = None
    rank_files = {}
    artifact_files = {}
    execution_plan = None
    calibration_rank_files = {}
    for job, done in zip(plan['jobs'], completed):
        p.require(job == done['job'], 'Completed job order/identity mismatch')
        directory = root / 'results' / job['directory']
        job_config = p.read(directory / 'config.json')
        p.require(job_config == p.config_for_job(config, job, execution_plan), 'Job configuration drift')
        command = p.read(directory / 'command.json')
        p.require(command.get('job') == job and p.sha(directory / 'command.json') == done['command_sha256'],
                  'Worker command identity drift')
        rows = []
        for rank in range(config['world_size']):
            name = f'rank{rank}.json'
            file = directory / name
            p.require(p.sha(file) == done['rank_sha256'][name], 'Rank result bytes changed')
            rank_files[str(file.relative_to(root))] = done['rank_sha256'][name]
            rows.append(p.read(file))
        rows = p.validate_rows(rows, job, job_config)
        current_environment = [r['environment_identity'] for r in rows]
        p.require(environment is None or environment == current_environment,
                  'Hardware/software identity changed between workers')
        environment = current_environment
        current_native = [dict(codec=r['codec_build'], gemm=r['native_build']) for r in rows]
        p.require(native_identity is None or native_identity == current_native,
                  'Native CUDA codec/GEMM changed between workers')
        native_identity = current_native
        artifact_files.update(validate_artifacts(root, directory, rows, job_config))
        if job['stage'] == 'calibrate':
            p.require(execution_plan is None, 'Repeated calibration in a frozen experiment')
            execution_plan = p.calibration_from_rows(rows, job_config)
            file = root / 'arrival-plan.json'
            p.require(p.read(file) == execution_plan, 'Arrival plan differs from calibration rank evidence')
            p.require(done.get('execution_plan_sha256') == p.digest(execution_plan)
                      and done.get('arrival_plan_file_sha256') == p.sha(file),
                      'Frozen arrival-plan hash changed')
            artifact_files['arrival-plan.json'] = p.sha(file)
            calibration_rank_files = {str((directory / name).relative_to(root)): value
                                      for name, value in done['rank_sha256'].items()}
            artifact_files.update(validate_calibration_evidence(root, directory, rows, job_config))
        elif job['stage'] == 'train':
            identity = p.training_identity(rows)
            p.require(identities is None or identity == identities,
                      'Training initialization/input changed between windows')
            identities = identity
            trajectories.setdefault((job['trial'], job['block']), {})[job['arm']] = rows
            samples['train'].append(dict(trial=job['trial'], block=job['block'], arm=job['arm'],
                role='full_step', value=max(r['elapsed_seconds'] for r in rows) * 1000 /
                config['timing']['timed_steps'], tokens_per_step=rows[0]['tokens_per_step'],
                mean_loss=statistics.mean(rows[0]['losses']), final_loss=rows[0]['losses'][-1],
                mean_grad_norm=statistics.mean(rows[0]['grad_norms']), final_grad_norm=rows[0]['grad_norms'][-1],
                peak_allocated_bytes=max(r['memory']['peak_allocated_bytes'] for r in rows),
                peak_reserved_bytes=max(r['memory']['peak_reserved_bytes'] for r in rows),
                native_workspace_bytes=max(sum(s['native_workspace_bytes'] for s in r['operator_states']) for r in rows),
                allocated_plus_native_bytes=max(r['memory']['peak_allocated_bytes'] +
                    sum(s['native_workspace_bytes'] for s in r['operator_states']) for r in rows),
                operator_scratch_bytes=max(sum(s['scratch_bytes'] for s in r['operator_states']) for r in rows)))
        elif job['stage'] == 'operator':
            current = p.training_identity(rows)
            p.require(operator_identities is None or current == operator_identities,
                      'Operator input/weights changed between windows')
            operator_identities = current
            for role in ('attn', 'mlp'):
                records = [next(v for v in r['operator_results'] if v['role'] == role) for r in rows]
                samples['operator'].append(dict(trial=job['trial'], block=job['block'], arm=job['arm'],
                    role=role, value=max(r['elapsed_seconds'] for r in records) * 1e6 /
                    config['timing']['operator_iterations'],
                    full_gemm_only_us=max(r['full_gemm_only_seconds'] for r in records) * 1e6 /
                    config['timing']['operator_iterations'],
                    bf16_relative_l2=max(r['bf16_relative_l2'] for r in records),
                    codec_reference_relative_l2=max(r['codec_reference_relative_l2'] for r in records),
                    artificial_bypass_ranks=[row['rank'] for row, record in zip(rows, records)
                        if record.get('arrival_diagnostic', {}) and record['arrival_diagnostic']['ready_bypass_observed']],
                    artificial_diagnostic_ranks=config['world_size'] if job['arm'] == p.ARRIVAL else 0))
    for group in trajectories.values():
        p.require(set(group) == set(p.ARMS), 'Incomplete four-arm training block')
        for other in p.FULL_TACO_ARMS[1:]:
            p.compare_taco_trajectories(group['flux_taco'], group[other])
    report = dict(protocol=p.PROTOCOL, run_id=config['run_id'], completed=True,
        build_jobs=config['build_jobs'],
        completed_scope=plan['scope'], hardware_target='RTX5090, single-node',
        standard_tp8_workload=(config['world_size'] == 8 and config['model'] == p.MODEL),
        implementation='Flux GemmV2 DP with modified CTA scheduling/publication and NCCL block transport',
        backend=p.BACKEND, uses_flux_gemm_templates=True, original_flux_kernel_unchanged=False,
        original_cuda_flux_reduce_scatter=False, producer_launches_per_call=1,
        native_gemm_family='FluxGemmV2-DP', cta_level_swizzle=True, environment_identity=environment,
        world_size=config['world_size'], model=config['model'], timing=config['timing'],
        runtime_environment=config['runtime_environment'], trials=plan['trials'], blocks=plan['blocks'],
        counterbalanced=p.balanced(plan['blocks']),
        ci_available=p.balanced(plan['blocks']) and bool(samples['operator']),
        short_run=not p.balanced(plan['blocks']), all_windows_retained=True,
        correctness_passed=True, full_quantized_trajectories_identical=bool(trajectories),
        trajectory_equivalence_arms=list(p.FULL_TACO_ARMS),
        formal_optimizer_steps=len(samples['train']) * config['timing']['timed_steps'],
        training_windows=len(samples['train']), rank_sha256=rank_files, artifact_sha256=artifact_files,
        native_builds_by_rank=native_identity, profile_requested=config['profile'],
        reference_sources=references,
        calibration_config=config['calibration'], calibration_rank_sha256=calibration_rank_files,
        arrival_plan=execution_plan_summary(execution_plan, config),
        execution_plan_sha256=p.digest(execution_plan) if execution_plan is not None else None,
        actual_kernel_overlap_validated=False,
        samples=samples, results={})
    for stage in ('operator', 'train'):
        if samples[stage]:
            report['results'][stage] = summarize_stage(samples[stage], stage, plan['trials'], plan['blocks'])
    return report


def markdown(report):
    lines = ['# RTX5090 四组实验', '',
        f"协议：`{report['protocol']}`；完成范围：`{report['completed_scope']}`。", '',
        '使用Flux GemmV2 DP模板主循环，修改CTA排程与完成发布；单次完整GEMM，并通过NCCL传输就绪块。'
        '不是原Flux ReduceScatter实现，也不宣称原kernel逐字未变。', '']
    if report['short_run']:
        lines += ['本次未完成八块平衡周期，仅用于短测；不计算置信区间。', '']
    lines += [f"完整训练窗口：{report['training_windows']}；累计正式 optimizer step："
              f"{report['formal_optimizer_steps']}（各窗口独立初始化）。", '']
    if report['arrival_plan']:
        lines += [f"独立到达校准计划：`{report['execution_plan_sha256']}`。", '',
                  '| 角色 | 到达生产顺序identity | 改变槽位 |', '| --- | --- | ---: |']
        for role, value in report['arrival_plan']['roles'].items():
            lines.append(f"| {role} | {value['arrival_identity']} | {value['arrival_changed_slots']} |")
        lines += ['', 'identity只描述生产排程；D仍采用每lane就绪消费，identity回退如实保留。'
                  '校准不计入正式operator/训练窗口；每个trial共用本批冻结计划。', '']
    for stage, trials in report['results'].items():
        for trial, roles in trials.items():
            for role, result in roles.items():
                lines += [f"## {stage} / {trial} / {role}", '',
                    f"单位：{result['unit']}。每窗口取所有 rank 最大墙钟耗时。", '',
                    '| 组 | 均值 | 中位数 | 最慢窗口 |', '| --- | ---: | ---: | ---: |']
                for arm, values in result['arms'].items():
                    lines.append(f"| {arm} | {values['mean']:.6f} | {values['median']:.6f} | {values['max']:.6f} |")
                if stage == 'train':
                    lines += ['', '| 组 | tokens/s | 平均loss | 末步loss均值 | 平均grad norm | 末步grad norm均值 | peak allocated GiB | peak reserved GiB | allocated＋native GiB |',
                              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
                    for arm, values in result['arms'].items():
                        lines.append(f"| {arm} | {values['tokens_per_second']:.3f} | {values['mean_training_loss']:.8g} | "
                            f"{values['mean_final_loss']:.8g} | {values['mean_grad_norm']:.8g} | "
                            f"{values['mean_final_grad_norm']:.8g} | {values['max_peak_allocated_gib']:.4f} | "
                            f"{values['max_peak_reserved_gib']:.4f} | {values['max_allocated_plus_native_gib']:.4f} |")
                else:
                    lines += ['', '| 组 | 独立完整GEMM均值 μs | 最大BF16相对L2 | 最大codec参考相对L2 |',
                              '| --- | ---: | ---: | ---: |']
                    for arm, values in result['arms'].items():
                        lines.append(f"| {arm} | {values['full_gemm_only_mean_us']:.6f} | "
                            f"{values['max_bf16_relative_l2']:.8g} | {values['max_codec_reference_relative_l2']:.8g} |")
                    lines += ['', '| 人工错峰诊断 | 观测到绕过的rank窗口 / 诊断rank窗口 |',
                              '| --- | ---: |']
                    for arm in (p.ARRIVAL,):
                        values = result['arms'][arm]
                        lines.append(f"| {arm} | {values['artificial_bypass_rank_windows']}/{values['artificial_diagnostic_rank_windows']} |")
                    lines += ['', '该诊断使用独立实例与人工lane延迟，不进入正式计时。未观测到绕过也如实报告；'
                              '观测到绕过不代表普通负载获得加速。']
                lines += ['', '| 前者 / 后者 | 配对下降 % | 95% CI | 更快块数 |',
                          '| --- | ---: | --- | ---: |']
                for pair, value in result['pairs'].items():
                    interval = ('不计算' if value['ci95'] is None else
                                f"[{value['ci95'][0]:+.4f}, {value['ci95'][1]:+.4f}]")
                    lines.append(f"| {pair} | {value['percent']:+.4f} | {interval} | "
                                 f"{value['faster_blocks']}/{value['paired_blocks']} |")
                lines.append('')
    lines += ['正下降率表示前者更快。CI按完整块bootstrap，未做多重比较校正；'
              'rank与窗口内step不作为独立样本。不同trial分别报告。', '',
              'loss和grad norm仅取rank0的正式步，不重复计TP rank；末步值先取每窗口最后一步，再跨窗口平均。'
              'operator张量scratch包含于Torch峰值，不重复相加；原生GEMM cudaMalloc工作区另按rank加至allocated后取最大。'
              'allocated与reserved不相加，CUDA context和NCCL内部占用未包含。', '',
              '完整GEMM独立计时不含通信/归约，不代替四组完整投影或训练比较。'
              '可选profile已核对文件哈希；通信event区间包含等待，报告不据此声称已验证kernel重叠。', '',
              '配置、源码快照、命令、逐rank结果、失败信息和可选profile保存在批次目录。', '']
    return '\n'.join(lines)
