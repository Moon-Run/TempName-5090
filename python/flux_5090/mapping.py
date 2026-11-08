"""Common physical-block and actual CTA order for the CUDA Flux producer."""
ARMS = ('flux', 'flux_taco', 'flux_taco_interleaved', 'flux_taco_interleaved_arrival')
ARRIVAL_ARMS = ARMS[3:]
BACKEND = 'flux-cutlass-dp-ready-nccl-v1'


def validate_shape(m, n, k, world, chunk_rows):
    if any(type(x) is not int or x <= 0 for x in (m,n,k,world,chunk_rows)):
        raise ValueError('Positive integer dimensions required')
    if world < 2 or m % (world * chunk_rows) or chunk_rows % 128 or n % 128 or k % 8:
        raise ValueError('Require M divisible by world*chunk_rows, chunk_rows/N by 128, K by 8')


def schedule(m, n, k, world, chunk_rows, arm, plan=None):
    validate_shape(m,n,k,world,chunk_rows)
    if arm not in ARMS:
        raise ValueError('Unknown arm')
    per = m // world // chunk_rows
    if arm in ARRIVAL_ARMS:
        from .planning import validate_role_plan
        validate_role_plan(plan,m,n,k,world,chunk_rows)
        order = plan['arrival_order']
    elif plan is not None:
        raise ValueError('Only the arrival arm accepts a role plan')
    elif arm == 'flux_taco_interleaved':
        order = [(slot % world) * per + slot // world for slot in range(m//chunk_rows)]
    else:
        order = list(range(m//chunk_rows))
    return [dict(slot=slot, physical=physical, destination=physical//per, within=physical%per,
                 row=physical*chunk_rows, rows=chunk_rows, lane=(physical%per)%2)
            for slot,physical in enumerate(order)]


def cta_mapping(m,n,k,world,chunk_rows,arm,plan=None):
    order = schedule(m,n,k,world,chunk_rows,arm,plan)
    rows, columns = chunk_rows//128,n//128
    if arm in ARMS[2:]:
        return [(order[round_*world+d]['physical']*rows+r,c)
                for round_ in range(len(order)//world) for r in range(rows)
                for c in range(columns) for d in range(world)]
    return [(item['physical']*rows+r,c) for item in order for r in range(rows) for c in range(columns)]


def wire_bytes(m,n,world,quantized):
    elements=(world-1)*(m//world)*n
    return elements//128*136 if quantized else elements*2


def split_sizes(world,rank,destination,elements):
    if not 0 <= rank < world or not 0 <= destination < world or elements <= 0:
        raise ValueError('Invalid collective dimensions')
    sends,receives=[0]*world,[0]*world
    if rank != destination: sends[destination]=elements
    else: receives=[elements if source != rank else 0 for source in range(world)]
    return sends,receives
