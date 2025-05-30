set -x RUST_BACKTRACE 1
set -x RUST_LOG debug

nix develop .#z3 --command fish -c '
    timeout --signal=KILL 30m cargo run --release -- get-solutions
'
nix develop .#z3-noodler --command fish -c '
    timeout --signal=KILL 10m cargo run --release -- get-solutions
'
