#!/etc/profiles/per-user/study/bin/fish

set -x RUST_BACKTRACE 1
set -x RUST_LOG debug

nix develop .#z3-noodler --command fish -c "
cargo run --release -- get-solutions
cargo run --release -- get-solutions

cargo run --release -- run z3-noodler 0.0
cargo run --release -- run z3-noodler 0.5
cargo run --release -- run z3-noodler 0.9
"

nix develop .#z3 --command fish -c "
cargo run --release -- run z3 0.0
cargo run --release -- run z3 0.5
cargo run --release -- run z3 0.9
"

nix develop .#z3-noodler --command fish -c "
cargo run --release -- run z3-noodler 0.0
cargo run --release -- run z3-noodler 0.5
cargo run --release -- run z3-noodler 0.9

cargo run --release -- run z3-noodler 0.0
cargo run --release -- run z3-noodler 0.5
cargo run --release -- run z3-noodler 0.9
"

nix develop .#z3 --command fish -c "
cargo run --release -- run z3 0.0
cargo run --release -- run z3 0.5
cargo run --release -- run z3 0.9

cargo run --release -- run z3 0.0
cargo run --release -- run z3 0.5
cargo run --release -- run z3 0.9
"
