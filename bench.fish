argparse i/iteration -- $argv
or return

set -x RUST_BACKTRACE 1
set -x RUST_LOG debug

echo Arg is $argv[1]
exit 0

nix develop .#z3-noodler --command fish -c "
cargo run --release -- get-solutions
cargo run --release -- get-solutions

cargo run --release -- run z3-noodler 0.0 --iteration $argv[1]
cargo run --release -- run z3-noodler 0.9 --iteration $argv[1]
"

nix develop .#z3 --command fish -c "
cargo run --release -- run z3 0.0 --iteration $argv[1]
cargo run --release -- run z3 0.9 --iteration $argv[1]
"

nix develop .#z3-noodler --command fish -c "
cargo run --release -- run z3-noodler 0.0 --iteration $argsv[1]
cargo run --release -- run z3-noodler 0.9 --iteration $argsv[1]

cargo run --release -- run z3-noodler 0.0 --iteration $argv[1]
cargo run --release -- run z3-noodler 0.9 --iteration $argv[1]
"

nix develop .#z3 --command fish -c "
cargo run --release -- run z3 0.0 --iteration $argv[1]
cargo run --release -- run z3 0.9 --iteration $argv[1]

cargo run --release -- run z3 0.0 --iteration $argv[1]
cargo run --release -- run z3 0.9 --iteration $argv[1]
"
