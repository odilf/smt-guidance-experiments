set -x RUST_BACKTRACE 1
set -x RUST_LOG debug

set iteration 0
while test $iteration -lt 5
    for help in 0.0 0.9
        set command "cargo run --release -- run $help --iteration $iteration"
        echo Running `$command` !!

        # nix develop .#z3 --command fish -c "$command"
        nix develop .#z3-noodler --command fish -c "$command"

    end

    set iteration (math $iteration + 1)
end

exit 0
