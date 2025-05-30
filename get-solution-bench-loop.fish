while true
    fish get-solutions.fish
    timeout --signal=KILL 1h fish bench.fish
end
