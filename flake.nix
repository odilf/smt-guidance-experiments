{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    rust-overlay.url = "github:oxalica/rust-overlay";
    z3-noodler.url = "github:odilf/z3-noodler";
  };

  outputs =
    inputs@{
      nixpkgs,
      flake-parts,
      rust-overlay,
      ...
    }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
        "x86_64-darwin"
      ];
      perSystem =

        { system, ... }:
        let
          overlays = [ (import rust-overlay) ];
          pkgs = import nixpkgs { inherit system overlays; };

          packages = [
            pkgs.rust-bin.beta.latest.default
            pkgs.uv
            pkgs.python3

            pkgs.rust-analyzer
            pkgs.ruff
            pkgs.python313Packages.python-lsp-server
            pkgs.python313Packages.jedi-language-server
            pkgs.pyright
            pkgs.sqlite

            pkgs.sqlitebrowser
          ];
        in
        {
          formatter = pkgs.nixfmt-rfc-style;

          devShells =
            let
              z3-noodler = inputs.z3-noodler.packages."${system}".default;
              mkShell = pkgs.mkShell.override { stdenv = pkgs.clangStdenv; };
            in
            {
              default = mkShell {
                Z3_SYS_Z3_HEADER = "UNSET";
                packages = packages;
              };

              z3 = mkShell {
                Z3_SYS_Z3_HEADER = "${pkgs.z3.dev}/include/z3.h";
                packages = packages ++ [ pkgs.z3 ];
              };
              z3-noodler = mkShell {
                Z3_SYS_Z3_HEADER = "${z3-noodler}/include/z3.h";
                packages = packages ++ [
                  z3-noodler
                ];
              };
            };
        };
    };
}
