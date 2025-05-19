{
  lib,
  stdenv,
  fixDarwinDylibNames,
  cmake,
  ninja,
  catch2_3,
  python3,
  fetchFromGitHub,
}:
let
  mata = stdenv.mkDerivation {
    pname = "mata";
    version = "1.15.0";
    src = fetchFromGitHub {
      owner = "VeriFIT";
      repo = "mata";
      rev = "1.15.0";
      hash = "sha256-mo/E+gk/5/56GyWPNbza75hGkPhvvXiGg89htqB4nH8=";
    };

    buildInputs = [
      cmake
      ninja
      python3

    ];
  };
in
stdenv.mkDerivation {
  pname = "z3-noodler";
  version = "1.4.0-next";

  src = fetchFromGitHub {
    owner = "VeriFIT";
    repo = "z3-noodler";
    rev = "297779043f6709e07d1eb805084a4d7a2a441aa3";
    hash = "sha256-vILeAC415apzCR+l9AIGC10lEKmjXIDp0gBSMyRQ68c=";
  };

  nativeBuildInputs = [
    python3
    mata
  ] ++ lib.optional stdenv.hostPlatform.isDarwin fixDarwinDylibNames;

  enableParallelBuilding = true;

  configurePhase = lib.concatStringsSep " " [
    "${python3.pythonOnBuildForHost.interpreter} scripts/mk_make.py --prefix=$out"
  ];

  doChecks = false;

  checkPhase = ''
    make -j $NIX_BUILD_CORES test
    ./test-z3 -a
  '';

  preInstall = ''
    mkdir -p $dev $lib $out/lib
  '';

  postInstall = ''
    mv $out/lib $lib/lib
    mv $out/include $dev/include
  '';

  outputs = [
    "out"
    "lib"
    "dev"
    "python"
  ];

  buildInputs = [
    catch2_3
  ];
}
